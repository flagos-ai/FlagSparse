# SpMV及mixed算子下一阶段优化：机制论证、实施门槛与实验闭环

日期：2026-11-19。目标库：FlagSparse-main/zyq_sparse。

状态：首批源码已接入（2026-11-19）：结构诊断、两个独立COO run-class候选、queue/sharded诊断细分及原测试联动。仅完成源码/AST/diff检查；GPU编译、准确性和性能均未验证。保留已有baseline修复及测试修改，不运行算子、pytest或离线编译，不暂存提交。后续尚未实施的方向仍为规划；本次交付及命令见第11节。

## 1. 本轮到底要提高什么

目标是提高固定类型、操作、布局、N和索引条件下，**全部算法加入后的逐矩阵最优性能**，并明确改善了哪些矩阵。不能只证明候选比另一个较慢的实验算法好。

当前证据截至11.17性能归档。11.18两个不规则候选尚缺通过smoke后的性能数据；11.19未发现新增实测目录。已有描述符修复仅静态审查通过，不能当作baseline已经恢复。

| 优先级 | 对象 | 已有证据 | 本轮目标 |
|---|---|---|---|
| P0 | baseline、smoke、比较边界 | 描述符已修源码；BF16窄输出数值/慢路径未定因；CSR SpMM转置计时边界曾改变 | 消除接口错误，建立能解释的对照，不调整容差 |
| P1 | SpMV CSR non短行 | scalar-bound/static已有明确贡献；mixed宽输出跨周自身约1.14–1.23× | 找稳定配置覆盖，保留旧实现；不再盲目增加预取变体 |
| P1 | SpMV CSR non长尾 | queue只在mip1有稳定局部价值；NNZ-local/grid尚未验收 | 检验定位/scan、队列调度和producer谁是瓶颈 |
| P1 | SpMV CSR trans/conj | 原生散射免重建带来约8倍自身改善 | 分清免重建与分片收益，控制清零/归约成本 |
| P2 | SpMV CSC trans/conj | scalar-bound有同批约20%的mixed整体增益 | 短列调参；长列方案以列结构证据为前提 |
| P2 | SpMV/SpMM COO | 通用segmented/stream是主力，mixed预取未证明普遍更好 | 检验是否能按局部row-run形态省掉不必要scan |
| P2 | SpMM CSR mixed | row_panel是主力，non跨周基本持平 | 优先控制转换/输出开销，再研究多面板的真实缓存复用 |
| P3 | SpMM CSC non mixed、SpMV CSC non | 分片有局部贡献；部分窄输出vendor异常 | 固定自身对照做S/H、初始化与输出归并消融，暂不叠加大算法 |

数据依据：[11.18优化对照表](../../../test_mixed_data/11.18data/analysis/progress_since_1111/IMPROVEMENT_TABLE.md)、[算法贡献报告](../../../test_mixed_data/11.18data/analysis/progress_since_1111/ALGORITHM_CONTRIBUTIONS.md)。路径按本文档所在目录定位；完整数据位于工作区test_mixed_data目录。

## 2. mixed为什么不一定比通用算法更容易提速

四种主测组合均为A/RHS同为FP16或BF16、计算及部分和FP32、输出同输入类型或FP32。变化主要在存储和输出，不是任意改变计算策略。

一个简化CSR SpMV逻辑流量模型为：

\[
B_{logical}=nnz(b_A+b_I+b_x)+(M+1)b_P+Mb_y.
\]

它假定每个非零元读一次x；没有表达缓存复用、真实事务粒度、atomic或额外缓冲，因此不能当作DRAM实测。int32列索引下，半精度的逐NNZ项约为2+4+2=8字节，FP32存储约为4+4+4=12字节；仅这个模型是1.5倍差别，不是存储减半就必然2倍提速，也不是任何硬件上的速度上限。int64索引下差别更小。CSR SpMV现有FP32路径计算FP64，不能将其与半精度FP32计算的速度差全归于存储类型。

| 成本 | 半精度输入后是否自动下降 | 推导出的优化重点 |
|---|---|---|
| 稀疏value、RHS加载 | 逻辑字节下降，实际事务未必同比下降 | 检查有效访存与缓存，不能凭类型断定带宽收益 |
| 列索引、指针、owner查找 | 基本不下降 | 减少重复定位/元数据处理可能更重要 |
| FP32累加、atomic、部分和 | 不下降 | 减少更新次数、中间缓冲和归并 |
| 排序、计划、launch和等待 | 不因输入半精度自动下降 | 必须测完整run，不能只宣传计算kernel |
| FP16/BF16输出 | 需要最终窄化；atomic仍应FP32 | 宽输出可直写；窄输出不能提前舍入 |

研究支持SpMV常受带宽和稀疏访问制约，但不证明本库每个case都带宽受限。[Bell与Garland的研究](https://research.nvidia.com/publication/2008-12_efficient-sparse-matrix-vector-multiplication-cuda)是分析依据，不是当前机器的profile结果。

## 3. SpMV：先验证已有候选，再决定新增计算体

### 3.1 短行：先解决配置范围，不再重复发明scalar/预取

保留`csr_row_scalar_bound`和`csr_row_static_tile`，不修改默认配置与旧kernel。配置变化使用原config，不为同一计算体重复注册名称。

第一组固定warps/stage，对scalar扫R=32/64/128；第二组对static固定K=16扫R=4/8/16，选定R后扫K=8/16/32/64；最后才比较后端允许的warp数。CUDA/ROCm分别得出结论，不共享最优配置。

对每个连续R行组定义长度l_r，用离线指标

\[
U_R=\frac{\sum_r l_r}{\sum_g R_g\max_{r\in g}l_r}
\]

描述scalar循环有效位置占比，R_g为该组有效行数；全空分母为0时记录N/A。它不是occupancy，也不表示内存合并效率。对static可补充真实块数ceil(l/K)与masked位置占比。

辩证判断：R增大能减少program数量，却放大同组长尾等待；K增大能增加行内并行，却增加短行浪费和归约。只有把这两个成本与完整时间对应起来，配置建议才有依据。若短行时间已接近launch/运行固定成本，继续复杂化kernel的收益空间很小。

目前已有`csr_row_scalar_tail_reduce`，先验证已有候选，不再新增同义“短行加长尾”算法。任何未来改变计算体的版本必须使用独立名称和旧实现对照。

### 3.2 长尾A：先验收csr_nnz_local_reduce_atomic

这是已接入源码的候选，不是本规划新发明。目标是wiki-Talk、Stanford等长行NNZ占比较高的矩阵；mip1、TSOPF作为结构不同的对照。

| 假设 | 怎么证明 | 反证与决策 |
|---|---|---|
| 旧NNZ算法owner搜索/scan过重 | 固定T，比较full/bounded，再比较single-row快路开关；看同owner tile比例 | 同owner比例高但仍慢，优先查atomic、随机x或资源占用，不继续堆定位技巧 |
| 只对跨tile行atomic能减少写竞争 | 固定其他配置比较boundary/all，结合模型atomic次数 | 更新数下降但时间不变，说明这部分未主导 |
| 按NNZ分工能改善拖尾 | 对照row_vector、queue及旧最佳，检查长行尾部kernel时长 | 只胜旧NNZ但不胜旧最佳，不计整体优化收益 |

首轮T=256固定；三项机制各单独比较后，才扫128/512/1024。动态快分支不保证更低寄存器占用，整个kernel可能仍按最复杂分支分配资源。

正确性关键：同一行不能一部分store、一部分atomic；完整行唯一写者，跨界行所有贡献atomic；重复指针使用upper-bound；复数分量和宽输出不提前窄化。先收集当前smoke的完整traceback，不能根据summary猜测数值或编译失败。

### 3.3 长尾B：静态grid先与persistent queue做单因素对照

`csr_longrow_grid_atomic`当前复用旧producer/partial/finish，只有consumer调度不同。必须先保持L/S、普通行配置、精度一致，比较两者。

记录expected_jobs/job_capacity。容量网格可能启动大量空program；persistent会付出任务领取atomic，但可能避免空网格。**静态网格不天然优于persistent，反之亦然。**普通Triton不能机械复制CUDA每线程工作窃取；适合先比较固定形状segment由硬件网格调度与有限persistent消费者。

### 3.4 有条件新增：csr_longrow_direct_atomic（暂定名，尚未实现）

仅当已验证producer/finish固定成本抵消consumer收益时推进，不能把它当作补救所有失败的默认下一步。

独立producer计算普通行，并对长行输出的计算精度累加区写零、生成任务；独立consumer直接向实际行位置累加，不再写紧凑partial后单独finish。两kernel在同一设备流顺序提交：producer完成后consumer开始；不能在一个kernel里假定跨program同步。

输出恰为计算类型且stride合法时直接使用out。半精度输出仍使用FP32缓冲，最后转换；FP32输入沿用FP64计算时也不能atomic到FP32输出。普通行与长行写者互斥、长行初始化先于消费，这是必须证明的不变量。

收益条件可写为：

\[
T_{saved\ init/finish/copy}>T_{extra\ producer\ stores}+T_{changed\ atomic\ locality}.
\]

此式是因果分项，不是用独立计时强行相减得到预测。直接按原行写入会比紧凑partial更分散，可能恶化局部性；消费者仍有atomic热点。先测试FP16/BF16→FP32与mip1，若完整run没有收益则停止。数值策略通过后再扩展regular和窄输出，未覆盖组合不可先声明支持。

### 3.5 CSR转置：先拆开免重建和分片的收益

继续使用`csr_transpose_sharded_atomic`，做三组同批对照：旧转置重建路径、同精度H=1原生散射、H=2/4/8原生散射。分别记录初始化、scatter、finish及完整时间。

只有H>1胜H=1，才能说分片降低冲突产生收益。若H=1已经最好，保留免重建方向，不继续增加H。额外空间随H×输出长度增长，可能使清零/归约占比高于atomic节省。

CSC non也是散射，但不能直接复制CSR转置源码或转换格式；先用现有CSC non作同类成本对照。

### 3.6 CSC trans/conj：短列保留，长列按证据迁移

短列对`csc_col_scalar_bound`和已有subgroup/vector做配置对照。统计必须用列长，而不是原矩阵行长。

若CSR NNZ-local在长行数据上通过机制和性能验证，且目标CSC中长列NNZ占比、同owner tile比例也满足条件，再设计独立原生CSC列NNZ候选：直接对CSC indptr定位列owner，计算A[r,c]×x[r]，写输出c，conj仅共轭A。完整列store、跨tile列atomic，不构造CSR。两种格式共享数学想法，不共享错误的转置生命周期。

若现有`csc_col_split_reduce`已更快，则没有必要为迁移本身再加一个名字。

## 4. COO：最值得预研的差异化候选

### 局部行段分类后选择归约方式

暂定独立名称`coo_runclass_atomic`（SpMV）、`coo_runclass_panel_atomic`（SpMM）；尚未注册，不能据此给运行命令。首版单tile，不同时叠加stream/carry或新的预取流水。

当前segmented与stream kernel对需要归约的普通块执行segmented scan。新候选按**当前输出索引**将一个tile分为三类：

| tile形态 | 判断 | 计算方式 |
|---|---|---|
| 所有有效NNZ同一输出行 | 所有有效owner均等于首owner | 直接tl.sum，实数每输出列一次atomic，复数两分量 |
| 没有相邻同行 | 每个有效位置与前一有效位置owner不同 | 不做scan，逐贡献atomic |
| 其余混合row-run | 存在相邻同行且不是全同行 | 沿用正确的segmented scan，段尾atomic |

**无序输入不能仅比较首尾owner判断整块同行。**例如[3,7,3]首尾相同，仍不是全同行。没有相邻同行也不等于全局owner唯一；非连续同一owner仍由atomic保证正确。trans交换坐标角色后重新判断，不能使用原row的分类。

普通Triton实现可用mask比较后tl.sum/tl.max产生program统一的分支谓词；SpMM对owner分类只在T轴做，不先扩展成[T,BN]判断。复数分别归约分量，部分和使用既定计算类型；半精度FP32缓冲与最终转换保持原契约。nonempty tile合法mask、空NNZ只初始化、N=0返回空输出。

成本判断：设三类比例分别为p_same、p_single、p_mixed。需要满足

\[
T_{classify}<p_{same}(T_{scan}-T_{sum})+p_{single}(T_{scan}-T_{direct})
\]

并覆盖混合分支额外控制/布局成本，才值得落地。这里只是结构性收益条件，不能把CPU估算比例当作GPU时间预测。

| 可能收益 | 可能反例 |
|---|---|
| 长连续row-run可绕过scan；随机分散输入可省无用scan | 大多数块都是中等混合run，分类本身成了纯额外开销 |
| 值和B面板加载仍复用 | 编译器保留最复杂分支资源，快路不一定降低寄存器 |
| 与dtype存储无关，可作为通用算法思想 | 不应只声明mixed；regular需按其原数值策略与能力逐项接入 |

先补离线块分类比例，再做编译/准确性样例。首轮配置与旧segmented一致，只比较分支选择。证明有收益后才考虑stream融合，避免carry类型/布局风险和归约变化一次叠加。

Triton提供associative_scan，但它不是零成本原语；接口说明不能代替机器码与布局成本测量。[官方scan接口](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)。

## 5. 其他mixed SpMM：先审计真实成本，再加计算体

### 5.1 CSR row_panel：重点保留，不重复造staged/prefetch

现有row_panel、staged、kparallel已有加载块、面板、累加器、panels等配置。不能仅把BN或累加器数换个值就称作新算法。

先在N=32以相同R/BN/A固定计算结构，测：原始半精度加载是否物化整份FP32副本、宽输出是否直接写out、窄输出是否重复分配/复制。这些已优化过的点要用实际执行证据确认，不预设仍有冗余。若发现实现缺陷，修复并记录前后版本；若只是另一种合法性能策略，保留对照配置或独立候选，不能改旧默认掩盖比较。

BN=16/32/64、A=1/2/4、加载块1/4/8单因素比较；记录寄存器、spill、实际stride。增大BN可能增加复用，也可能因寄存器使驻留下降；N=32时BN=64有一半mask，不能预设更好。

### 5.2 条件预研：输出面板的program顺序

仅N>BN、确有多个输出面板时，考虑独立`csr_row_panel_grouped`调度候选。计算体数学不变，明确改变行组×面板的program ID映射，保持每个输出tile恰好一次。原算法不变。

相邻稀疏行若共享B行索引，先处理同输出面板的一小组稀疏行可能提高B缓存复用；也可能打破A的局部性。N=32且BN=32只有一个面板，这个方法没有跨面板重排空间，**不应拿它解决首要N=32问题**。

先离线统计相邻行组的B索引重用比例，再在N=128/256看L2/DRAM事务。Triton官方GEMM教程提供program重排的思想，但稀疏B访问是否复用取决于索引，不能把稠密GEMM的百分比移植过来。[官方教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。

### 5.3 CSC non：先量化S/H的总成本

继续对`csc_col_partition_panel_atomic`做控制：固定H=1比较S，固定S比较H；然后独立比较BN。S改变单列的任务切分，H改变输出分片，作用不同。

若输出M×N较大且nnz小，H份缓冲初始化/归并可能主导。此时应保留H=1/直接合法计算类型out的路径，不增加预处理队列。若长列计算拖尾且输出热点明显，才继续增加S/H。输出FP16/BF16不能直接作为FP32 atomic缓冲，不能靠提前窄化提速。

SpMV CSC non也按同一“散射竞争 vs 输出缓冲”问题分析，但N=1的成本比例不同，不能从SpMM结果直接推出参数。

## 6. 研究思想取舍：为什么不直接照搬CUDA论文

| 思想/工具 | 可以借鉴 | 不能机械移植 |
|---|---|---|
| NNZ均衡、merge分工 | 限制每个任务的工作量，避免一条超长行或大量空行拖垮任务 | 每NNZ全局二分不等于merge-path；CUDA每线程串行merge不自然对应普通Triton张量program |
| persistent任务消费者 | 一些容量过大的任务空间可减少空program | 任务领取atomic和串行控制不免费；静态网格可能更合适 |
| 半精度存储 | 缩减value/RHS字节，加载时提升保持数值策略 | 索引、FP32 atomic和调度成本不会自动减半 |
| cache modifier | CUDA上可做受控访存实验 | PTX cache选项不是所有HIP后端相同语义；首轮不新增跨平台默认cache策略 |
| 更大tile、更多预取 | 可能提高独立访存数量 | 活跃寄存器和布局转换可能抵消，甚至spill；不是默认正向变量 |
| Tensor Core | 确有可复用稠密块时可能有效 | 任意稀疏单向量不能只因FP16输入就获得GEMM式吞吐；本轮不纳入 |

Merge-based作者给出的关键对象是行边界与NNZ逻辑合并流的均衡分解，而非简单按行或仅按NNZ切片。当前NNZ-local只有后者及局部快路，不能称为完整复现。[作者论文说明](https://mgarland.org/papers/2016/spmv/)、[作者实现](https://github.com/dumerrill/merge-spmv)。若空行/owner定位仍主导，再给出M+NNZ co-rank与重复指针的完整映射证明，暂不启动第三套大执行计划。

缓存修饰的后端语义见[Triton load文档](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。源码支持某语法，不等于当前Triton3.6和设备组合已编译通过；功能落地必须在目标版本验收。

## 7. 需要哪些数据，哪些已有，哪些需补

| 数据 | 当前状态 | 用途 |
|---|---|---|
| ms、gpu_ms、process_cpu_ms、prepared_run_ms、诊断process/compute | 现有入口已有；逐路径核对填充 | 判断完整run与主机提交/分配成本；不强行用两者相减作为CPU算法时间 |
| regular/mixed类型、opA/opB、layout/stride、索引、版本、输入指纹 | 已有部分字段，缺失必须显式标记 | 防止错误配对和跨周归因 |
| CSR行长分布、长行NNZ占比、同owner tile比例、混合tile跨度、预期job比例 | 前轮已补CPU结构诊断，需收到实际CSV | 为NNZ-local/grid确定目标集合 |
| scalar局部U_R、COO三类tile比例、CSC列结构与输出行热点、SpMM相邻行索引复用 | 下一步需增量补充到现有诊断helper；不宣称当前CLI已输出 | 验证本规划的新假设 |
| queue init/producer/consumer/finish、shard init/scatter/finish | 独立诊断逐路径确认；缺失再补 | 找出固定成本，producer同时做计算，不能全算process |
| 寄存器、spill、L2/DRAM事务、atomic、launch间隙 | 需计算节点profiler，只选代表矩阵 | 区分访存/控制/竞争，不给30矩阵全开profile |
| actual_jobs、实际跨界计数 | 只在独立诊断读回；正常run不新增同步 | 校验预测，不把expected当实际GPU计数 |

完整采样不能插入上述细分事件。始终保留ms=process_cpu_ms+gpu_ms；GPU计时与同步墙钟分别报告，不简单相加。gpu_ms包含完整事件区间，不能称为仅算术指令时间。逻辑GB/s是模型，不是显存利用率。

## 8. 实验顺序、规模与停止条件

### 第一道：可用性与数值

1. 复测已修复的descriptor：原72例pytest与144条件矩形diagnostic，命令见[MIXED_PRECISION_VENDOR_REFERENCE](../MIXED_PRECISION_VENDOR_REFERENCE.md)。
2. 收集并修复NNZ-local/grid smoke首个失败的traceback；不根据summary推定原因。
3. 原quick覆盖六种输入dtype（其中regular为FP32/FP64/complex64/complex128）及四种半精度组合、算法实际声明op、混合索引、out/stride和参数冲突；SpMM覆盖opB。先小样本、再一个真实矩阵，再全30矩阵。
4. 不修改共享容差；输入/参考溢出、算子误差、vendor误差分开。时间有效的FAIL仍进入用户要求的性能统计，同时单列新FAIL/旧PASS，不能据FAIL性能推荐部署。

### 第二道：机制实验

先选wiki-Talk、Stanford、mip1、TSOPF，加roadNet、ecology短行负对照及一个规则中长行矩阵。FP16→FP32和BF16→FP32优先，FP64作为不同计算/atomic成本对照。窄输出随后验证输出处理成本，不从宽输出结果直接推断。

| 实验 | 只改什么 | 需要固定 |
|---|---|---|
| scalar/static | R，再K，再warps | 输入、类型、输出、op、其他launch参数 |
| NNZ-local | owner_search、single_row_fastpath、atomic_mode依次 | T和warps，最后才扫T |
| queue/grid | consumer调度 | producer、L/S、精度、partial/finish |
| native CSR trans | H=1/2/4/8 | 相同原生scatter路径和计算精度 |
| COO run-class预研 | 无分类/分类；三类合成数据 | 单块，不叠stream预取；固定T/BN |
| CSR panel | 单独改变加载、BN或累加器 | 固定其他值，按实际stride分组 |
| CSC partition | S、H、BN依次 | 每次只改变一个，包含H=1对照 |

### 第三道：完整矩阵族收益

两种宽输出先测全30矩阵non/int32；SpMM先N=32，再N=8/128。机制成立后扩展全部四组合、regular、trans/conj、opB和布局。CUDA和ROCm分别统计；当前DTK vendor不支持时报告自身改善，不伪造厂商加速比。

每配置至少三个独立批次，旧、新交错顺序；预热/JIT在采样外。沿用原过滤规则，额外保留样本数/中位数/CV用于波动审查。一次用于选配置，后续独立批次验证，避免在同一批噪声上既选赢家又宣布提升。

令o_i为同批旧算法集合最佳时间，n_i为候选时间：

\[
g_i=o_i/\min(o_i,n_i),\quad GM_{family}=\exp(\mathrm{mean}(\log g_i)).
\]

报告候选/旧最佳的直接比值、加入后的全体GM/AM、预先按结构定义的目标子集GM、胜率、最差回退、准确性状态。有限算法集合的事后最佳不等于默认auto，也不等于已经实现自动路由。

| 决策 | 标准 |
|---|---|
| 继续优化 | 在独立重复中稳定胜过旧最佳；机制指标方向与假设一致；目标子集提升约1.2–1.3×可作为投入参考 |
| 保留为局部候选 | 少数矩阵明显改善且全体最佳有增益；明确适用结构，不切默认 |
| 停止叠加复杂度 | 只胜较慢实验算法，或只快compute但完整run不快，或优势小于实测波动 |
| 暂停性能结论 | 新编译/正确性故障、输入/计时不一致、baseline失败未区分 |

上述倍率是工程投入门槛，不是预测或承诺；不为了过线筛掉不利矩阵。

## 9. 实施任务与原框架接入

| 阶段 | 修改范围 | 交付物 |
|---|---|---|
| A：先收尾 | 现有baseline helper、原CI/accuracy；已有NNZ-local/grid对应文件 | descriptor复测、smoke错误根因与修复记录 |
| B：补可解释数据 | tests/benchmark_utils.py、原六个test入口及现有CSV字段 | U_R/COO块形态/CSC结构；诊断与完整计时隔离 |
| C：已有算法配置验证 | 原config、现有pytest与benchmark透传 | 有限配置表与同批消融，不改默认 |
| D：门槛通过后新增 | 在现有kernel/config/runtime中加入独立候选，不新增Python文件 | 优先COO run-class；direct-longrow和CSC长列迁移按证据选其一，不同时铺开 |
| E：联动与记录 | 原注册、规格、公开清单、支持表、pytest、runner、本文档 | 每个新名可查询并到达自身实现；dtype/op/后端如实声明 |

COO新算法共享局部分工思想。首批按用户确认同步接入SpMV与SpMM独立名称、实现和测试，注册的能力声明不等于实机验证通过，均标记unverified。改变计算方式使用独立名称；仅配置消融使用旧名称+config。一般性接口错误修复直接修复并记录版本，不制造“修复算法”名字。

保留main原prepare生命周期，不增加跨调用结构缓存、值缓存、图绑定或调优器。计划/初始化/转换/归并计入当前约定边界；专用config不得跨算法继承；同算法索引回退保留类型、op、out。

首批不再开发一组六个mixed预取版本、不改NNZ compact/fixed、不展开Tensor Core/INT8/异类型输入。尚未实现的候选不提供伪CLI命令。每阶段完成源码接线后再核对命令；CUDA沿用tools/run_cuda_tests.sbatch，计算节点用bash，登录节点提交用sbatch，保留cuSPARSE。DCU走原runner，明确baseline支持状态。

## 10. 每轮返回数据后必须回答的问题

1. 新算法加入后，每个固定条件的整体最佳GM提高多少，哪些矩阵贡献最大？
2. 是kernel变快、减少了运行固定成本，还是baseline变化？分别用自身耗时与vendor耗时证明。
3. 与假设相符的结构/phase/profile证据是什么？若不符，哪一项被证伪？
4. 候选是否新增准确性问题，旧默认是否回归？宽输出有没有中途窄化？
5. 下一轮是调参数、消除一个已定位成本、保留局部候选，还是停止？不能只重复“需要进一步优化”。

原始数据与分析输出继续放在所属日期目录，保留来源、版本、输入指纹、完整配置与负结果；不混用跨日期最小时间或静默回填vendor。

## 11. 首批源码交付：COO run-class与可解释诊断

### 11.1 实际变更与不变项

基线commit为`7ef29d7`，分支`zyq_sparse`。开始时工作区已有cuSPARSE描述符修复、相关测试及本文档；均保留。此次未新增Python文件。旧COO segmented/stream/prefetch JIT、CSR所有旧JIT及sharded JIT的AST计算体与HEAD逐个核对不变；默认算法和默认配置没有切换。

| 项目 | 实现位置 | 交付内容 |
|---|---|---|
| COO独立计算 | `src/flagsparse/sparse_operations/spmv_coo.py::_coo_runclass_panel_kernel` | 新kernel，SpMV为BN=1特化，SpMM为输出面板特化；没有调用旧scan/stream执行入口 |
| SpMV注册 | 同文件，`coo_runclass_atomic` | 原prepare/run、高层API及查询入口；无需新顶层函数 |
| SpMM注册 | `src/flagsparse/sparse_operations/spmm_coo.py`，`coo_runclass_panel_atomic` | 原注册表/route prepared；load模式保留B实际stride |
| 配置/能力 | `_spmx_column_config.py::resolve_coo_config` | 新算法独占classify配置；原算法合法值不变 |
| 结构模型 | `tests/benchmark_utils.py` | CSR组内利用率、COO原始owner分类、CSC列长和non输出贡献分布 |
| 运行诊断 | `_spmv_csr_kernels.py`、`_spmv_csr_transpose.py`、`spmv_csr.py` | queue增加consumer/finish拆分；sharded增加init/scatter/reduce；只在诊断调用插入事件 |
| 测试交付 | 原两个COO accuracy、CSR accuracy、`test_spmv_csr_policy.py`、GPU CI策略 | 仍由原marker/runner收集；没有第二套入口 |

原benchmark和runner已按注册表扫描/透传，不复制实现、不改变默认runner交付。GPU CI的COO synthetic套件显式compare。两级公开查询API已有，不增加算法专属导出函数。`conf/operators.yaml`、`ops_support.py`增加新名称及未验证说明。

调用链复核还修正了regular SpMM COO入口一个遗漏：原`--config`已被解析，但没有传到route prepared。现已贯通CSV/synthetic/单矩阵三条分支，限定单个显式扩展算法，并将实际元数据写入原CSV。扩展算法的配置/类型故障不再吞成SKIP；明确后端能力缺失仍可SKIP。默认不传config的算法选择与计时不变。

### 11.2 分类的数学语义与成本

每个program处理一个连续NNZ tile；SpMM再乘一个输出列面板。令有效owner序列为`r[0:k]`，k>0：

1. `all(r[i]==r[0])`：沿NNZ维直接求和，一次atomic/输出列/实分量。
2. 否则，`all(r[i]!=r[i-1], i>=1)`：每贡献atomic，不执行scan。
3. 否则：分段inclusive scan，仅局部段尾atomic。

第一项优先，因此一个有效元素的尾tile属于整块同行。`[3,7,3]`属于第二项，不能用首尾相等判定第一项。第二项允许非相邻重复owner，atomic保证全部累加。第三项的同一owner可以有多个非连续段，分别atomic。跨program的重复坐标也全部保留。

分类只读取T个owner，不沿BN重复形成谓词。没有跨块carry、预取、全局排序、coalesce、owner-map或执行计划。trans交换坐标角色，conj仅共轭A。

| 类型 | SpMV乘加/atomic | SpMM乘加/atomic | 输出 |
|---|---|---|---|
| FP16/BF16 | FP32 | FP32 | 同类型或FP32 |
| FP32 | FP32 | FP64 | FP32 |
| FP64 | FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | FP64实虚分量 | complex64 |
| complex128 | FP64实虚分量 | FP64实虚分量 | complex128 |

这是当前COO路径的精度契约，不能拿CSR SpMV FP32→FP64策略混称。输出恰好为计算类型时，合法strided out可直接作为atomic缓冲；否则使用计算精度缓冲，最后转换/复制一次。初始化、必要转换及写回均计入run。SpMM支持opA×opB九种组合；索引沿用原校验及执行宽度限制。

理论收益是省去不必要的scan，**不是减少所有tile的atomic数量**：整块同行与原segmented都只写一次，收益候选是sum对scan的成本差；无相邻同行与原segmented都逐贡献写，收益候选是绕开scan。一般分段tile新增分类成本，可能退步。分类分支、整数归约、寄存器和代码体积均是代价，实机需要检验。

| 配置 | SpMV | SpMM |
|---|---|---|
| 默认T | 128 | 32 |
| 显式T | 128/256/512/1024 | 32/64/128 |
| 默认BN | 1 | min(16,覆盖N的最小二次幂) |
| 显式BN | 无该字段 | 1/2/4/8/16/32 |
| local_reduce | classify/segment/none，默认classify | 相同 |
| conversion_mode | 不提供 | load/materialize，默认load |
| warps/stages | 4/1，经原后端合法集合筛选 | 相同 |

`segment`强制走scan，`none`不执行分类及scan。它们是同一候选内部的消融，不新增公开名字。实际计算为FP64时检查FP64/FP64 atomic；classify检查scan和reduction；none不要求scan。能力未知明确排除，配置不合法报错，不偷偷改参数。新元数据实现版本为5，不增加设备计数回读。

### 11.3 新增诊断字段与边界

`--perf-metrics`的mixed组合CSV新增`coo_structure`、`csc_structure`；原`csr_structure`追加`scalar_groups`。原regular CSR入口复用同一CSR helper。**本次没有扩展regular COO/CSC的性能统计框架：这些入口仍不接受无type-combos的perf-metrics。** regular类型的计算正确性/性能可照原脚本测试，结构与dtype无关，可用同格式/op/输入顺序的mixed诊断作对照，不能把不同加载顺序的数据混用。

CSR逻辑组利用率：

\[
U_R=\frac{\sum_i l_i}{\sum_g |g|\max_{i\in g}l_i},\quad R\in\{32,64,128\}.
\]

尾组的`|g|`取有效行数；零分母输出null及原因。记录分子/分母，不把U称为硬件occupancy。COO按T=32/64/128/256/512/1024记录三类tile数量、tile占比、NNZ数及NNZ权重；原始输出owner顺序不排序，每批最多约2^20个NNZ，partial tile仍计一次。CSC记录原列长分布、阈值以上NNZ占比；non另记录每个输出行接收贡献数量分布。

全部结构分析在CPU、计时外完成，不供kernel消费。诊断不是预处理加速。

queue/grid现在都记录`init_ms/producer_ms/consumer_ms/finish_ms`。producer同时执行普通行数值计算和长行描述符构建，因此归入compute，不能全部称为process。sharded记录`init_ms/scatter_ms/reduce_ms`，三者都属于compute；H=1直接out无归并时对应阶段只含诊断空段开销。正常完整run不插入这些分段事件。

仍然`ms=process_cpu_ms+gpu_ms`；完整事件时间与分段和不要求相等。`prepared_run_ms`是单独的同步墙钟采样。不能用新增分段之和覆盖完整时间，也不能把等待归为CPU处理。

### 11.4 计算节点命令（CUDA，保留cuSPARSE）

以下在**已经分配的计算节点**运行。bash不申请新节点；登录节点应使用sbatch并提供站点所需资源参数。所有命令均保留vendor，未加入`--no-vendor`。先确认目录和GPU分配。

```bash
MATRIX_DIR="$HOME/run/matrix"
RESULT_ROOT="$PWD/11.19data/runclass_candidates"
test -d "$MATRIX_DIR" || { echo "Missing matrix directory: $MATRIX_DIR"; exit 1; }
mkdir -p "$RESULT_ROOT"
COMBOS="float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32"
WIDE="float16:float16:float32,bfloat16:bfloat16:float32"

# 1. 新算法smoke：六种输入类型、四组合、所有op及边界/控制组。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_coo,spmm_coo --phase accuracy --mode quick \
  --pytest-args "-k runclass -x" --results-dir "$RESULT_ROOT/smoke"

# 2. 原marker完整quick，验证旧/新及诊断受影响路径；通过后将quick改normal。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmv_coo,spmm_coo --phase accuracy --mode quick \
  --results-dir "$RESULT_ROOT/accuracy_quick"

# 3. 首批真实矩阵对照，FP32宽输出、non、SpMM N=32。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_coo,spmm_coo --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos $WIDE --ops non --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics" \
  --op-benchmark-args "spmm_coo=--op-b non --layout row --dense-cols 32" \
  --results-dir "$RESULT_ROOT/wide_non"

# 4. 四组合及完整op；SpMM同时覆盖opB与两种布局。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_coo,spmm_coo --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos $COMBOS --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --perf-metrics" \
  --op-benchmark-args "spmm_coo=--op-b all --layout all --dense-cols 32" \
  --results-dir "$RESULT_ROOT/mixed_all_ops"

# 5. regular四种类型，原性能入口；opB扩展由上述mixed与原accuracy覆盖。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_coo,spmm_coo --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --dtypes float32,float64,complex64,complex128 --ops non,trans,conj --index-dtypes int32,int64 --timing" \
  --op-benchmark-args "spmm_coo=--layout all --dense-cols 32" \
  --results-dir "$RESULT_ROOT/regular_all_ops"
```

第4步是长批次，必须在smoke及一个真实矩阵通过后执行。可先把`MATRIX_DIR`替换成一个实际存在的`.mtx`文件路径再运行第3步（不要对文件使用上面的`test -d`）。normal准确性扩展原有索引/尺寸；性能需要扩N时将dense-cols改为`8,32,128`，仅type-combos入口支持列表。

需要收齐六算子的结构诊断时，使用第3步参数，将ops改为`spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc`，并追加：

```text
--op-benchmark-args "spmm_csr=--exclude-tle --op-b non --layout row --dense-cols 32"
--op-benchmark-args "spmm_csc=--op-b non --layout row --dense-cols 32"
```

trans/conj另批执行，不把non结构拿来解释转置输出owner分布。cuSPARSE baseline独立检查；精度FAIL保留有效时间和加速比，接口异常保留traceback，库不可用/明确不支持只跳过baseline。BF16异常仍不能因本次源码接入就宣称解决，描述符复测命令见`docs/MIXED_PRECISION_VENDOR_REFERENCE.md`。

### 11.5 消融命令与收敛规则

以下每个算法配置单独CSV，先固定T/BN，仅变分类方式。SpMM固定load；与旧`coo_segmented_panel_atomic`比较时也显式load，避免转换方式成为混杂因素。

```bash
for MODE in classify segment none; do
  CFG="{\"block_nnz\":128,\"local_reduce\":\"$MODE\"}"
  bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
    --ops spmv_coo --phase performance --mode quick \
    --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
    --benchmark-args "--type-combos $WIDE --alg coo_runclass_atomic --config '$CFG' --ops non --index-dtypes int32 --timing --perf-metrics" \
    --results-dir "$RESULT_ROOT/mv_$MODE"
  CFG="{\"block_nnz\":32,\"block_n\":16,\"local_reduce\":\"$MODE\",\"conversion_mode\":\"load\"}"
  bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
    --ops spmm_coo --phase performance --mode quick \
    --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
    --benchmark-args "--type-combos $WIDE --alg coo_runclass_panel_atomic --config '$CFG' --ops non --op-b non --layout row --dense-cols 32 --index-dtypes int32 --timing --perf-metrics" \
    --results-dir "$RESULT_ROOT/mm_$MODE"
done

# 旧SpMM同转换模式、同T/BN对照。
CFG='{"block_nnz":32,"block_n":16,"local_reduce":"segment","conversion_mode":"load"}'
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmm_coo --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--type-combos $WIDE --alg coo_segmented_panel_atomic --config '$CFG' --ops non --op-b non --layout row --dense-cols 32 --index-dtypes int32 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/mm_old_segment_load"
```

独立重复至少3批（改变RESULT_ROOT后缀），交错新/旧运行次序；不要覆盖CSV。先比较classify/segment/none，再固定模式扫T，最后改变BN或conversion_mode。scalar-bound R=32/64/128、static先固定K=16扫R=4/8/16，再固定R扫K=8/16/32/64，均使用原名字+显式config，不制造同义新算法、不改变默认值。

第一轮要回答：COO三类NNZ权重是否与收益一致；一般分段占比高时是否回退；分类模式是否超过**同批旧最佳**；加入算法族后全体GM、目标子集GM、胜出比例分别改变多少。CSR诊断用于决定NNZ-local或queue是否值得继续，sharded拆分区分初始化/热点atomic/归并成本。无实测收益不推荐默认配置；不因某慢旧算法被超越就宣称整体提升。

### 11.6 尚未完成的硬件验收

源码/AST/diff检查不是Triton编译验证。未运行任何算子、pytest、功能导入或离线编译；无CUDA/ROCm通过声明。原11.18 CSR smoke只有summary、不含失败traceback，尚不能定位或宣称修复；需保留该任务的pytest完整日志（不能仅summary）。这不阻断独立COO候选源码接入，但在CSR长尾性能结论前必须解决。

验收分三项独立记录：源码接线完成、实机正确性通过、性能假设成立。当前只有第一项完成；baseline描述符修复等待复测，BF16数值/慢路径仍待厂商诊断数据。

## 12. 11.19 数据反馈与 2026-11-21 定向修复

本节记录第11节之后的状态，原实验记录保留。修复前目标库为
`zyq_sparse`，HEAD=`37b9c3b4e7843e6528053be00de67b0d7b9bd1c4`，工作区干净。
数据分析在工作区 `test_mixed_data/11.19-data/analysis/REPORT.md`；
性能 FAIL 的有效时间仍保留，不能以整体最优有提升代替数值验收。

### 12.1 NNZ local 编译修复

`nnz_local_reduce_atomic_kernel` 的 single-owner 快捷路径是运行时分支：
快捷分支的 start/stop 是标量，分段分支的同名变量是 `[T]`。
Triton 3.6 在分支合流时拒绝形状不同的同名变量。11.19 的 quick 206 个失败、
normal 494 个失败均报告此错误，尚无该实现的有效性能结论。

将标量边界命名为 `single_start/single_stop`，向量边界命名为
`row_starts/row_stops`。仅隔离分支局部变量，不改变 upper-bound、分段 scan、
完整行 store、跨界 atomic、计算精度、输出缓冲、转置生命周期及计时范围。
仅该算法的 `implementation_version` 提升为2；所有默认配置和路由保持原样。

原 `test_spmv_csr_nnz_local_native` 增加 full-search + fast-path 的配置。
原准确性文件增加 `test_spmv_csr_nnz_local_branch_regression`：同一 launch
包含完整单行tile、1+127跨行tile、跨三个tile的长行、尾部padding和连续空行，
分别测试 bounded/full、boundary/all、BF16宽输出与complex128。
原case继续覆盖其它dtype、op、索引、fallback与out，不新增测试入口。
policy测试校验默认配置未改、显式fast/search参数完整传递、实现版本可查询。

### 12.2 scalar-bound/strip 长行误差：保留失败，补充可辨析证据

normal另有16个数值失败，均在BF16→FP32的第155行（长度8193），涉及
`csr_row_scalar_bound/strip`、默认/32行配置及四种索引组合。
记录的最大相对差约1.6182e-6，超过共享rtol=1.3e-6；不能简单归为容差问题。

本轮不改这两个旧kernel、不更换测试输入、不新增xfail/skip、不提高计算精度、
不放宽容差。原short-row测试失败后，针对最差的半精度→FP32行，在CPU用
同一存储输入诊断：顺序FP32累加、显式树形FP32累加、FP64参考、乘积绝对值和、
归一化误差及GPU结果是否与顺序FP32完全相同。数据写入既有pytest JSON的
`properties.short_row_numerics`，然后重新抛出原断言，结果仍为FAIL。
只在准确性失败后执行诊断，不进入算子或benchmark计时，也不修改公共接口。

若GPU匹配顺序FP32且树形误差明显改善，才有证据规划独立的长行数值稳健版本；
若不匹配，则优先检查实际乘加顺序、编译融合、地址和覆盖。CPU诊断不是
GPU指令序列证明，仍需计算节点验证。本轮不把尚未运行的诊断称为已定位数值根因。

### 12.3 后续验收与范围

先复测分支回归及原NNZ-local quick，再运行normal NNZ/short-row；normal不应
在第一个数值FAIL处终止，以保留所有失败诊断。NNZ编译/正确性通过后，再进行
一个真实矩阵及完整性能对照。原runner、marker、benchmark无需新增开关或入口，
CUDA继续通过既有sbatch包装脚本，完整命令仅在对话提供。

本地仅做AST与diff检查；未运行算子、pytest、功能导入或离线编译，未暂存提交。
环境挂载断连和cuSPARSE窄输出慢路径未因本次源码修复而得到解决；
COO转换/分类和CSR参数消融继续按原报告建议推进，不与编译修复混算收益。

## 13. CSR non 独立候选：csr_short_rowqueue_vector

### 13.1 优化对象与区别

11.19数据提示，CSR不应只增加超长行分段版本。mip1平均行长约155.77，
129–1024区间含有大量NNZ；旧1024阈值的长行队列只迁走约2.12%的NNZ。
wheel_601最大行长602，旧阈值不会迁走任何行；其64行组的逻辑槽位利用率约8.47%。
这意味着中长行拖住多行组也值得独立优化。neos3则是多数极短行、少数长行，
不能仅凭最大行长把整矩阵划为超长行负载。

本轮新增 `csr_short_rowqueue_vector`，不更改旧kernel、默认配置或路由。
其思路是**短行标量并行、超过阈值的整行排队后独立向量归约**。
与 `csr_longrow_queue_atomic/grid_atomic` 的区别是队列只有行号，不含分段描述符，
不分配数值部分和、不做浮点atomic、不启动最终归并kernel。
与 `row_vector` 的区别是空行和短行不各占一个program；与scalar-bound的区别是
超过阈值的行不再延长短行组的循环。它不是超长单行跨program并行方案。

### 13.2 执行步骤与正确性

1. 每次run创建一个int32计数器和int64行号队列，只清零计数器。
2. producer每program负责R个连续行。长度大于S的行用整数cumsum及每组一次整数
   atomic取得队列槽位；其余行以标量链计算。循环上限是本组普通行最大长度，不超过S。
3. consumer每program处理一条排队的完整行，以K个lane按实际行长循环，末尾归约写回。
   producer与consumer在同一实际执行流顺序提交，无设备自旋或跨program屏障。
4. 两条路径按 `length <= S` 与 `length > S` 完整互斥。空行由producer写零；
   所有输出仅有一个最终写入者，因此无需清零完整输出，也无需数值归并。

每条排队行至少含S+1个NNZ，队列容量上界为
`C = min(M, floor(nnz / (S + 1)))`。consumer启动C个program，在设备上读实际计数，
多余program立即退出；不为了缩小grid读回计数。C超过int32计数可表示范围时明确报错。
描述符峰值为 `8*C+4` 字节，数值工作空间为0；本算法的workspace限额同时约束描述符。
容量不是有效排队行数。benchmark的CPU结构诊断另外记录实际排队行/NNZ比例及
有效program/容量比，明确标为结构模型，不混称GPU运行时读回数据。

FP16/BF16乘法前提升FP32，部分和全程FP32；支持同类型或FP32输出。
FP32输入保持现有新CSR路径FP64计算，FP64保持FP64；complex64/128分别使用
FP32/FP64实虚分量。复数分量分别归约，直接按真实x/out stride访问，最终store才窄化。
长行采用lane子序列累加后归约，求和顺序不同于短行标量链，需沿用原容差验证。

### 13.3 配置、开销与验收

| 参数 | 默认 | 显式候选 |
|---|---:|---|
| rows_per_program | 64 | 32/64/128 |
| short_row_threshold | 32 | 0/8/16/32/64/128/256 |
| block_nnz | 128 | 64/128/256 |
| producer_num_warps | 2 | 已知合法launch集合 |
| num_warps | 4 | 已知合法launch集合 |
| num_stages | 1 | 1 |
| workspace_limit_bytes | 268435456 | 显式正整数 |

S=0是全部非空行排队的控制组。默认warp依现有后端能力选择合法保守值；显式非法
参数拒绝，不裁剪。首版仅声明CUDA/ROCm、non、六种dtype及四种半精度组合。
需要整数scan/atomic和归约能力；不要求FP32/FP64浮点atomic，FP64计算能力仍按类型检查。

代价包括计数器初始化、队列写入和第二次launch，以及保守容量中的空program。
均匀短行通常没有理由胜过无队列的scalar-bound；只有少数超长行时，一个consumer
独占一整行仍可能成为尾部瓶颈。大容量低利用率或超长单行应继续对照NNZ/split算法。
本算法的预期收益来自短行不等待中长行及更少的计划/数值归并开销，不能在实测前宣称提速。

计时仍为 `ms=process_cpu_ms+gpu_ms`，完整run含全部初始化、producer、consumer。
诊断 `init_ms` 归处理；producer同时分类与短行计算，记 `fused_plan_compute_ms`，
不能把producer全当纯分类时间。`compute_ms=producer_ms+consumer_ms` 仅是另次诊断，
不覆盖完整事件耗时；`finish_ms=0`。

原API、compare、runner、支持表和原CSR准确性文件接入新名称。测试覆盖六种dtype、
四种半精度组合、合法混合索引、空行/零行矩阵、满队列、602/713中长行、单超长行、
S=0及阈值边界、重复run、宽输出、strided x/out、NaN覆盖、容量不足和同算法索引回退。
非non操作按规格明确拒绝；无新顶层API和新测试入口。

计算节点先运行新名称的小样本准确性，再以相同输入对照scalar-bound、row_vector、
旧queue/grid与新候选。先固定R64/K128比较S8/32/128/256，随后固定S比较K。
同时报告各算法自身收益与加入新候选后的逐矩阵最优GM、胜出占比及回退；
无有效baseline时只报告自身相对改进。保留精度FAIL的有效时间及独立准确性状态。
完整命令仅在对话提供。源码接入与静态审查不代表Triton编译、正确性或性能已通过。

## 14. 有界consumer独立算法：csr_short_rowqueue_strided_vector

### 14.1 数据依据和实施边界

11.21消融中，neos3在S32/S128均只有一条排队行，但启动容量分别为62273/15930；
wheel_601均有602条排队行，容量分别为65782/16828。分类相同而consumer时间下降，
表明空program调度成本值得单独验证。原算法S128在FP16/BF16宽输出下只让整个算法族
最佳GM提升约1.98%/1.87%，主要胜出集中于上述矩阵，mip1仍弱于旧分段队列。

新增 `csr_short_rowqueue_strided_vector`，保留 `csr_short_rowqueue_vector` 的kernel、
launch、默认S32和所有其他算法默认路由。共用不修改的短行producer及乘积helper；
新consumer和run函数独立。目标是减少按保守容量启动的空program，不宣称解决单条
超长行的跨program并行，也不引入任务领取atomic、GPU计数读回、跨调用缓存或调优器。

### 14.2 核心实现与正确性

队列容量仍为 `C=min(M, floor(nnz/(S+1)))`。新consumer实际grid为
`G=min(C, grid_blocks)`；C为0时不启动consumer。

每个program加载已经完成的producer计数Q，按 `slot=pid, pid+G, pid+2G, ... < Q`
处理队列。每轮加载一条完整行，重置实/虚两个lane累加器，按实际行长执行K-lane
加载与累加，然后统一归约并按实际out stride写回。slot、G、地址和行指针计算为int64；
队列计数仍使用已检查范围的int32。

对每个合法slot，`pid=slot mod G`、`round=floor(slot/G)`唯一，因此无重复或遗漏任务。
producer仅写短行，consumer仅写排队行；两者输出互斥，不需要浮点atomic或最终归并。
三阶段按同一实际执行流顺序提交，不使用自旋或跨program屏障。

该改动不减少行号队列的分配容量，不消除计数器初始化及producer开销。Q为0但C大于0
时仍启动G个空program。G太小会增加串行轮数；固定步长也可能在行长高度不均时形成
尾部等待。任务顺序由producer整数atomic决定，不能假定全局按行号有序。

### 14.3 配置、数值、接口与诊断

| 字段 | 新算法默认 | 含义 |
|---|---:|---|
| short_row_threshold | 128 | 沿用11.21较好阈值作为实验起点 |
| rows_per_program | 64 | producer每program行数 |
| block_nnz | 128 | consumer每行lane块长 |
| grid_blocks | 256 | consumer program上限，显式整数1–1024 |
| producer_num_warps / num_warps | 2 / 4 | 沿用后端合法launch筛选 |
| num_stages | 1 | 不新增流水因素 |
| workspace_limit_bytes | 268435456 | 仍约束8*C+4字节队列空间 |

主要性能候选为grid_blocks=128/256/512/1024；更小整数用于正确性边界及机制控制。
显式非法参数报错；`grid_blocks`不能传给旧算法。配置解析仅复用旧字段校验，不改变
旧算法解析结果，不跨算法继承prepared配置。运行不缓存任务队列。

仍仅声明CUDA/ROCm、non、FP16/BF16/FP32/FP64/complex64/complex128。
半精度计算和中间累加FP32，输出可为同类型或FP32；FP32输入计算FP64，FP64保持FP64，
complex64/128分别按FP32/FP64分量计算。宽输出不先窄化。合法索引、strided x/out、
原始存储重叠检查及同算法索引回退沿用主库契约。trans/conj明确不支持；不以旧转置
路径冒充新算法。原高层API、prepare/run、查询、compare和两级导出不新增专属函数。

新增元数据：`consumer_grid_limit`为请求上限，`consumer_grid_blocks`为实际启动数量，
`consumer_rounds_capacity=ceil(C/G)`为容量轮数上界（C为0时为0），不是实际任务轮数。
`task_acquisition=static_grid_stride`。regular和mixed CSR CSV均保存这些字段。
benchmark在已有CPU结构诊断中另算实际Q、有效program比例、每program任务数与最大轮数，
明确属于CPU结构模型，不在正常run额外同步GPU。复数沿用现有指标模型的限制，
若逻辑吞吐/带宽不可用仍记录原因及有效计时。

计时保持完整 `ms=process_cpu_ms+gpu_ms`，CPU process正常为0。init包含计数器清零、
必要视图物化；producer同时分类和短行计算，consumer属于compute；无finish。
完整测量与诊断仍是独立run；开启timing会发生额外诊断执行及独立队列重建。
所有初始化、队列构建、计算和写回留在run内，不改变baseline和精度FAIL性能记录规则。

### 14.4 原测试链路与后续验收

原CSR accuracy参数化加入新名称，覆盖六dtype、四种半精度组合、合法索引、输出精度、
空维度、空行、长短行混杂、重复调用、非法配置、op冲突、存储重叠和安全索引回退。
新增grid边界用例使用Q=0/1/G-1/G/G+1/2G+3；G=1/8/256，normal增加1024。
使用不同长度及符号的行，检查每轮累加器清零、任务唯一性、strided输入/NaN输出、
返回对象身份及未改输入。测试内可在run结束后读回队列验证，不进入算子或性能计时。
启动捕获检查一次完整run及额外诊断分别创建新队列，实际grid不超过上限。

纯Python策略测试增加配置隔离、旧默认保持、能力筛选、无浮点atomic要求及结构模型。
原runner、marker、GPU CI和benchmark按已有注册机制枚举新名称，无新入口或Python文件。

性能实验须先固定S128/R64/K128，对照旧队列S128及新G128/256/512/1024。
不能只比较旧默认S32和新默认S128并把阈值收益归给有界grid。compare继续提供全部旧
算法对照；算法族旧最佳应同时纳入本轮旧S128控制结果，再计算加入新候选后的最佳GM。
regular四类型、mixed四组合分别测试，保留厂商baseline、吞吐/逻辑带宽和阶段诊断。
先小样本准确性，再单矩阵，最后30矩阵。胜出矩阵交错重复以排除11.21已观察到的批次差异。

源码审查基线为zyq_sparse的1ea86738d65f4f8ff2f02c3ab90b511b3542bcb6；初始工作区干净。
当前仅静态审查，Triton编译、准确性及性能均待计算节点验收。完整测试命令只在对话提供，
本轮不暂存提交，不切默认，不把容量缩小等同于硬件性能已提升。

## 15. 长行固定协作归约：csr_longrow_cooperative_reduce

### 15.1 与有界consumer同时保留，分别检验

本候选与第14节同时接入原算法族，不替换彼此或任何旧算法。第14节减少空consumer启动；
本节针对一条长行不能跨program并行的问题。二者均为CSR SpMV non独立算法，当前不扩展
转置生命周期。支持FP16/BF16同类型及FP32输出、FP32、FP64、complex64、complex128。
半精度计算/部分和为FP32，FP32输入按原新CSR策略使用FP64，复数使用对应分量精度。

11.21的BF16宽输出中，mip1原队列算法约0.2257ms，而整行队列S128约0.2625ms；
wiki-Talk原NNZ-local约0.2624ms，整行队列约0.3756ms。排队行分别承载约83.16%和83.55%的NNZ。
这些结果支持继续研究行内并行，不能据此认定根因已经是HBM、寄存器或atomic竞争。
目标必须由本次同条件实测确认，不把上述旧数据作为新实现收益。

### 15.2 执行与正确性不变量

每次run顺序执行初始化、producer、consumer、finish，使用当前执行流，kernel边界负责
跨program可见性，不使用自旋屏障。

1. 初始化一个int32队列计数器，分配int64行号队列及计算精度部分和。部分和使用empty，
   不做全容量清零；consumer必须写完所有被finish读取的槽位，包括零工作分区。
2. producer处理R行×V个lane。长度不超过L的行按实际组内最大普通行长度分块向量累加，
   最终沿V归约写输出；空行写零。长度超过L的行仅入队，整数scan和一次组级整数atomic
   取得独占队列位置，不计算这些行的乘积、不展开segment描述符。
3. consumer的grid为(G,P)，G=min(C,grid_blocks)。每个program处理队列位置
   pid0、pid0+G、…，固定pid1=p负责该行第p个协作区间。每次换行重置累加器。
4. finish的grid为(G,)，按相同队列步长遍历，加载每行P个部分和归约，按实际out stride
   写回。普通行和排队行的输出所有权互斥；每个队列位置只被一个finish program处理。

设一行非零元范围为[a,b)，K为block_nnz，Q=ceil((b-a)/K)。第p个协作分区为：

\[
u_p=a+K\lfloor Qp/P\rfloor,\quad
v_p=\min(b,a+K\lfloor Q(p+1)/P\rfloor).
\]

相邻区间端点连续，首端为a、末端覆盖b，各区间互斥，K块数量相差至多1。空区间写零，
其余区间从u_p按K递增并屏蔽v_p后的lane。不同program写PARTIAL[slot,p]独占位置；
复数分量分开存储。finish只读取slot<count的所有P个位置，因此不会读到未写的容量尾部。
所有指针、队列位置、chunk数量和乘法地址使用int64；int32只用于已检查上限的队列计数。

与旧方案的实质差别：整行vector没有行内跨program并行；旧longrow队列生成逐段任务并
通过浮点atomic合并；本候选只保存行号，段边界在consumer内算出，固定P槽位独占写入。
固定P牺牲可变段数的粒度适配，换取简单计划与无浮点atomic归并。

### 15.3 配置、空间与可观测信息

| 字段 | 起始值 | 合法范围/用途 |
|---|---:|---|
| long_row_threshold | 1024 | 整数0至1048576；性能优先256/512/1024/2048/4096 |
| cooperative_parts | 8 | 1/2/4/8/16；P=1是无跨program协作控制组 |
| grid_blocks | 128 | 1至1024；这是队列行维上限，不是总consumer数量 |
| rows_per_program | 8 | 4/8/16/32，producer同时处理行数 |
| lanes_per_row | 16 | 2/4/8/16/32/64，受subgroup宽度限制 |
| block_nnz | 128 | 32/64/128/256，consumer分区中的向量宽度 |
| producer_num_warps | 2 | 后端合法集合；也用于finish |
| num_warps | 4 | consumer合法warp数 |
| num_stages | 1 | 本轮固定 |
| workspace_limit_bytes | 268435456 | 描述符与数值临时空间合计上限 |

起始值未调优；显式非法参数报错。CUDA/ROCm能力来自现有对象，要求int64、归约、scan和
整数atomic；FP64计算按实际类型检查，不要求浮点atomic支持。

队列容量C=min(M,floor(nnz/(L+1)))。数值空间为C×P×每部分和字节数，描述符为8C+4字节。
复数部分和计两个实数分量。总量超限直接报错，不自动降低P、不隐式分批或换算法。
输出及已有输入不算临时workspace；resolve_conj/neg沿用原路径处理，必要物化仍在run内。

新增元数据包含cooperative_parts、row_grid_blocks、partial_slot_capacity，以及数值/描述符
空间。consumer_grid_blocks=G×P，consumer_grid_limit=配置grid_blocks×P；
consumer_rounds_capacity=ceil(C/G)，不是实际任务轮数。正常run不为统计读回GPU计数。
benchmark在计时外计算cooperative_queue结构模型：实际超阈值行数、NNZ占比、预计非空/空
协作任务数量及行维轮数；GPU入队顺序不确定，不根据CPU行序伪造worker尾时或占用率。

### 15.4 成本假设与验证顺序

可能减少：单条长行的串行计算尾部、逐segment描述符、任务领取atomic及浮点atomic竞争。
新增：P份行指针读取、C×P部分和空间、独立finish及空分区写零。P太大或长行太少且很短时，
固定成本可能占优；大量长行时G过小也可能形成串行轮次。普通行producer仍存在组内等待。
不声称此方案必胜NNZ-local或旧动态队列，更不把它当作所有不均匀矩阵的统一答案。

原accuracy新增全部8种输入/输出对、P=1/4/8（normal增加2/16）、空维度、无长行、单长行、
多轮队列、L/K/P分区边界及混合索引；使用NaN部分和检查空分区写零，并按行逐part检查
贡献范围。覆盖strided输入/out、宽输出、别名拒绝、容量不足、重复run、独立诊断和安全回退。
配置/能力/结构模型测试扩展原policy文件，原marker与runner入口不变。

先过smoke，再regular/mixed分别对全部矩阵compare，同时包含有界consumer和本候选。
固定L1024/G128/K128，只改变P=1/2/4/8/16；下一步固定有效P再改变L，不同时改全部参数。
重点分析mip1/wiki-Talk的consumer与finish，以及neos3/wheel等少量异常行；短行和小矩阵
继续记录回退。P>1必须胜过同候选P=1才能归因于协作；还须超越旧算法族最佳才算增加整体
覆盖。固定条件报告GM/AM、胜出比例、回退和准确性，FAIL的有效耗时按既有规则保留。
保留cuSPARSE baseline、吞吐及逻辑带宽，完整时间仍为cpu+gpu，诊断另行运行。

当前交付为源码接入，GPU编译、准确性和实际收益均未验证。沿用第14节基线commit，保留
已完成的有界consumer未提交改动，不改变旧计算体、默认配置、C API JIT契约或默认路由。
完整计算节点命令仅在对话提供，不新增Python文件，不暂存提交。
