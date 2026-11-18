# 极不均衡 SpMV：AlphaSparse、Triton 与当前实现的对照审查

日期：2026-11-18。目标：FlagSparse-main / zyq_sparse。源码基线：011c2c4c1bc6003471f02a4a602c2cc0d40f39f9，并包含工作区已有 scalar-tail 候选改动。

本次只做源码、既有结果和公开研究审查，没有修改 kernel，没有运行算子、pytest、功能导入或 GPU 编译，也没有暂存、提交。下述候选名称只是设计建议，尚未注册，不可据此直接运行测试。

## 1. 结论与证据边界

**极不均衡矩阵仍值得采用跨 program 的 NNZ 分工；当前失败不能证明这个方向不适合 Triton。** 已有 `legacy_segbin` 本来就是均分 NNZ、局部分段 scan、atomic 汇总，并在最近 mixed CSR non 120 个矩阵条件中胜出 12 次，regular 中胜出 16 次。`csr_longrow_queue_atomic` 在 mip1 的八种类型条件全部胜出。方向已有局部正证据。

但也不能据此断言只换一个 tile 参数即可成功。当前实现存在明确的不同成本：全范围逐元素行定位、所有 tile 都做 scan、额外执行计划、GPU→CPU 读回、多层归并、普通行生产阶段的低效分工、逐任务全局队列领取。需要区分这些成本，不能把它们统称为“负载均衡开销”。

此前新增的 scalar-tail 仍然是一组行/列由一个 program 独占的局部补偿算法。它不将单个超长行分给多个 program，不能代替本篇讨论的全局均衡算法。保留为短行占多数、局部中长尾的独立实验，不将其作为极端长尾的主方案。

证据分级：

- **源码已确认**：工作划分、读回、临时缓冲、kernel 数目、执行分支和计算类型。
- **测量已确认**：既有完整调用时间、单独诊断时间、矩阵行长统计和胜出条件。
- **尚未证实**：寄存器溢出、L2/显存命中、实际加载事务、atomic 串行化比例、occupancy。当前没有对应 profiler 证据。

## 2. 最近数据说明的两种长尾

数据依据：`test_mixed_data/11.17data/analysis/candidates_review/REPORT.md`、`new_candidate_structure.csv`、`new_candidate_per_matrix.csv`。下表均为 CSR non、FP16 输入/FP32 输出、int32 指针和索引；时间为完整 ms 换算为 μs，保留精度状态的原有统计规则，不代表这些记录全部准确性通过。

| 矩阵 | 平均行长 | 最大行长 | 行长 >1024 的 NNZ 占比 | 同批旧最佳 μs | queue μs |
|---|---:|---:|---:|---:|---:|
| mip1 | 155.77 | 66395 | 2.12% | 299.03 | 232.81 |
| wiki-Talk | 2.10 | 100022 | 53.87% | 329.44 | 769.02 |
| Stanford | 8.20 | 38606 | 29.52% | 163.59 | 226.68 |
| roadNet-TX | 2.76 | 12 | 0% | 89.99 | 610.02 |
| ecology1 | 5.00 | 5 | 0% | 87.49 | 472.61 |

1. **稀疏极端尾部**：mip1 的超长行 NNZ 占比很小，但拆分后仍显著改善。少数 program 的收尾可以决定完整 kernel 时间，不能仅按 NNZ 占比判断是否值得拆行。
2. **大量空短行与重型长行并存**：wiki-Talk、Stanford 需要同时处理普通行成本与长行分工。长行占比大并没有让现有 queue 自动获胜。
3. **负对照**：roadNet/ecology 没有超过阈值的长行。其 queue 融合 producer 诊断分别约 550.91/417.79 μs，不能将这部分回退归因于长行浮点 atomic 冲突。

queue 全 30 矩阵 FP16 宽输出的初始化诊断中位数约 57.34 μs，ordinary+plan 为 117.76 μs，consumer+finish 为 68.61 μs。它们来自诊断测量，不能相加替换完整 gpu_ms，更不能把其中一个阶段的改善直接称为整体收益。

## 3. AlphaSparse 实际有哪些相关思想

### 3.1 Flat：先均分 NNZ，再解决行归约

本地 CUDA/HIP `alphasparse_spmv_csr_flat.h` 的共同结构：

- 每块固定处理 1024 个 NNZ，GPU 分区 kernel 找到对应行范围，同时处理输出缩放。
- 主 kernel 连续读取 value/index，将乘积放到 shared memory。
- 归约端按行区间采用 scalar 或 vector 方式，局部行结果 atomic 到输出。

值得借鉴的是**连续数据加载与不规则归约解耦**。不能照搬的是 CUDA/HIP 线程编号、shared memory 数组、显式 shuffle/barrier。普通 Triton 应优先表达固定向量块、规约和 scan，由编译器映射布局；不能假定搬成同样大小的二维张量就复现了原 shared-memory 行归约。

Flat 的分区也有一次额外 launch。它不是“零准备成本”的性能证据，不能将 AlphaSparse 主 kernel 时间和我们完整 run 直接比较。

### 3.2 Merge：均分的是行边界与 NNZ 的合并序列

当前 HIP dispatcher 的 MERGE 实际进入 `spmv_csr_merge_ginkgo`，不是另一个 `merge.h` 实验实现。其工作总量取 M+NNZ，先求 merge-path 边界，再在 block 内对行边界与非零元流共同推进，处理局部和及边界 atomic。

关键区别：**纯 NNZ 均分不等于 merge-based**。大量连续空行也消耗行边界处理；merge 分解同时约束这部分工作。

当前 `csr_nnz_tile_reduce` 的逐元素二分 owner、分段 scan、独立部分和归并不是这篇算法的等价复现。其测试不能用来否定 merge-based。

### 3.3 Adaptive：不要混淆源码思想与当前分派

`adaptive.h` 包含经典 CSR-adaptive 的 row-block 描述与 stream/vector/长行分工，并有主机行指针处理和长行自旋协调；当前 HIP dispatcher 的 ADAPTIVE 则进入 `adaptive2.h`，其中先在 GPU 按行长分类，再启动不同计算路径。两者不可混称为同一个已启用算法。

可以借鉴有界工作块、同类任务分组和长行拆分；本轮不采用跨 program 自旋、不把 CPU 预处理移到计时外，也不重移植用户已经测试过的 TLE/AlphaSparse 算法。

## 4. 外部研究与 Triton 实现能提供什么

- Merrill/Garland 的 [merge-spmv 作者实现](https://github.com/dumerrill/merge-spmv) 将行边界与非零元编号合并后均分，目的包括避免单个长行或大量空行压垮一个处理单元。应迁移其工作量模型，而非直接复刻逐线程推进控制。
- [Hola-SpMV，ICS 2017](https://hpdgp.mpi-inf.mpg.de/pappe/HolaSparse.pdf) 在全局 NNZ 均分下做局部适应：特别区分一个 block 只涉及一行的情形，并使用边界 atomic。对我们最直接的启发是：整块只有一行时不必为每个元素查行号，也不必做分段 scan。论文中的寄存器交换、warp 级决策和共享内存技巧不是普通 Triton 可逐句对应的 API。
- [ModernGPU segmented reduction](https://moderngpu.github.io/segreduce.html) 明确提供带预处理复用的接口，并区分空段支持成本。这提醒我们必须把一次执行与重复结构的准备收益分开，不能拿复用后的结果证明每次重建计划也便宜。
- [rocSPARSE 7.1.1 Generic 文档](https://rocm.docs.amd.com/projects/rocSPARSE/en/docs-7.1.1/reference/generic.html) 区分 rowsplit、adaptive、LRB、nnzsplit，并指出准备成本与重复调用次数的权衡。这是版本化的设计参考，不代表当前 DTK 已具备这些路径或类型支持。
- 找到的 Triton 项目 [Btorch](https://criticality-cognitive-computation-lab.github.io/btorch/design/sparse_connectivity/) 文档讨论 row-block、hub 分段和无需主机同步的固定网格。其场景含稀疏脉冲向量，且文档说明 Triton kernel 为 FP32，其他类型可回退；这里仅参考设计文档，没有完成源码复核，不将其性能或 mixed 能力作为本库依据。
- [blksprs](https://github.com/FelixSchoen/blksprs) 面向块稀疏张量。块内稠密计算与任意 CSR 长尾不同；不能据其 Triton 实现就推断本库可用 Tensor Core 提速。

普通 Triton 可直接利用固定向量加载、`tl.sum`、`tl.associative_scan` 和 program 级统一分支。[scan API](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html) 不保证某个用户张量形状一定只用寄存器或固定几次 shuffle。[AMD Triton 优化指南](https://github.com/ROCm/triton/wiki/General-Guide-of-AMD-Triton-Performance-Optimization) 特别说明非 GEMM 内循环流水需使用相应 `tl.range` 机制，也需检查 spill；仅改 launch 的 num_stages 或手写“双块提前加载”不足以证明流水已实现。实际行为仍需核对计算节点 Triton 版本。

## 5. 我们现有实现的具体差异与代价

| 当前算法/路径 | 已实现的均衡 | 未解决或额外承担的成本 |
|---|---|---|
| scalar、subgroup、scalar-tail | 行组内向量化或尾部并行 | 仍由组内最长行控制；单超长行不跨 program |
| row_refill | 窗口内空槽领新行 | 剩余一个超长行时其他槽仍空闲；整数 scan 与状态更新有成本 |
| rowgroup_stream_reduce | 行组 NNZ 连续加载 | program 工作量仍由固定行组 NNZ 决定，非全局均分 |
| legacy_segbin | 固定 NNZ tile，原生 CSR，无全局 owner-map | 每个 NNZ 全范围二分；所有 tile scan；全部局部行段 atomic |
| NNZ compact | 固定 NNZ tile、独占部分和 | 动态压紧、计数 item、unique/repeat_interleave、分层归并 |
| NNZ fixed | 预定边界容量、取消动态压紧 | 仍有 row-wave `.tolist()`，边界 kernel、clone、缓冲初始化、多层 launch 与最后写回 |
| longrow_queue | 长行拆成有界 segment | 普通行 producer 的 [8,32] 分工、描述符、逐任务全局 atomic 领取、partial 初始化及 finish |
| 主库 ROCm prepared NNZ 特例 | 缓存 owner-map，减少 run 定位 | 增加每 NNZ 行号流；最后 tile 重算跨界行前缀，超长行收尾仍集中；不是每次 run 的公平对照 |

代码锚点：`spmv_csr.py::_spmv_csr_segbin_kernel`、`_spmv_csr_nnz.py::_run_mixed_fixed/run_mixed/_nnz_kernel/prepare`、`_spmv_csr_kernels.py::longrow_producer_kernel/longrow_consumer_kernel`。

避免重复发明：fixed 路径已经用首尾 owner 限定搜索；同 owner 时上下界相等，不执行逐元素搜索循环，但后续仍走分段 scan。下文 A 不把“限定搜索”本身作为相对 fixed 的新贡献；相对 fixed 的差异是融合定位、同 owner 普通归约以及删除描述符/分层归并。相对 legacy_segbin，限定搜索才是新增差异。现有 SpMM 的 NNZ 边界 atomic 也已有相关写回思想，本轮只是重新组织原生 SpMV 的实际成本，并非宣称发明全新的数学分解。

必须单独指出计算精度：`legacy_segbin` 的 FP32 输入采用 FP32 累加；目前新 CSR SpMV 候选的 FP32 输入采用 FP64。不能把二者 regular FP32 差距全部解释为调度。机制对照优先半精度→FP32、FP64、对应复数分量相同的条件；不通过改旧精度制造提速。

### 5.1 一个可计算但不是硬件利用率的模型

设组内行长为 l_i，R 行、每行 V 个逻辑位置，组共同循环，则：

`U_group = sum(l_i) / (R * V * ceil(max(l_i)/V))`。

全空组单独统计。该量解释 mask 浪费，不能称 occupancy 或显存利用率。refill 可以改善某些窗口，不能消除单行不可拆带来的最长临界路径。

NNZ 路径则应统计每个 tile 的首尾 owner、同属一行的 tile 比例、跨界行贡献数、局部行号跨度及空行比例。只看平均行长、最大行长或 >1024 NNZ 占比不够。

## 6. 优先方向 A：同 owner 快路与边界 atomic 的独立 NNZ 算法

建议设计名 `csr_nnz_local_reduce_atomic`，尚未注册。以 `legacy_segbin` 为直接对照，不将“均分 NNZ”冒称为全新思想。新差异是融合首尾定位、同 owner 专用规约、有界局部搜索及完整行直接写回；相对 compact/fixed，删除运行时边界描述符和多层归并。

### 执行

1. 先按计算类型清零输出累加目标。计算类型与输出相同且 out 合法时直接使用 out，否则使用计算精度缓冲。
2. Q=ceil(NNZ/T) 个 program，每个只负责 `[qT,min((q+1)T,NNZ))`。
3. 在 kernel 内对首尾有效 NNZ 做 upper-bound 查找，正确跳过重复行指针。
4. **首尾同 owner**：直接读取 T 个 value/index/x，提升后乘加，用 `tl.sum`；无需逐元素二分和 segmented scan。
5. **首尾不同 owner**：只在 `[first_owner,last_owner]` 范围搜索各元素 owner；固定向量做 segmented scan，只有局部段尾写结果。
6. 行区间完整包含在本 tile 时直接 store；跨 tile 行的所有段都 atomic。判定为 `row_start >= tile_start and row_end <= tile_end`，同一行不能同时出现 store 与 atomic。
7. 全部完成后才按目标输出类型转换。空行由初始化覆盖；尾 tile、复数分量、地址宽度和 strided out 明确 mask。

### 成本与可证性质

- 负载按 NNZ 有界，单长行跨多个 program，不需要动态全局任务队列。
- 旧 segbin 每 tile 逻辑上执行 O(T log M) 行定位。新同 owner tile 只需两个 O(log M) 边界查找；混合 tile 仍需 O(T log D) 局部定位，D 是首尾行号跨度。D 可能因空行很大，不能保证处处更快。
- 一个 NNZ tile 至多涉及两条跨 tile 行；同属一行只写一次。因此实数边界 atomic 次数不超过 2Q，复数按分量增加。该上界不代表无竞争；极长行仍会成为 atomic 热点。
- 相比 fixed，正常路径不需 owner 描述符、计数读回、部分和数组和多层归并。代价是计算精度 atomic、初始化和可能的输出转换，以及非确定的累加顺序。
- T 增大减少边界次数，但增大混合 tile 的 scan/寄存器负担；优先 256/512/1024 显式对照，不假定越大越好。
- 统一 kernel 的动态同 owner 分支可以减少实际执行指令，**不保证寄存器分配按快分支缩小**；编译器可能仍按复杂分支分配资源。

最有希望的目标是 wiki-Talk、Stanford：长行 NNZ 比例高且有大量完全落在长行内部的 tile。mip1 也测试，但其长行 NNZ 很少，混合 tile 定位可能抵消收益。roadNet/ecology 为负对照，保留已有短行赢家。

### 分段 scan 的语义

使用 `(value, head_flag)` 的结合运算：右操作数遇到新段时选右值，否则相加；flag 用逻辑或传播。只能对同一连续 row-run 求和，不能用全块前缀和相减替代，以免产生严重数值抵消。复数实虚分量分别处理。浮点加法仍有舍入顺序差异，沿用共享容差验证。

### 消融必须拆开

在同一个新候选内依次对照：全范围/首尾限定定位；强制 scan/同 owner sum；所有段 atomic/仅边界 atomic。每次固定 T、warps、计算精度，最后才改变 T。否则无法判断改善来自哪一项。

## 7. 方向 B：将长行队列改为有界静态任务网格的独立候选

针对 mip1 类“少量重尾”保留专用分工思路，但不先重做所有阶段。建议首先实现独立消费路径，保留旧 queue 完整对照：

- producer 仍建立长度至多 S 的 segment 描述符并写 GPU count。
- 主机由已有 M/NNZ 和 L/S 得到安全容量：`long_cap=min(M,floor(NNZ/(L+1)))`，`job_cap=floor(NNZ/S)+long_cap`。
- 直接启动 job_cap 个 program；每个读取 GPU count，编号越界则退出，合法任务计算一次。**不逐任务 atomic 领取，也不把实际 count 读回 CPU。**
- 已等长的 segment 交由 GPU 网格调度，避免额外建立软件调度瓶颈。

这不是声称静态网格一定胜过 persistent。容量比实际 job 数大时会启动很多空 program；persistent 更少的 block 可能反而获益。必须记录诊断用 actual_jobs/job_cap，先同 producer 对照调度，再决定是否优化 producer。

producer 的改进不能仅“换成 short scalar”了事：普通行可长至 L，scalar 对这些行仍会退化。下一步需把普通行计算成本与描述符生产拆开测量，再选择有界短行/向量组织；若额外 launch 抵消收益，则停止这个分支。不得在未解决 producer 回退时将其宣称为通用长尾替代。

当半精度→FP32 的输出可直接累加时，可另做消融：producer 对长行输出置零，consumer 直接 atomic 输出，省 partial+finish。程序必须同流顺序提交；旧 producer、旧 consumer 不修改，regular FP32→FP64 计算缓冲等情况单独处理。

## 8. 后续方向 C：真正的 M+NNZ 合并分工

只有 A/B 证据显示空行处理或局部 owner 定位仍主导，才推进真正 merge-path。

可迁移的是：program 对 M+NNZ 逻辑合并流固定预算，边界先做 co-rank，局部数据规模受限。难点是 CUDA 每线程串行 merge 在普通 Triton 中并不自然；若翻译成“每 NNZ 再全局二分”，就失去原设计目的。局部行界编码必须处理多个空行落在相同 NNZ 位置，简单 head 布尔标记会丢失行号跃迁。

因此该方向先要求给出重复行指针的映射证明、有限形状和临时量上界，再做原型；当前不承诺可直接移植，不把它作为本轮立即增加的第三个大算法。

## 9. Mixed 的差异及不能机械采用的优化

- FP16/BF16 减少 A/x 存储，但索引仍为 4/8 字节、地址和行定位仍有成本、累加与 atomic 通常仍为 FP32。元数据与调度相对更贵，所以首先减少计划和定位比单纯加半精度预取更值得验证。
- 把标量乘加改成 Tensor Core 需要可复用的稠密块/批次等条件；任意 CSR 单向量不能仅因输入半精度就获得 GEMM 式收益。
- A/value/index 连续不代表 `x[col]` 连续。负载均衡改善后若 x 的间接访存占主导，继续减少 scan 或扩大 T 可能收益有限；应测 L2 命中和实际事务。
- 更多预取提高 memory-level parallelism 的同时延长寄存器生存期，可能降低驻留或 spill。现有 mixed 专用版本没有普遍获胜，不能继续把预取数量当作默认正向变量。
- 新算法仍应是通用数据分工。各 dtype 按已约定计算精度适配；不再次把名称限定为 only-mixed。FP64 atomic 能力与成本必须单独检查。

## 10. 验证顺序与停止条件

### 先补少量诊断

在既有结构分析/测试入口补充，而不是另建 benchmark：

1. T=128/256/512/1024 的同 owner tile 比例、混合 tile 行号跨度分位数、边界段数量。
2. 对 queue 单独记录 init、producer、consumer、finish；actual_jobs 仅诊断读回，正常 run 不增加同步。
3. kernel-only 与完整 run 同时记录；正常性能采样不插入诊断事件，`ms=process_cpu_ms+gpu_ms` 不变。
4. 计算节点选取样本看 owner-search/scan、寄存器与 spill、global atomic、x 读事务、kernel 间空隙。逻辑带宽不能替代这些证据。

优先矩阵：wiki-Talk、Stanford、mip1、TSOPF；负对照 roadNet、ecology；再加一个规则中长行矩阵。先 mixed 两种宽输出与 FP64，再覆盖全部六种 dtype、半精度同输出及声明 op。CSR transpose 保持现有执行矩阵生命周期，不宣传为转置优化；后续若迁移至 CSC，必须直接按 CSC 列结构实现 trans/conj。

### 正确性先于收益认定

重点验证空行、重复行指针、尾 tile、单行跨数百 tile、同一行所有贡献都走 atomic、内部行 store 与边界 atomic 不混写、复数、NaN out 与合法 stride。精度 FAIL 的有效性能继续保留，但不能作为可部署优化通过的依据。已有新 FAIL/旧 PASS 的案例独立审查，不调整容差。

### 评价算法族，而非只评价候选自身

每个固定类型/op/索引条件，令旧算法集合最短时间为 o_i，新候选为 n_i，则加入后的改进为 `g_i=o_i/min(o_i,n_i)`，全体收益为 `exp(mean(log(g_i)))`。仍报告候选自身时间、胜出比例及严重回退，至少三轮交错顺序复测，避免用小于测量波动的优势定义可靠赢家。

先通过小样本，再扩展全部 30 矩阵。若候选仅胜过旧 NNZ 却从未胜过同条件旧最佳，则它是机制实验进展，不是整体算子提升。若 A 的同 owner 快路确有 kernel 收益但完整时间没有收益，先处理运行固定成本；若完全同 owner 的长行 tile 仍无收益且 profiler 显示 atomic/随机 x 主导，不继续盲目加预取或扩大 tile。

## 11. 推荐决策

1. 保留 scalar-tail 作为局部补偿候选，明确其不是全局长尾方案。
2. 优先落实 A 的独立实现与三项机制消融。直接对照已有 legacy_segbin、NNZ fixed/compact 及同批旧最佳；它回应了当前已确认的定位、scan、归并成本。
3. B 先做消费调度与输出归并的受控实验，服务 mip1 型少量极端尾部；未解决 producer 前不扩大目标。
4. C 为后备研究，不能以“merge”命名包装普通全局二分 NNZ tile。

本次没有新增算法注册。下一轮实施仍需保持原算法及默认路由，接入原 benchmark/pytest/runner，并分开报告源码接通、硬件正确性与实测收益。

## 12. 下一轮实际落地计划（2026-11-18）

状态：待实施计划。本节确定第一批实现范围；不是已有 API 或已验证支持声明。

### 12.1 本轮范围与顺序

| 优先级 | 独立名称 | 交付目标 | 首批目标矩阵 |
|---|---|---|---|
| P0 | 现有 benchmark 的结构诊断 | 验证 NNZ tile 快路覆盖率和队列容量浪费 | 全部 30 矩阵，可离线计算 |
| P1 | `csr_nnz_local_reduce_atomic` | 无全局计划的 NNZ 分工、同 owner 快路、边界 atomic | wiki-Talk、Stanford、TSOPF，兼测 mip1 |
| P2 | `csr_longrow_grid_atomic` | 与旧 queue 相同 producer/partial/finish，仅用静态容量网格消费 | mip1、Stanford、wiki-Talk |

保留现有 scalar-tail 改动和所有旧算法/默认选择。P1 形成完整源码交付后再接 P2；P2 首版不优化普通行 producer，不省略 finish，避免一次改变三个因素。真正 merge-path、输出分片、producer 改写、Tensor Core、结构缓存及自动调优均不在本轮。

目标库仍为 FlagSparse-main/zyq_sparse。仅修改现有 Python 文件及相关文档，不新增 Python 文件；不暂存、不提交。本地仅源码、AST、diff 检查，不运行 pytest、算子、功能导入或离线编译。

### 12.2 类型、操作与后端契约

| 输入 A/x | 计算与 atomic 类型 | 合法输出 |
|---|---|---|
| float16/bfloat16 | float32 | 同输入类型或 float32 |
| float32 | float64 | float32 |
| float64 | float64 | float64 |
| complex64 | 两个 float32 分量 | complex64 |
| complex128 | 两个 float64 分量 | complex128 |

P1 声明 non/trans/conj，后两者复用当前 CSR 按次重建的执行路径，所有 tile 必须基于实际执行 CSR。它是长尾计算优化，不是转置生命周期优化。P2 与旧 queue 对齐，首版仅 non；显式 trans/conj 报错，compare 给排除原因。不得为了填满操作列表改变旧 queue 的比较边界。

两个新算法均沿用当前新 CSR 路径 CUDA/ROCm 能力边界。P1 需要 int64 寻址、归约、scan 和实际计算类型 atomic；P2 另需 producer 已有整数 atomic/scan。能力未知明确不可用，不借用其他 kernel。FP64 atomic 不能用 FP32 代替。

没有新增高层函数或位置参数。通过原 alg/config、prepare/run、out/out_dtype、查询与 compare 调用；算法切换隔离专用配置，索引回退保持同算法与精度。原始输入重叠检查、strided out、复数分量 stride 和 prepared/op 冲突继续有效。

### 12.3 P1 配置与 kernel 实现任务

独立扁平配置：

| 字段 | 默认 | 可显式选择 |
|---|---|---|
| block_nnz | 256 | 128/256/512/1024 |
| owner_search | bounded | bounded/full |
| single_row_fastpath | true | true/false，严格布尔值 |
| atomic_mode | boundary | boundary/all |
| num_warps | 4，经已有合法集合选择 | 已知合法值 |
| num_stages | 1 | 首版仅 1 |

所有实验配置只作用于新算法，不能改旧 segbin。full+关闭快路+all 是机制控制组，并不声称与旧 segbin 编译指令逐项相同。显式非法配置报错；未知字段不能被忽略。

在 `_spmv_csr_kernels.py` 新增独立 JIT：

1. 每 program 以 int64 计算 tile 起止。NNZ=0 不启动计算 kernel；初始化仍保证空行输出。
2. upper-bound 搜索状态的形状和 dtype 固定；P 的有效索引范围为 [0,M]，无效 NNZ 不得参与搜索结果或加载。
3. bounded 模式在本 kernel 内得到两个 owner，不生成独立 Bounds 数组。full 对照关闭其搜索缩小效果；快路若关闭，不能暗中依靠同 owner sum 跳过 scan。
4. 同 owner 快路对实数执行 tl.sum，对复数分别规约两分量；统一判定完整行与跨界行。
5. 一般路径按连续行分段 scan，head 由实际 row_start 与 tile 首位置确定，tail 包括最后有效 NNZ。padding 不能贡献有效输出。
6. boundary 模式只有完整行 store，其余 atomic；all 模式所有有效局部行段 atomic。每一行所有 tile 的分类必须一致。
7. 正常执行：初始化 → 计算 → 必要的输出转换，同一实际设备流提交。无 CPU 计划、行列表、计数回传或边界部分和数组。

计算类型与输出恰好一致时可直接累加到合法 out；否则分配计算精度缓冲，最后直接 copy/转换到合法 out，保持对象身份。数值临时空间上界为输出元素数×计算元素字节数（复数计两个分量）；分配失败明确报错。零输出维度、零 NNZ 与合法非连续 out 分别处理。

源码阶段必须给出三项不变量：NNZ 覆盖完整且不重叠；完整行唯一写者；跨界行不存在 store/atomic 混写。所有部分结果在最终转换前保持计算精度，沿用既有融合策略和容差。

### 12.4 P2 最小差异实现任务

新名称 `csr_longrow_grid_atomic` 使用独立 run 分支和静态 consumer；允许复用不修改计算体的旧 producer、finish 和容量 helper。这一版刻意只改变消费调度，不需要复制整套 producer。

配置沿用旧 queue 的 rows_per_program、short_block_nnz、long_row_threshold、segment_nnz、block_nnz、producer_num_warps、num_warps、num_stages、workspace_limit_bytes。默认值与旧 queue 一致；新算法拒绝 worker_programs，因为静态网格不使用该字段。

执行步骤：

1. 原方式分配、初始化 counters、descriptors 和 partial。
2. 原 producer 完成普通行与任务构建。
3. 网格容量由 host shape 安全上界决定；每个 program 加载 GPU count，`pid<count` 才加载描述符并计算一个 segment。无效 program 不访问 job 数组，不做浮点 atomic。
4. 每个有效 segment 向原 partial 独占槽所属行 atomic；使用原 finish 写回。

job_cap=0 时不启动 consumer。容量乘法、grid 合法性、整数计数上限与 workspace 继续检查，不静默截断或隐式回到 persistent。producer 与 consumer 分 kernel 同流顺序执行，无跨 program 自旋。

normal/quick 需验证与旧 queue 相同输入、配置、producer 逻辑和精度；测试只能要求共享容差内一致，不能要求 atomic 位级顺序一致。若 P2 没有完整调用收益，不继续叠加 producer 优化掩盖此结果。

### 12.5 结构诊断与结果字段

在现有 `tests/benchmark_utils.py` 复用少量 helper，接入 `tests/test_spmv_csr.py` 的 regular 和显式组合两条路径。使用已有 `--perf-metrics` 开关；不建立新脚本或新 runner 开关体系。

使用 benchmark 已有 CPU CSR 行指针，在性能测量外计算 T=128/256/512/1024 的：

- tile_count、single_owner_tile_count/ratio；
- mixed_tile_row_span 的 p50/p90/max；行跨度含中间空行，不能称为非空行数量；
- full_row_count、crossing_row_count、predicted_boundary_atomic_updates；
- 对实际 L/S 配置计算 expected_long_rows、expected_jobs、job_capacity 及预计有效任务比例。

CPU 推导的任务数明确标记 expected，不能冒充 GPU 实测 count。必要的 actual count 仅在独立诊断执行后读取，不能改变正常 run。转置条件的结构统计对应执行矩阵，在 CPU 测试 setup 构造；不为算子添加转置缓存。NNZ=0 时比例约定与原因明确，不产生 NaN 除零。

非空行跨越 tile 数可从指针直接计算：`c_r=floor((end_r-1)/T)-floor(start_r/T)+1`。仅 c_r>1 的行贡献到 boundary atomic 更新预测之和；复数另记录分量倍数。该预测是代码模型计数，不是硬件 atomic 事务计数。

详细结构配置以 JSON 或稳定字段写入逐行 CSV，不扩大终端表头。元数据保留实现版本、实际算法、计算/输出类型、配置来源、owner 搜索、快路、atomic 模式、输出缓冲身份、实际 stride、workspace 与未验证状态。

P2 独立诊断细分 init、producer、consumer、finish；不将 producer 标为纯处理时间，它同时计算普通行。P1 行定位属于 compute，转置继续按旧约定归类。完整计时始终为 ms=process_cpu_ms+gpu_ms；无 CPU 计划时 process_cpu_ms=0。保留 prepared_run_ms、吞吐、逻辑带宽及采样过滤；分段之和不覆盖总时间。

### 12.6 现有文件联动清单

- `_spmv_csr_config.py`：两个规格、计算类型、配置解析、能力筛选；不能只加名称白名单。
- `_spmv_csr_kernels.py`：P1 新 kernel、P2 新 consumer 和对应 run helper；不修改旧计算体。
- `spmv_csr.py`：高层/prepare/run 路由、输出验证、元数据及安全索引回退。
- `tests/test_spmv_csr.py`、`tests/benchmark_utils.py`：regular/mixed 枚举、CSV 与结构诊断；保持同一准备和完整计时边界。
- `tests/pytest/test_spmv_csr_accuracy.py`、`tests/ci/test_spmv_csr_policy.py`：原 marker 下扩展，不新增重复收集入口。
- `conf/operators.yaml`、`ops_support.py/csv`、既有 GPU CI/公开清单及本优化文档：增量同步；支持表标为未验证。
- `run_flagsparse_pytest.py`、`tools/run_cuda_tests.sbatch`：优先仅核对原路由和透传，若注册驱动已足够则不改文件；默认交付和旧参数保持不变。

### 12.7 正确性与性能验证矩阵

quick：小尺寸覆盖六种输入、半精度同类型/宽输出、各算法声明的 op。P2 non 外的显式请求应测试报错；不能把它错误加入全部 ops 的笛卡尔积。

P1 必须覆盖：前/中/尾连续空行、全空矩阵、tile 正好对齐行界、行长 T-1/T/T+1、单行跨大量 tile、首尾 owner 相同/不同、尾 tile、重复坐标、未排序列、复杂复数与抵消、宽输出判别数据、NaN/strided out、返回身份、输入不变和重叠拒绝。

P2 必须覆盖：没有长行但容量上界非零、actual_jobs 远小于 capacity、单个 segment 尾部、同一长行多个 segment、producer 尾行组、容量/计数上限与非法配置。normal 增加合法混合索引、长归约及接口回退；mock 注入实际执行位置，检查回退后算法与 config 不丢失。

策略测试验证查询、compare、执行共享能力判定，非法配置不忽略，配置不跨算法继承，新路径不误入旧缓存/图绑定。只有明确能力缺失允许 SKIP，编译/执行错误必须 FAIL，不放宽容差。

性能顺序：

1. 计算节点 smoke 正确性，随后一个真实矩阵；出现编译/执行故障先停止扩大批次。
2. 六个代表矩阵，优先 FP16→FP32、BF16→FP32、FP64；regular FP32 单独注明实际计算类型差异。
3. P1 先固定 T256/warps，逐项开启局部定位、同 owner sum、边界写回，再扫 T；旧 legacy_segbin 和旧最佳始终保留。
4. P2 与旧 queue 保持 producer、S/L/K/warps 相同，只比较调度；以容量有效比例和完整时间解释胜负。
5. 有稳定目标收益后扩展 30 矩阵、全部声明类型/op/合法索引，至少三轮交错执行，CUDA/ROCm 分开统计。

cuSPARSE baseline 保留；HIP baseline 不可用应记录原因，不阻断 FlagSparse。准确性 FAIL 有效性能仍记录；单独提供有限输入、正确性通过子集用于部署判断，不替换用户要求的全样本统计。

主要验收是旧最佳与加入后的最佳 GM/AM、逐矩阵胜出比例和回退；同时报告候选自身，不能只挑赢的矩阵。目标子集重复达到 1.3×作为继续投入参考，至少一个旧最佳被稳定超越；此为决策门槛而非性能承诺。只胜过旧 NNZ 而未超过旧最佳时，不宣称整体提升。

### 12.8 交付关卡

1. **静态交付**：参数链条、索引边界、写入所有权、精度/stride、原 kernel/default 不变，AST/diff 检查完成。
2. **硬件正确性**：用户计算节点完成 quick、目标边界和真实矩阵检查，分别记录 CUDA/ROCm 通过范围。
3. **机制有效性**：消融证明改善源于预期变化；必要 profiler 区分 scan、搜索、atomic、spill、x 间接访存及提交间隙。
4. **算子族收益**：固定条件下加入后最佳 GM 与目标矩阵确有提升；无收益分支保留结论但不替换默认。

实施后再交付经实际 CLI 核对的完整命令。CUDA 已分配计算节点使用 `bash tools/run_cuda_tests.sbatch --repo "$PWD" -- ...`，登录节点用 sbatch；DCU 使用原 runner。regular/mixed 分开，保留 baseline、timing、perf-metrics。下节记录本次源码接入状态及计算节点命令；源码交付不等于硬件验证通过。


## 13. 2026-11-18 源码交付与计算节点验收

### 13.1 本轮实际变更

起点为 `zyq_sparse`，HEAD `011c2c4c1bc6003471f02a4a602c2cc0d40f39f9`，已有未提交 scalar-tail 代码与文档保留。实施前后对 227 个已有 JIT 函数做 AST 摘要比对，计算体均未改变；新增 5 个 JIT 函数（owner 搜索、scan、写回 helper、NNZ kernel、静态 consumer）。没有新增 Python 文件，没有暂存或提交。

- `csr_nnz_local_reduce_atomic` 已通过原规格、配置、prepare/run、高层 API 和 compare 接入。它没有结构计划、计数读回或部分和数组，计算精度与输出不同才创建计算精度输出缓冲。
- `csr_longrow_grid_atomic` 已通过同一链路接入，仅 non。复用原 producer/finish 与容量检查，新增静态 consumer；拒绝无效的 `worker_programs` 配置。容量程序不等于有效任务数，consumer 在设备上用 count 屏蔽无效 program。
- `--perf-metrics` 为 regular CSR SpMV 与 mixed CSR SpMV 追加 `csr_structure` JSON：行长 p50/p90/p99/max、空行、长行 NNZ 比例、四种 tile 诊断及实际队列配置的任务容量比例。trans/conj 使用执行矩阵结构，统计不参加性能采样，也不供 kernel 使用。
- regular 同时追加完整 GPU 吞吐与逻辑带宽、同步墙钟 `prepared_run_ms`；墙钟与原事件路径都复用 out。复数暂不套用现有实数 logical-access v1 模型，指标明确 N/A，但时间、加速比与结构诊断保留。atomic 预测按实数分量计数，复数实际分量更新数乘 2，不等于显存事务。
- grid 的 init/producer/consumer/finish 分段在 regular 和 mixed CSV 中记录。producer 同时完成普通行计算，不能视为纯预处理；不以分段和替换完整 `gpu_ms`。
- 原 accuracy 文件补齐索引回退注入、宽输出跨 tile/segment、实际 strided out、八种输入/输出对、各自合法 op、容量不足与计划禁用检查。原 CI 策略测试补齐能力、配置及 CPU 结构模型检查。算法枚举驱动原 runner，不新增 runner 开关或单独测试入口。

支持表为 **UNVERIFIED**。本地仅 AST、源码及 diff 检查，未执行功能导入、pytest、算子或离线编译。CUDA/ROCm 的编译、准确性、性能均待下面的计算节点验收；不预先宣称提速。

### 13.2 正确性不变量与成本

NNZ tile 完整划分 `[0, nnz)`，尾 tile mask。upper-bound 使用最后一个 `P[row] <= pos` 的行，正确跳过连续空行；同 owner 快路与一般 scan 都将每个有效 NNZ 纳入一次。只有 `row_start >= tile_start && row_end <= tile_end` 才直接写回，这样整行不可能再出现在其他 tile；跨界行的每个局部贡献都 atomic，不存在 store/atomic 混写。初始化在计算前完成，空行保留零。

静态 consumer 的 `pid<count` 与 producer 的连续 job 编号一一对应；无效 pid 不读描述符，有效 pid 只计算一次完整 segment。producer 与 consumer、finish 分 kernel 同流顺序提交，finish 不与 partial atomic 并发。

NNZ 候选新增/保留的成本是两次 endpoint 搜索、跨行 tile 的 owner 搜索、scan、输出初始化与必要转换；同 owner tile 省去逐元素 owner 搜索和 scan，边界模式减少完整行 atomic。静态 grid 省去 persistent 任务计数器的 atomic 获取，但会启动容量内无效 program；容量利用率低时可能比旧队列差。两者都不承诺降低 x 的随机访存成本。

### 13.3 CUDA：已分配计算节点，先 smoke

下列命令在仓库根目录执行。`bash` 不申请节点，只使用当前已分配计算节点；登录节点需另行 `sbatch`。CUDA 命令全部保留 cuSPARSE baseline，没有 `--no-vendor`。

```bash
MATRIX_DIR="$HOME/run/matrix"
RESULT_ROOT="$PWD/11.18data/irregular_candidates"
test -d "$MATRIX_DIR" || { echo "Matrix directory missing: $MATRIX_DIR"; exit 1; }
mkdir -p "$RESULT_ROOT"
SMOKE_MATRIX=$(find "$MATRIX_DIR" -type f -name '*.mtx' -print -quit)
test -n "$SMOKE_MATRIX" || { echo "No .mtx files"; exit 1; }
COMBOS='float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32'

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase accuracy --mode quick \
  --pytest-args "-k 'nnz_local_native or longrow_queue_native or targeted_wide_output_not_roundtrip' -x" \
  --results-dir "$RESULT_ROOT/smoke_accuracy"
```

检查 smoke 结果，不要在新算法编译/执行 ERROR 后继续扩大批次。它含两新算法和旧 queue 对照的六种 dtype、四种半精度组合、local 三种 op、grid non。随后对一个真实矩阵执行：

```bash
for ALG in csr_nnz_local_reduce_atomic csr_longrow_grid_atomic; do
  OPS=all
  if [ "$ALG" = csr_longrow_grid_atomic ]; then OPS=non; fi
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
  "$SMOKE_MATRIX" --alg "$ALG" --dtypes float32,float64,complex64,complex128 \
  --ops "$OPS" --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 2 --iters 5 --timing --perf-metrics --fail-fast \
  --csv-csr "$RESULT_ROOT/${ALG}_regular_smoke.csv"
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
  "$SMOKE_MATRIX" --alg "$ALG" --type-combos "$COMBOS" \
  --ops "$OPS" --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 2 --iters 5 --timing --perf-metrics --fail-fast \
  --csv-csr "$RESULT_ROOT/${ALG}_mixed_smoke.csv"
done
```

`--fail-fast` 针对接口/编译/运行异常，既有矩阵精度 FAIL 仍写入 CSV。厂商状态与 FlagSparse 状态分别检查，不能因 cuSPARSE PASS 就认定候选正确。

### 13.4 原 runner：准确性与完整 regular/mixed 对比

smoke 通过后，完整 quick/normal 准确性使用原 marker。normal 包含更多边界和混合索引，耗时较长。

```bash
for MODE in quick normal; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase accuracy --mode "$MODE" \
  --results-dir "$RESULT_ROOT/accuracy_$MODE"
done

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --dtypes float32,float64,complex64,complex128 --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/regular_all"

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos $COMBOS --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/mixed_all"
```

regular 与 mixed 合起来覆盖六种 dtype、四种半精度组合、所有声明 op 与四种指针/索引组合。grid 在 trans/conj 被 compare 明确排除，不应人为给它补转置执行；local 三种 op 均测。compare 保留旧算法，才能计算旧最佳与加入后的最佳变化。完整命令时间较长；资源有限先将两个索引参数均改为 int32，再扩展。

### 13.5 机制消融：单一显式算法，不把 config 传给 compare

先测两种宽输出，固定 non。以下四项逐次改变搜索范围、同 owner 快路、完整行写回，固定 T=256。之后固定最后一项机制扫 tile 大小。

```bash
WIDE='float16:float16:float32,bfloat16:bfloat16:float32'
NAMES=(full_scan_all bounded_scan_all bounded_fast_all bounded_fast_boundary)
CONFIGS=(
  '{"block_nnz":256,"owner_search":"full","single_row_fastpath":false,"atomic_mode":"all"}'
  '{"block_nnz":256,"owner_search":"bounded","single_row_fastpath":false,"atomic_mode":"all"}'
  '{"block_nnz":256,"owner_search":"bounded","single_row_fastpath":true,"atomic_mode":"all"}'
  '{"block_nnz":256,"owner_search":"bounded","single_row_fastpath":true,"atomic_mode":"boundary"}'
)
for I in 0 1 2 3; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
  "$MATRIX_DIR" --alg csr_nnz_local_reduce_atomic --config "${CONFIGS[$I]}" \
  --type-combos "$WIDE" --ops non --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 10 --iters 50 --timing --perf-metrics \
  --csv-csr "$RESULT_ROOT/local_${NAMES[$I]}.csv"
done
for T in 128 512 1024; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
  "$MATRIX_DIR" --alg csr_nnz_local_reduce_atomic --config "{\"block_nnz\":$T}" \
  --type-combos "$WIDE" --ops non --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 10 --iters 50 --timing --perf-metrics \
  --csv-csr "$RESULT_ROOT/local_T${T}.csv"
done

for ALG in csr_longrow_queue_atomic csr_longrow_grid_atomic; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
  "$MATRIX_DIR" --alg "$ALG" \
  --config '{"long_row_threshold":1024,"segment_nnz":1024,"block_nnz":128,"producer_num_warps":4,"num_warps":4}' \
  --type-combos "$WIDE" --ops non --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 10 --iters 50 --timing --perf-metrics \
  --csv-csr "$RESULT_ROOT/${ALG}_matched.csv"
done
```

同样的 local 消融需补 FP64 对照：将 `--type-combos "$WIDE"` 替换为 `--dtypes float64`，使用不同 CSV 文件名。queue/grid 对照只改变 consumer，默认 queue worker=128；grid 没有 worker 参数。两者无实机收益时不叠加其他优化解释结果。CUDA/ROCm 分开保存，至少三轮交错顺序重测重点候选。

### 13.6 DCU 对应入口

在已配置好的 DCU 计算节点直接使用原 runner。下面以 mixed 完整扫描为例，baseline 不支持时记录 N/A，不关闭 FlagSparse 测试；不要将缺失 baseline 当作速度零或加速比零。

```bash
python run_flagsparse_pytest.py \
  --ops spmv_csr --phase accuracy --mode quick \
  --pytest-args "-k 'nnz_local_native or longrow_queue_native or targeted_wide_output_not_roundtrip' -x" \
  --results-dir "$RESULT_ROOT/dcu_smoke_accuracy"

python run_flagsparse_pytest.py \
  --ops spmv_csr --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos $COMBOS --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/dcu_mixed_all"
```

DCU regular 将 benchmark-args 中 type-combos 换成 `--dtypes float32,float64,complex64,complex128`，结果目录改为 dcu_regular_all；消融以 `python tests/test_spmv_csr.py` 替换 CUDA wrapper，其余参数不变。

### 13.7 返回数据后的判定

按固定设备、dtype/输出类型、op、索引分组，保留全部有效时间及精度状态；先对每矩阵分别求旧最佳和含新候选最佳，再计算全矩阵 GM/AM、覆盖数、胜出率与逐矩阵变化。候选自身回退与算子族最佳变化分开报告。结构与耗时对齐，重点检验：单 owner 比例是否解释快路收益，跨行跨度是否解释搜索/scan 开销，边界更新数是否解释 atomic 消融，容量利用率是否解释静态 grid 的失败。

若完整 GPU 时间改善而墙钟未改善，需要检查提交及分配成本；不能将 compute 单段改善当作完整收益。没有 baseline 的组合只报告自身相对改进。所有分析随原始数据保存在对应日期目录，不回填无法同条件匹配的历史厂商时间。
