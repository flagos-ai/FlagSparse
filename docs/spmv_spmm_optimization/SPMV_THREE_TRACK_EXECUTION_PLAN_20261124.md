# SpMV 三组实验落地方案：直接计算、长尾分工、访存与开销

日期：2026-11-24。目标：`FlagSparse-main/zyq_sparse`。

**状态：2026-11-24 源码接入与静态检查完成；未运行编译、pytest、算子或性能验证。两个算法及诊断开关已接入原测试链条，硬件可用性和收益仍待计算节点验收。**

依据：[理论与实践复盘规划](SPMV_PORTFOLIO_RESEARCH_PLAN_20261124.md)，以及工作区 `test_mixed_data/11.23-newdata/analysis/portfolio_review_20261124` 的原始数据复核与边际贡献表。

## 1. 本轮实施清单及范围

| 编号 | 工作项 | 交付性质 | 是否新增公开算法 |
|---|---|---|---|
| D1 | row_vector的K/warp/stage有限消融 | 已有配置、旧kernel不变 | 否 |
| D2 | scalar-bound/static-tile有限消融 | 已有配置、旧kernel不变 | 否 |
| D3 | program窗口内一次性行重组 | 独立kernel、配置、注册、测试 | **`csr_window_grouped_reduce`** |
| L1 | P/static/ticket的真实任务粒度复核 | 复用已有可选诊断；必要时补统计字段 | 否 |
| L2 | 行边界与NNZ联合分解 | 独立kernel、配置、注册、测试 | **`csr_merge_event_atomic`** |
| M1 | x访问局部性和窗口分组成本模型 | 原benchmark内可选离线诊断 | 否；拟新增诊断开关 |
| M2 | 普通run时间线、资源与访存分析 | 计算节点profiler；保留原性能主表 | 否 |
| M3 | 已有fused/static consumer匹配对照 | 使用已接入的独立融合候选，不再新增名称 | 否 |

三组工作并列准备。D1/D2/M1/L1无需等待两个新算法写完；D3/L2分别通过自己的证明、编译、正确性和性能门槛，互不阻塞。

本轮以CSR SpMV为实现主体。CSC trans/conj的既有配置验证可作为后续同思想验证，不在此轮顺带复制新kernel或修改SpMM。COO/CSC保留原回归入口。

保留旧算法计算体、默认参数、默认路由、已有compact/fixed、static/ticket、缓存和图绑定范围。只增量修改现有Python模块，不新增Python文件或调优器；所有新算法从原API调用，不增加每算法顶层函数。

## 2. 公共数值、接口和能力契约

两个新算法拟声明相同数值范围：

| A与x | 计算/寄存器/atomic部分和 | 输出 |
|---|---|---|
| FP16、BF16 | FP32 | 输入类型或FP32 |
| FP32 | FP64，沿用现有新CSR策略 | FP32 |
| FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | complex64 |
| complex128 | FP64实虚分量 | complex128 |

- 支持`non/trans/conj`，后两者沿用当前CSR执行矩阵重建生命周期；两种新算法都针对**实际执行CSR**，不宣称转置免重建。
- A/x不得异类型；不改变乘加融合策略、共享容差或默认mixed路径。
- prepared保留原形状、op、类型与算法配置契约。算法切换不继承专用配置；out类型在run解析。
- 直接按合法x/out stride访问，返回传入out身份；原始输入与out的别名检查不能被prepare副本掩盖。
- 索引兼容回退仍为同算法；保留config/op/输出类型/out。所有全局行号、NNZ位置、地址乘法和co-rank上下界使用必要的int64宽度。
- 能力边界为现有新CSR的CUDA/ROCm。D3需要整数gather、比较交换、浮点归约；L2额外需要scan和实际计算精度的atomic。FP32输入的L2需要FP64 atomic，不能按输出FP32误判能力。
- 未知能力报告不可用；编译错误不伪装成能力缺失。不能切到其他算法后仍报告新名称。

## 3. 第一组：直接计算实验

### 3.1 D1：row_vector只做配置消融

现有配置是嵌套结构，不是扁平`block_nnz`。有效配置示例：

```json
{"row_vector":{"block_nnz":128,"num_warps":2,"loop_num_stages":1}}
```

保留旧`row_vector_kernel`，不增加vector同义名称。按顺序测试：

1. 固定warp=2、stage=1：K=64/128/256/512，共4个点。
2. 固定第一步选出的K：warp=1/2/4中的后端合法值，最多再加2个点。
3. 固定K/warp：loop stage=1/2，最多再加1个点；只有编译产物确实不同且没有明显spill才进入完整矩阵测试。

最多7个不同点，不做4×3×2全组合。参数先在预登记开发子集选定，再冻结为固定profile对全部30矩阵测试；逐矩阵参数最优另列为研究上限，不能充当部署结果。

重点：TSOPF、pkustk13、c8_mat11、msc10848，以及mip的中行主体。保留短行与wiki作为回退控制。比较完整run、纯kernel诊断、寄存器、spill、归约指令和访存。

旧row_vector可能通过原包装完成strided输入/输出适配；本实验保持包装边界一致，不能把新候选直接写out的收益混成K变化。

### 3.2 D2：scalar/static只做配置消融

现有扁平配置示例：

```json
{"rows_per_program":64,"num_warps":2,"num_stages":1}
```

用于`csr_row_scalar_bound`。`csr_row_static_tile`另带`block_nnz`：

```json
{"rows_per_program":8,"block_nnz":16,"num_warps":2,"num_stages":1}
```

固定其余参数，scalar先R=32/64/128，再验证合法warp=1/2/4；static先R=8、K=8/16/32/64，再固定K测R=4/8/16。stage保持1，符合当前校验器，不给旧短行路径强加stage=2。

不加入新的S阈值搜索。S128继续作为整个算法组合的必要旧对照。凡是需要修改旧计算体才能实现的想法，另立候选评审，不混入这两项配置消融。

### 3.3 D3：新增`csr_window_grouped_reduce`

**要解决的瓶颈：** 相邻行长不一致时，一组R行按最长行循环，短行lane长期mask。旧refill每轮领取新行，需要重复整数scan和状态维护。本算法在有限窗口内只组织一次，然后静态执行，比较一次性组织是否更便宜。

**它不是超长行拆分。** 单个超长行仍由一个program负责；跨program长尾由L2研究。本算法目标是局部中等不均匀，而非所有长尾。

拟新增扁平配置：

```json
{"window_rows":64,"rows_per_group":8,"lanes_per_row":8,
 "row_order":"length_desc","num_warps":2,"num_stages":1}
```

候选W=16/32/64/128，R=4/8/16，V=4/8/16/32/64且不超过后端subgroup；W和R为二次幂、W≥R、W%R=0。warp从已有合法集合选择。`row_order=original`为新kernel内部对照；W=R可用于测排序本身开销，不期待分组max减少。

**实现步骤：**

1. program拥有`[pid*W,min((pid+1)*W,M))`，加载W个行长及局部行号0..W-1。
2. `length_desc`按行长降序、相同行长按局部行号升序排列。padding行长设为-1。排序对象是**成对的行长和行号**，禁止用可能溢出的`length*W+id`组合键。
3. 使用固定小规模bitonic compare-exchange网络：每层通过`tl.gather`读取xor伙伴，同时交换key与id；比较器显式实现上述字典序。所有交换距离/层数为编译期常量。不依赖未核实的`sort_pairs` API，也不调用Triton私有排序helper。
4. 每次从排好序的W个行号取R个，换算为int64全局行号，重新加载这些行的起止指针。
5. 以`[R,V]`累加器按该组实际最长行循环；lane k计算`start+k+t*V`，实虚分别累加，最后沿V归约。
6. 直接写原行号的out；完成本组后复用累加器处理下一组。空行写零，不单独清零整输出。

初版只在排序期保留key/id，不把全部W行的start/end和累加器长期驻留。代价是计算期再次读取行指针，必须计入成本；不假定它们必在cache中。

**正确性证明义务：** 交换网络必须产生局部行号的排列，valid行无丢失、无重复；不同program窗口不交；每行只在一个组出现；组内NNZ子序列互斥且覆盖该行。padding必须尾置并在所有行指针、A/index/x/out访问上mask。`tl.gather`索引先限制在合法范围，不能依赖`where`阻止越界。

**成本模型：**

`scheduled_positions = Σgroups R*V*ceil(max(row_length)/V)`。

离线比较original和length_desc的scheduled_positions；这不是占用率或执行时间。排序复杂度约O(W log²W)，还增加行号gather和指针读取。只在同一组内重排行不降低max，必须改变分组成员。

**实验证据要求：** 先固定W/R/V比较`original`与`length_desc`；确认排序本身有效后才比较W。另与旧row_tile/subgroup、refill、scalar/static/vector及整体旧最佳对照。不能只比refill这个已知较弱的实现。

## 4. 第二组：长尾工作分解

### 4.1 L1：任务粒度诊断，不再新增固定P/ticket变体

复用当前`--task-diagnostics`及其独立运行流程，保持原计时采样不受追踪影响。保存实际队列顺序、每part有效NNZ/迭代数、空part、worker累计工作、最长worker/均值、初始任务与后续ticket领取数。

同批保留：S128、bounded P8/P16、static P8/P16、ticket B1代表点和已有fused-static。默认Gm1024/Gl128作为基准，但诊断必须记录实际grid而不是只记录配置上限。

`length/P`不等于等工作；先确定任务差异和空任务，再判断是否值得可变Q。旧longrow-queue已有固定segment思想，**本轮不新建一个仅把P改为Q的算法**。只有诊断证明其收益空间，并提出比旧segment更低成本的描述符/归并机制后，才单独扩展计划。

### 4.2 L2：新增`csr_merge_event_atomic`

**与旧NNZ-local的区别：** 旧算法按全局NNZ切tile，再定位owner，已有首尾范围缩小和单owner快路径。本算法将行边界和NNZ合为事件序列，均分事件，减少每NNZ独立owner定位，覆盖连续空行等结构。

初版不增加跨调用描述符，不做CPU计划，不使用全局任务队列。保留旧NNZ-local/compact/fixed作为对照。

拟新增配置：

```json
{"strips_per_program":64,"events_per_strip":4,
 "num_warps":4,"num_stages":1,"workspace_limit_bytes":268435456}
```

C=`strips_per_program`取32/64/128；L=`events_per_strip`取2/4/8；Q=C*L≤1024，均为二次幂。配置合法还需满足目标launch、scan、FP64和atomic能力。这里strip是一个**逻辑tensor元素负责的事件区间**，不承诺与一个CUDA线程一一对应。

#### 4.2.1 事件定义及co-rank

记`P=indptr`，行数M，非零元数Z。合并两个升序序列：

- M个行结束事件`P[1],...,P[M]`；
- Z个NNZ事件`0,...,Z-1`。

相同位置时，行结束事件先于NNZ事件，确保重复指针对应的空行先被跨过。总事件数D=M+Z，host只用已有形状检查加法是否超int64，不读GPU计数。

事件前缀长度d的co-rank为`r+p=d`：已消费r个行结束事件、p个NNZ。搜索区间为`max(0,d-Z)≤r≤min(d,M)`。边界有效时满足：

```text
P[r] <= p       （r>0且p<Z时约束）
P[r+1] >= p     （r<M且p>0时约束）
```

实现使用完整的端点分支/哨兵语义：r=0/M、p=0/Z不能越界读取。第二式来自`p-1 < P[r+1]`，是稳定tie规则，不得误改成严格大于p。先给这个纯索引规则建立小例穷举oracle，再接浮点计算。

program负责`[pid*Q,min((pid+1)*Q,D))`；其C个strip各负责L个连续事件。每个strip只在起点co-rank，然后最多推进L次，不在每个NNZ重新二分。program两端co-rank给出其连续NNZ区间`[p_begin,p_end)`。

#### 4.2.2 连续加载与有限事件消费

1. program按`p_begin+arange(Q)`连续加载A/index及x，mask到`p_end`；生成对应计算精度乘积。复数分别保留实虚分量。
2. strip维护固定shape `[C]`的当前r/p、累加值、是否已有NNZ及首尾片段状态。
3. 每一步若下一个事件为行结束，则处理当前行累加并推进r；否则从已加载的乘积tensor中gather `p-p_begin`、累加并推进p。
4. gather下标先clamp到0..Q-1，再用“有效事件且为NNZ”屏蔽；无效尾strip不参与数值更新。
5. 循环结束处理尚未遇到行结束事件的尾片段。使用是否消费过NNZ的布尔标志，不能用“部分和非零”判断有效性，避免抵消数据丢失。

不需要为每个strip另做一次终点co-rank。记录起点`p0`：循环中遇到行结束时，该行end已被消费区间覆盖，若行start≥p0就是strip完整行，否则是首片段；循环结束后的p即p1，尾片段再用start≥p0且end≤p1判断完整性。这样端点分类不依赖尚未计算的p1，也不漏掉“NNZ已全部消费、结束事件恰在strip之外”的完整行。

这样保留A/index的连续读取机会，但支付局部gather/状态推进。Q个宽乘积、C个状态和边界数组可能显著增加寄存器。首版不增加双缓冲或prefetch，避免同时引入多个机制。

#### 4.2.3 两级边界归约及唯一写规则

每个strip覆盖连续事件和连续NNZ区间，最多有首尾两个不完整非空行片段：

- NNZ全部包含在该strip的行直接store，且不进入边界记录。
- 首尾不完整片段保留在program内的固定2C槽位；首尾属于同一行只保留一次贡献。
- 槽位owner保持事件顺序。无贡献槽位携带有序owner、零值和`active=false`，不能在中间塞M哨兵破坏单调性；只有末尾越界strip使用末尾哨兵。
- 对2C槽位做按owner的segmented scan，同时OR active标志。仅段尾且active的记录产生输出，防止零占位记录覆盖已直接store的完整行。
- 如果该行全部NNZ包含于program的`[p_begin,p_end)`，归并后直接store；否则atomic到计算精度目标。

完整行判定用实际行指针：`P[row]>=p_begin 且 P[row+1]<=p_end`。若行跨program，它的**所有**program片段都必须atomic，不能有的store、有的atomic。

因此全局atomic只来自跨program的首尾行，最多每program两个逻辑行更新；复数乘以两个分量。program内跨strip不是全局atomic理由。这是与“每strip直接atomic”原型的重要成本区别。

#### 4.2.4 初始化、数值缓冲与边界

空行没有NNZ贡献，目标先清零。输出类型就是计算类型且out合法时直接用out；否则分配M元素的计算精度缓冲，所有program结束后一次转换/复制到out。FP32输入不能atomic到FP32输出，因为本库此路径要求FP64计算。

workspace上限只统计额外全局数值缓冲，不包含必要返回张量；按复数实际字节数计算。寄存器/shared消耗单独记录，不能混为全局workspace。超过上限报错，不自动分批或换算法。

Z=0时清零后返回，跳过空事件遍历；M=0输出为空。所有位置/stride用足够宽度。输入不被改写。

#### 4.2.5 性能假设及停止条件

owner搜索数量大约由逐NNZ变为`ceil((M+Z)/L)`个strip起点，加program端点；但旧NNZ-local的单owner快路径本来就省搜索，所以不能按Z直接计算理论加速倍数。

本算法可能输在：空行事件过多、co-rank本身、局部gather、2C scan、宽寄存器或atomic。目标wiki/Stanford/长尾合成矩阵，mip与TSOPF作为必须面对的实际对照，不保证它们都适用。

先固定Q比较C/L分工，例如Q=256时(C,L)=(128,2)/(64,4)/(32,8)，隔离“搜索摊销与每strip串行工作”的变化；选定L后再改变Q。主对照必须含旧NNZ-local（含其单行快路径）、row-vector、S128及整体旧最佳，不能只比慢的compact/fixed。

## 5. 第三组：访存与执行开销

### 5.1 M1：补结构诊断，暂不凭猜测新增cache算法

已在现有`tests/benchmark_utils.py`和`tests/test_spmv_csr.py`接入`--locality-diagnostics`。它仅对CSR SpMV生效，要求`--perf-metrics`和CSV输出；regular和显式mixed使用同一个helper。源码已接通，硬件运行待验证。

诊断全部在性能采样之外，使用CPU执行CSR结构。trans/conj按原列稳定排序生成诊断用行索引，保留重复项及存储顺序，不做coalesce；不能只用转置行长推断x局部性。

现有`--task-diagnostics`继续只对具备任务追踪实现的队列算法收集实际任务；对直接算法返回明确not_applicable，不能制造一份“实际队列”数据。M1的新局部性诊断与任务诊断独立，是否收集由各自开关决定。

输出内容：

| 数据 | 定义 | 用途 |
|---|---|---|
| 原始/分组scheduled_positions | 同W/R/V下按组max计算的位置数 | D3排序可节省多少逻辑工作 |
| distinct_x_ratio | 窗口内不同x地址数/NNZ数 | 随机gather与重复潜力 |
| model_line_count | 按32/64/128字节粒度统计不同模型块数 | 比较局部性，不称真实cache line命中 |
| adjacent_window_reuse | 相邻窗口模型地址块交集/后一窗口块数 | program重排是否可能伤害复用 |
| COO式局部owner段/CSR单owner tile率 | 沿现有结构诊断定义 | 区分scan、owner搜索与atomic机会 |
| 精确/采样范围 | 确定性抽样规则、数量、覆盖NNZ | 不把采样统计写成全量 |

对大矩阵以确定性等间隔最多4096个窗口抽样，额外单列最长行/边界窗口，不混成无偏总体估计；空分母返回N/A及原因。x实际stride和元素字节数参与地址代理，缓存块起始对齐用归一化偏移并明确不代表真实设备地址。

详细数组写入同CSV旁压缩JSON，CSV记录模型版本、摘要和路径。regular/mixed加入原始结构与数值指纹、部署commit/dirty状态，避免再出现regular不能核对相同输入的问题。诊断失败单独记录ERROR/traceback，不擦除已保存的性能行；不得记为成功收集。

### 5.2 M2：profiler与执行边界匹配

先选mip/wiki/TSOPF、一个短行控制和一个相同长度但列分布不同的合成对。profile独立运行，不用profiler下的时间替代性能CSV。

需要：各kernel真实持续时间、kernel之间GPU空隙、分配/同步、寄存器和spill、L1/L2/DRAM计数、加载sector和atomic情况。CUDA与ROCm使用各自可用工具；字段不可用明确标注，不能把logical_gbs当硬件利用率。

`gpu_ms`仍是完整run事件窗口，可能含主机提交引发的GPU空隙；不能由它减去几个独立phase时间来估算launch成本。`process_cpu_ms=0`不代表没有Python开销，只代表没有计入的CPU算法计划工作。

首轮不加入graph-replay新开关。若时间线明确说明提交空隙主导，再单独设计可capture的绑定实验；主结果与厂商必须使用同类提交模式，禁止只给FlagSparse使用graph后继续称普通run对比。

### 5.3 M3：已有融合consumer对照

`csr_three_tier_fused_static_reduce`已有源码，硬件尚未验收。本组只测它与原static/bounded的匹配配置：相同S/L、P、Gm/Gl、输入、输出复用、producer/finish和数值精度。

融合可能同时减少launch和改变program映射，需根据资源与时间线分别解释。若只是少一个kernel却使寄存器或关键路径变坏，停止围绕它扩展，不为维持方向而增加更多阈值。

## 6. 原接口和文件联动清单

后续实施只改既有文件：

| 位置 | 计划改动 | 必须保留 |
|---|---|---|
| `_spmv_csr_config.py` | 两个独立规格、专用配置、能力与dtype/op检查 | 旧默认和旧profile不变；非法显式配置报错 |
| `_spmv_csr_kernels.py` | 新kernel/数学helper/运行封装 | 旧kernel AST与JIT签名不变；不改C API契约 |
| `spmv_csr.py` | 显式执行分支、元数据、同算法fallback | 默认gather/cache/直接绑定范围不扩大 |
| `_spmv_csr_benchmark.py` | 仅必要元数据接通 | 原事件窗口/采样过滤/完整与诊断分离 |
| `tests/test_spmv_csr.py`、`tests/benchmark_utils.py` | 注册驱动compare、诊断、字段和逐行保存 | 单显式算法才可传专用config；regular与mixed不漏字段 |
| `tests/pytest/test_spmv_csr_accuracy.py` | 两算法原生正确性与接口边界 | 原marker、quick/normal、共享golden/容差/JSON |
| `tests/ci/test_spmv_csr_policy.py`及既有CLI/registry策略测试 | 规格、配置、路由、诊断纯CPU模型 | 不新增第二套测试入口 |
| `conf/operators.yaml`、`ops_support.py`、相关文档 | 新算法清单、范围和未验证状态 | 不更改无关算子文档 |
| `run_flagsparse_pytest.py`、GPU CI入口 | 审计原参数透传和算法枚举；只补实际缺口 | 默认脚本选择、marker、参数优先级不变 |

不新增算法顶层函数；因此两级函数导出通常无需增加符号，但必须核对已有算法查询导出的结果包含新名称。runner通过既有`--benchmark-args`/`--op-benchmark-args`透传配置与诊断，不增加另一套runner开关。

新增元数据分别记录row_order/W/R/V、C/L/Q、计算/输出类型、输出缓冲身份、初始化/转换、workspace、原生路径、转置生命周期、配置来源和实现版本。有效边界数等统计只在诊断中收集，正常run不为元数据读GPU计数。

诊断归属明确如下：D3的局部排序与乘加在同一kernel，记录`fused_plan_compute_ms`及`local_ordering_scope=inside_compute_kernel`；若`compute_ms`也覆盖该kernel，明确二者为重叠诊断，不得相加。`process_gpu_ms`只记录可独立测量的原转置等处理，不把non时该字段为0解释成“没有排序成本”。L2的co-rank、局部scan、初始化、atomic和最终转换全部归入compute；不生成独立执行计划。正常完整时间始终包含以上全部工作。

同一对照中的输出分配/复用遵循该入口原契约，记录`output_reused`和`output_allocation_policy`；不能只为某个候选提前分配out，而让另一个算法承担额外分配后称为kernel优化。regular与mixed入口本来存在的边界差异单独标识，不跨入口直接做因果归因。

## 7. 正确性与语法验收

### 7.1 本地允许做的事

实现时记录旧kernel/config摘要，完成AST、diff和参数传递审查。当前环境不执行库功能导入、算子、pytest或离线Triton编译。静态检查不能宣称GPU可编译。

co-rank、compare-exchange和边界归约先写入现有纯策略测试的CPU oracle；测试留计算节点或后续获准的CPU测试环境运行。本轮方案阶段不运行这些测试。

### 7.2 原pytest覆盖矩阵

- quick：两算法 × regular四类型及mixed四组合 × 三种op，小矩阵；索引用代表性合法组合。normal扩展四种索引组合、long cases、stride和配置边界。
- 通用：空矩阵、空行、重复项、未排序列、矩形、非对称复数、抵消、NaN out、宽输出非roundtrip、输入不变、输出身份、重叠拒绝、op冲突、配置隔离、索引fallback。
- D3：W/R/V边界、相同行长确定性tie、padding、W=R、W>R、所有行空、窗口末尾、单极长行与长短交错；`original`/`length_desc`都检查。
- L2：重复行指针、开头/中间/末尾空行、d=0/D、co-rank tie、NNZ正好落在strip/program端点、单行跨多个strip/program、完整行与边界行混合；零值片段和抵消不能丢失。
- L2边界测试同时验证中间记录的owner单调、active标记、最多两个有效跨program行；CPU oracle穷举小CSR结构及所有d，与合并事件真值逐项比较。
- atomic算法允许既定容差内的非确定求和，不用两次GPU结果逐bit一致作为诊断正确性条件；数据访问/写入覆盖由独立oracle验证。
- 未声明能力可SKIP；配置、编译、执行及正确性问题必须FAIL，不扩大异常捕获、不放宽容差。

策略测试和准确性测试均保留原收集方式。新测试函数建议使用`window_grouped_native`、`merge_event_native`可读片段，方便原runner的`--pytest-args -k`定向；正式命令待实际函数落地后核对。

## 8. 实验分级与验证门槛

### G0：设计与索引规则

必须给出唯一输出/合法atomic证明、dtype与stride传递、最坏workspace和循环state类型。D3不能假设不存在的排序API；L2不能把每个strip的部分和直接全部atomic后声称实现了program级边界合并。

### G1：小样本编译及准确性

regular/mixed分开smoke，再normal。无编译错误或不明SKIP，全部新增接口/合成正确性用例通过才进入长批次。真实矩阵历史精度FAIL保留并逐项对照，不以改变输入或容差消除它们。

### G2：机制实验

| 工作项 | 必要对照 | 继续条件 |
|---|---|---|
| D1/D2 | 同kernel每次单一配置变化 | 目标子集优势跨至少3轮重现；无无法解释的spill/回退 |
| D3 | 相同W/R/V的original/length_desc | 重组减少逻辑工作，且节省超过排序/gather成本；若只控制版更快则单独解释 |
| L2 | 同Q不同C/L；旧NNZ-local fastpath开启 | owner组织成本有降低，边界处理未抵消；不能仅与慢计划对照 |
| M3 | 相同任务划分和配置的static/fused | 完整run有收益，并辨析launch、映射、资源变化 |

访存与寄存器指标只解释性能，不替代完整run结果。若stage或cache实验没有生成不同机器码，不把数值噪声当优化。

### G3：全30矩阵组合价值

首轮性能：regular FP32/FP64，mixed两种FP32宽输出，non、int32；之后覆盖complex64/128、两种half输出、trans/conj和合法混合索引。准确性始终覆盖全部声明范围，不因性能优先级缩减支持验收。

每轮保留全旧算法compare及显式S128；候选与配置按固定顺序轮换，至少3轮。一个类型/矩阵/轮内采用同一compare vendor锚点；同轮算法时间才可组合，不跨日期捡最小值。

先冻结开发子集上的配置，再在剩余矩阵验证。报告：固定配置单算法GM/AM、逐矩阵配置oracle、旧最佳/加入单候选/加入全部候选的GM、胜出率、>5%胜出、回退、移除损失与准确性状态。三种“最佳”不能混为自动路由。

30矩阵已经多次参与历史设计，剩余子集只算本轮配置选择的留出验证，不能宣称是从未见过的独立测试集。泛化判断另用控制行长/列局部性的合成矩阵，并在后续有新矩阵集时补充外部验证。

建议研发继续投入门槛：预登记目标子集GM稳定改善≥10%，或全体最佳GM稳定增加≥1%；特殊少数矩阵若有稳定大提升也可保留，需单独说明覆盖。1%～3%单轮胜出不足以继续堆叠版本。配置、编译或新数值回归未解释时，不进入推荐配置。

计时不变：`ms=process_cpu_ms+gpu_ms`；完整采样不插诊断事件。保留`prepared_run_ms`、吞吐、逻辑带宽、现有过滤规则及精度FAIL的有效性能。cuSPARSE baseline继续开启；baseline的准确性和性能独立标识。

## 9. 实施顺序与停止范围

1. 接通M1诊断和D1/D2实验记录，L1复用已有任务诊断；先固定旧对照与可追溯输入。
2. 并列形成D3和L2的索引/归约oracle及配置规格，再分别实现独立kernel，逐项过G0/G1。
3. D3进行重组对照，L2进行同Q的C/L对照；同时用M2解释资源和访存。已有fused按M3验收一次。
4. 通过G2的候选再进入全矩阵G3，更新相关文档与支持表，默认路由仍不切换。

暂不实施全局结构缓存、索引压缩、可变Q新队列、在线调优、cache-hint算法名和graph主测入口。它们必须由本轮诊断提供理由再单独立项。

交付命令继续使用`tools/run_cuda_tests.sbatch`，已分配计算节点用bash，登录节点用sbatch。沿用原runner的accuracy/performance、quick/normal及参数透传。完整命令待CLI与测试名称实际落地后直接给用户，不在方案中把尚未实现的新名称写成已可运行。

## 10. 核对过的外部语义

- Triton v3.6.0公开`standard.py`与`language/__init__.py`核对了sort/gather/scan等入口；没有把假定的key/value排序API写入实现契约。[standard.py](https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/triton/language/standard.py)、[公开导出](https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/python/triton/language/__init__.py)
- merge事件空间借鉴Merrill/Garland的工作分解思想；本方案的Triton strip映射、局部gather和边界scan是待验证设计，不是原论文的性能结论。[Merge-based SpMV](https://mgarland.org/papers/2016/spmv/)


## 10. 本轮源码落地记录

### 实现及对照边界

- `_spmv_csr_kernels.py` 新增独立 `window_grouped_reduce_kernel`、`merge_event_atomic_kernel`，没有替换旧 kernel 计算体。
- 窗口排序用固定比较交换网络，按行长降序、相同长度按原行号升序。`row_order=original` 编译特化完全跳过排序。增加 `window_rows=16`，配合 `rows_per_group=16` 形成 W=R 控制组；其他默认不变。
- merge 的 co-rank 搜索最小 r，使 `P[r+1] >= d-r`，上下界为 `max(0,d-Z)`、`min(d,M)`，端点用 masked load；不读取越界 P[M+1]。
- 每个 strip 内 NNZ 区间连续：内部完整行只由该 strip 写一次，首尾片段进入有序 2C 槽位。片段 owner 在填充槽位中仍非递减，active 标志避免零占位覆盖已写输出。
- 同一跨 program 行不可能被任何 strip 判为完整行，因此只走计算精度 atomic。全部有效 NNZ 恰属于一个事件区间；没有重复乘积贡献。空行由初始化输出零。
- FP32 输入使用 FP64 计算与 atomic；half/BF16 使用 FP32。仅最终输出转换，返回 strided out 原对象。workspace 仅限额外数值输出缓冲，不含返回张量及寄存器。
- 使用现有 API、规格、类型解析、每次转置、索引回退及 benchmark run，未扩展旧直接绑定、缓存或图捕获。两个新算法没有单独顶层 API。

### 局部性诊断

`--locality-diagnostics` 要求 CSR SpMV、`--perf-metrics` 和 CSV 输出。regular 与 mixed 均在完整 GPU/墙钟采样后执行；不改变 `ms=process_cpu_ms+gpu_ms`、分段诊断或 FAIL 有效性能规则。

原 CSV 新增诊断状态、原因、结构/输入指纹、摘要和附件路径；`<CSV>.locality/*.json.gz` 保存采样窗口及极端补充窗口。模型默认 W64/R8/V8，新窗口算法使用实际 W/R/V。其他算法使用相同对照模型，不声称该模型就是它们的调度。

- 每窗 distinct-x 比例、32/64/128 字节逻辑地址块数、相邻真实窗口的地址块交集比例。
- 原顺序与局部重组的调度位置数 `sum_group(R*V*ceil(max_length/V))`；这是无效 lane 工作代理，不是周期估计。
- 最多4096个确定性均匀窗口，极端8行所在窗口与首尾窗另列；标记 exact 或非无偏抽样，不能把补充极端样本混入总体均值。
- trans/conj 在 CPU 诊断副本中稳定构建实际方向结构，保留重复项；不改变算子输入或运行计划。
- 地址块假设基址零偏移，使用实际元素字节数与 RHS stride。不能解释为 L1/L2 命中率、显存利用率或实测事务。
- 诊断故障记录 ERROR、reason 与 traceback，保留已得到的性能行。生成指纹与 CPU 诊断的时间不计入算子过程时间。

### 原测试链路

原 `test_spmv_csr.py` 的 `compare/all` 从注册信息枚举；统一 runner 原脚本/marker/透传规则无需改写。GPU CI 的交付静态检查增加两个名字。

原 accuracy 文件增加 `window_merge` 用例：八种输入/输出组合、三种 op、空维度/空行、尾部、长行、重复/未排序索引、strided x/out、半精度宽输出、参数冲突和索引回退。normal 扩展四种索引组合、65537 NNZ 长行及不同 C/L/W/R/V。原通用边界、抵消和回退用例也通过注册枚举包含新算法。

增加直接验证比较交换网络顺序的 GPU oracle，以及按稳定事件显式枚举验证每个 diagonal 的 co-rank oracle；不把 CPU 模型作为 GPU 编译通过的证据。策略测试检查配置、FP64 atomic 能力、局部性模型和诊断不进入采样循环。

### 三层验收状态

| 项目 | 状态 | 下一步 |
|---|---|---|
| 索引划分、唯一写入、计算精度及成本审查 | 源码论证完成 | GPU oracle 与准确性核实 |
| Python AST、旧 kernel 对照、参数链条 | 本地静态检查 | 详见交付摘要 |
| Triton 编译、寄存器/布局转换、正确性 | 未运行 | 先 smoke，再 normal |
| 整体最佳 GM、旧 S128、单算法及回退 | 未运行 | 同批 regular/mixed 对照与消融 |

D1/D2/L1/M2/M3 沿用既有实现与开关，只安排配置/诊断实验，不新增公开算法。窗口排序有 `O(W log²W)` 比较交换及 gather 成本；merge 有 co-rank、局部 gather/scan、初始化及跨 program atomic 成本。收益仍是可证伪假设，不能凭张量形状或源码完成宣称性能提升。
