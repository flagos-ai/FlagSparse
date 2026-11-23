# CSR SpMV 三类任务协作候选与验证计划

日期：2026-11-22。状态：独立算法源码、注册、benchmark与pytest覆盖已接入；仅完成静态审查，未编译、未运行。本文不包含计算节点完整命令，命令在会话中交付。

## 1. 证据、目标与边界

证据来自 `test_mixed_data/11.22data/analysis/REPORT.md` 及其原始 CSV，而不是只看 runner summary。

- mip1 长度大于1024的行只有2.12% NNZ；129～1024长度范围却包含81.04% NNZ。当前协作 producer 仍承担这部分主要负载。
- wiki-Talk 大于1024的行包含53.87% NNZ，同时存在2246783个空行。当前R8/V16 producer既遍历全部行，又让普通行组按组内最大行长循环。
- BF16宽输出的长行consumer从P1到P8，mip1/wiki分别改善约3.54/3.83倍；但完整调用仍未稳定超过旧最佳。
- 当前有界S128与旧默认S32同时改变了阈值和consumer；不能把全部收益归因于网格限制。旧S128和重复P8必须补为同批对照。

新增候选名称 **`csr_three_tier_queue_reduce`**，独立注册、独立执行分支。原有 kernel、默认参数、默认路由、compact/fixed、缓存与图绑定均不改变。

首版仅声明 `op=non`。trans/conj不得静默进入其他算法；将来若扩展，另行审查转置生命周期。CUDA/ROCm沿用当前普通CSR候选的能力来源，不扩大后端支持声明。

覆盖FP16、BF16、FP32、FP64、complex64、complex128，同型A/x；半精度输出为输入类型或FP32。计算策略保持：半精度FP32，FP32/FP64输入FP64，复数使用对应分量精度。显式算法不得改走mixed专用kernel。

## 2. 四个阶段与所有权证明

每次run重新分类，不跨调用缓存，不生成全NNZ owner-map，不转换格式、不排序、不读回GPU计数。

设行长为 n，阈值满足 `0 <= S < L`：

| 类别 | 区间 | 输出责任 |
|---|---|---|
| 短行 | 0≤n≤S | 融合分类producer直接计算并写回，包括空行 |
| 中长行 | S<n≤L | 中长行队列consumer独占整行，最终归约后写回 |
| 超长行 | n>L | P个program写独占部分和，finish最终归约并写回 |

区间互斥且完整。每行恰好属于一个类别，输出无需独立全矩阵清零。队列仅保存原行ID，不复制CSR值或列索引。

### 2.1 初始化

分配两个int32计数器并清零；分配int64中长/超长行ID缓冲，以及计算精度的超长行部分和。计数范围必须经过容量检查，不允许计数溢出。行号、行指针、地址乘法和计数到地址的转换保留所需int64宽度。

### 2.2 融合分类和短行计算

一个program处理R个连续行，加载行起止，生成三个互斥mask。

- 中长、超长标记各执行一次整数cumsum；非空类别每program一次计数atomic_add，取得该类别连续队列区间；仅有效行写入队列。
- 短行计算使用 `[R]` 标量累加器，按该组短行的实际最长长度循环，循环上限不超过S。
- 不让中长行或超长行参加该循环，也不为每个空行维护V个lane。
- 实数只保留实数累加器；复数保留两个对应计算精度的累加器。
- 所有CSR、x读取分别mask；空行直接写0；尾行不访问越界行指针。

R增加可减少wiki的分类program数量，但可能增加短行组内等待和整数scan成本。因此R=128是起点而非已验证最优，必须与R64对照。

### 2.3 中长行整行向量consumer

首版每program独占一个队列槽位，读取GPU count并判断有效性。grid使用保守中长行容量；有效program按K个NNZ循环，lane内累加、最后一次归约。没有浮点atomic、没有中长行部分和，也没有finish。

此选择故意不默认限制到小G：先验证将mip1的大量中长行移出producer是否有效，不同时引入整行串行多轮。容量导致的空program是明确代价；已有G实验若证明有界领取更好，再以同候选显式策略做独立消融，不能无证据设置默认。

### 2.4 超长行协作与finish

consumer采用二维grid `(min(C_long,G_long), P)`；同一队列行的P份按K块平衡分割，slot按行grid步长推进。

令 `chunks=ceil(n/K)`，第p份区间为：

`[start + floor(chunks*p/P)*K, min(end, start + floor(chunks*(p+1)/P)*K))`。

每份只写 `[slot,p]` 对应的计算精度部分和；空份也必须写0。连续区间完整覆盖且互斥。同一设备流随后启动finish，每行归约P项后按out stride写回。

没有跨program自旋，没有隐含全局屏障，没有浮点atomic。不同kernel之间依赖实际执行流顺序。全部中间结果保持计算精度，半精度仅在最终store窄化。

## 3. 保守配置、容量与内存上限

暂定起点，不依据未完成消融宣称最优：

| 字段 | 起点 | 校验/说明 |
|---|---:|---|
| rows_per_program | 128 | 32/64/128；producer使用 |
| short_row_threshold | 32 | 0/8/16/32/64/128；必须小于L |
| long_row_threshold | 1024 | 正整数且大于S；验证128/256/512/1024 |
| block_nnz | 128 | 64/128/256；首版中长/长行共用 |
| cooperative_parts | 4 | 1/2/4/8/16；P1保留机制对照 |
| long_grid_blocks | 128 | 正整数，受当前合法上限约束 |
| producer_num_warps | 2 | 后端合法集合选择，显式非法报错 |
| num_warps | 4 | 中长/超长consumer，后端合法性检查 |
| num_stages | 1 | 首版固定 |
| workspace_limit_bytes | 268435456 | 包含全部内部队列、计数器和数值部分和 |

没有独立新调优器，不增加运行时搜索；改变P时不偷偷改变K或L。

无需读取GPU计数的安全容量：

`C_mid=min(M,floor(nnz/(S+1)))`

`C_long=min(M,floor(nnz/(L+1)))`

前者把超长行也纳入上界，故可能过度分配，但绝不低估。不能把这个上界当作实际中长行数。

临时空间上界（不含调用方out）：

`8 + 8*(C_mid+C_long) + C_long*P*component_bytes*components`

实数components=1，复数=2。任何新增内部缓冲必须加入计算。容量为0可从CPU已知维度安全取消对应launch；容量非0但实际队列为空时仍按当前无读回策略运行。超过int32计数范围、空间上限或分配失败明确报错，不换算法、不隐式CPU同步后重新计划。

相较当前协作，新增一条中长队列及consumer；期望减少producer的中长行工作和空行program数。只有减少的完整时间大于新增分类、queue、launch、访存成本，方案才成立。

## 4. 接口、计时与记录

- 进入现有 `_spmv_csr_config.py`、`_spmv_csr_kernels.py`、`spmv_csr.py`；不新增Python文件或顶层算法API。
- 规格、查询、prepare/run、高层入口、compare共用名称与能力判断；non以外显式报错。
- 现有out形状、dtype、stride、原始输入别名校验保留；算法切换不继承旧专用配置；安全索引fallback保留算法、配置、输出与精度。
- 能力需覆盖整数scan、整数atomic、对应实数计算/归约和索引；无需浮点atomic。能力未知明确不可用。
- prepared只保存原始矩阵与固定选择，不保存两条运行队列。
- `ms=process_cpu_ms+gpu_ms`；正常CPU算法处理为0。完整run包括初始化、分类/短行计算、中长consumer、超长consumer、finish及必要输出处理。
- 初始化诊断独立记录；融合分类与短行计算不能虚构拆分成纯process和纯compute。保留 `fused_plan_compute_ms`、medium/long/finish诊断；完整采样不插入这些分段事件。
- 元数据记录实现版本、S/L/R/K/P、两个容量、grid、数值/描述符空间、精度、输出身份与stride。实际队列数不在正常run读回；CPU结构预测必须注明来源。
- 默认算法、旧缓存、专用图/直接绑定不扩大。新路径先经正常run调用。

## 5. 现有算法消融与实施门槛

命令交付分regular四类型与mixed四组合，均使用原sbatch包装器和runner，保留vendor、timing、perf-metrics。

1. 每类一次compare建立同批旧最佳。S128旧队列、有界S128/G256、协作L1024/P8三组轮换次序，各重复3次。
2. G=128/256/512/1024；S128、R64、K128固定。G256已在重复对照中覆盖，不重复无意义测量。
3. L=128/256/512/1024与P=1/2/4/8交叉，共16组；R8/V16、G128、K128固定。先以P4列判断L，再以同L行判断P，不能只从16次中挑最低值就认定可复现。
4. 显式配置每次只绑定一个算法。不要将专用config传给compare。两个模式和不同配置依次执行，不在同一GPU并行。
5. 比较算法时固定同矩阵vendor分子；独立报告自身耗时比。四种mixed输入指纹、配置、基线/自身精度分别核对。保留所有有效FAIL耗时，误差回归仍必须定位。

现有参数实验若已覆盖目标收益，应先保留该低成本结论，不为新增名称强行增加kernel。若仍显示producer瓶颈，则推进三类候选；不以P1的自我加速代替超过旧最佳。

## 6. 测试和源码验收计划

扩展原 `test_spmv_csr_accuracy.py` 与既有策略测试；marker、quick/normal、共享golden/容差、JSON与runner不另建体系。

- 每个dtype、四种half组合；空矩阵、全空队列、每次仅一个类别非空、三类交错。
- 长度0、1、S-1/S/S+1、L-1/L/L+1、K边界、P份边界；行组尾部及队列容量边界。
- 超长行、重复列、未排序列、复数抵消；NaN预填与strided out、返回身份、原始输入不变和重叠拒绝。
- 容量不足、非法S/L、能力未知、配置隔离和同算法索引回退；int32/int64合法混合索引。
- 所有队列ID唯一；每行仅一条路径写回；跨run没有缓存数据。可以在测试中读取计数/队列验证，正常性能run不得增加读回。
- 首版non声明与compare过滤一致。编译/运行故障FAIL，只有明确能力不足SKIP。
- 不修改原有JIT/C API签名，不修改旧计算体；更新原registry、支持表、benchmark元数据和相关文档。

本地只源码、AST和diff；计算节点先smoke，再一个真实矩阵，再全部30矩阵。六种dtype和四种half组合正确性完成后，性能优先FP32/FP64及half宽输出；同型half输出与复数保留完整回归。

## 7. 性能判定与停止条件

困难目标为mip1/wiki/TSOPF；neos/wheel和roadNet为空/稀队列控制组。旧longrow atomic与NNZ-local必须保留为困难矩阵对手。

重复后的目标完整时间相对旧最佳稳定改善约1.2倍作为继续投入门槛；收益必须明显超过批次波动。再计算30矩阵等权GM/AM、逐矩阵最佳增量、胜出与回退，不将离线择优当作已经接入自动选择。

若新增中长consumer成本抵消producer收益，停止扩展同一结构；若wiki仍由空行分类主导，先看program数、指针访问和实际提交成本，不继续堆P。没有证据时不切默认，不宣称寄存器/访存已优化。

新算法源码完成、硬件正确性、完整性能收益分别记录。未验证候选不替换旧算法。

## 8. 源码接入与实际差异

- `_spmv_csr_config.py`：独立名称、non-only规格、六种dtype/四种半精度组合、后端能力、扁平配置与 `three_tier_queue_capacity`。
- `_spmv_csr_kernels.py`：新增 `three_tier_producer_kernel` 与 `run_three_tier_queue`。后两类复用未改动的 `short_rowqueue_vector_kernel`、`cooperative_row_consumer_kernel` 和 `cooperative_row_finish_kernel` 数学计算；没有调用这些旧算法的run，也没有改写旧kernel计算体。独立性体现在新分类、两条队列、不同任务分配及运行元数据，不通过改名包装旧run。
- `spmv_csr.py`：原prepare/run、输出类型与安全索引回退路径接入；正常run两条队列重建，宽输出直接接收计算结果。
- 原regular/mixed benchmark输出增加medium/long容量、grid、两个consumer诊断字段。`csr_tile_structure` 增加独立三类模型，不再将新配置误识别为只有 `grid_blocks` 的旧协作配置。结构统计在计时外，明确为CPU预测。
- 原accuracy文件增加 `test_spmv_csr_three_tier_native`（regular/mixed明确ID）、三条路径的宽输出测试；复用通用边界、抵消及索引fallback参数化。测试中检查队列归属、按次分配、strided out、别名、工作空间和部分和初始化，正常run不做这些读回。
- 原策略测试覆盖容量、后端、非法配置、non过滤、配置隔离和结构模型；原GPU CI交付检查增加名称。公开查询/两级函数导出沿用原API，无新顶层函数；runner按原marker及compare枚举，无需新增参数。

`consumer_ms` 诊断为 `medium_consumer_ms + long_consumer_ms`，便于旧表头兼容；它不是完整gpu_ms的替代。全部精度FAIL记录和厂商baseline路径保持原规则。

静态检查不验证Triton编译、GPU数值或性能。CUDA/ROCm smoke及真实矩阵验收待计算节点结果，支持表保持UNVERIFIED。

## 9. 2026-11-23：有界中行独立候选

### 证据与范围

基线分支 `zyq_sparse`，commit `057f0b4ec564860ef44bf7d06d94d8c2f8ec6621`，实施前工作区干净。
`test_mixed_data/11.22-newdata/analysis/REPORT.md` 记录了240条件零次最佳，以及两组smoke的
strided out验证失败。先修复 `_spmv_check_output` 的THREE_TIER算法集合遗漏，保留正stride、
形状/类型/设备、lazy view和输入存储别名检查，不把测试改成连续out。

新增 `csr_three_tier_bounded_reduce`，保留 `csr_three_tier_queue_reduce` 的实现、默认参数和默认路由。
范围仍为CUDA/ROCm、non、六种输入dtype及四种半精度组合。半精度FP32累加，FP32输入沿用FP64，
复数沿用对应实数分量精度。未知后端能力明确排除。

### 唯一的调度差异

新run使用原三类producer，两条队列保持每次调用重建。设中行容量上界为C，有效行数为Q，
显式配置上限为G，则实际中行网格为 `g=min(C,G)`。program p读取GPU有效count，依次处理
`p, p+g, p+2g, ... < Q`，每行仍由原整行向量数学内核完成一次最终归约并直接写out。

- 新配置 `medium_grid_blocks=512`，合法整数1～1024（拒绝bool/浮点/字符串）。
- 其余默认保持S32/L1024/R128/K128/P4/长行G128；producer 2 warps、consumer 4 warps、stage1，按原能力解析。
- 独立入口 `run_three_tier_bounded`，不调用旧run。复用未修改的
  `three_tier_producer_kernel`、`short_rowqueue_strided_vector_kernel`、
  `cooperative_row_consumer_kernel`、`cooperative_row_finish_kernel`。
- 极长行不进入中行CTA，继续P份协作、独占部分和、最终归并。不增加全局任务领取atomic、
  浮点atomic、CPU count读回、缓存或转格式。
- producer、两类consumer、finish仍分开提交到同一执行流。本轮不同时融合kernel或改动P1写回方式。

正确性：每个中行queue slot与一个p模g余数唯一对应，每行在本CTA中完整归约；短/中/长分类互斥，
所有行恰好写一次，空行由producer输出零。累加器在每条新中行开始时清零，地址和slot使用int64。
长行部分和全部在finish之前提交完成，没有跨CTA自旋屏障。合法strided out直接按YS写回，
半精度只在最终store窄化；FP32宽输出不会经过半精度中间结果。

容量和工作空间没有因G缩小而虚报减少：仍为两条行ID队列和长行计算精度部分和，沿用原256MiB限额。
减小的是启动CTA数量，不是队列容量、有效工作量或GPU读回值。

### 成本假设和风险

wiki-Talk旧中行capacity=152163、Q=10347；G512时由152163个CTA改为512个program，
最长约21轮中行任务。mip1容量66463、Q37347，同配置最长约73轮。上述只预测任务数，不预测耗时。
有界调度可能减少空CTA/重复调度，但也可能减少并行度，或因queue顺序不同造成尾部不均。
长行独立P/G消融仍必要：新旧compare里P4/P8不同，不能把所有差异都归因于有界网格。
全短行矩阵仍支付空consumer的launch，本候选不争取替代scalar/static全局最优。

元数据新增 `medium_grid_limit`、`medium_rounds_capacity`；后者为容量轮数上界，不是实测轮数。
结构诊断另记 `expected_medium_active_programs/empty_programs/max_rounds`，来自CPU真实行长。
`task_acquisition=medium_bounded_grid_stride_long_fixed_parts` 区分旧capacity版本。
常规与mixed CSV均保留这些字段及各阶段诊断；不新增GPU同步获取精确统计。

### 原测试链路与验收

原 `test_spmv_csr_three_tier_native` 扩展capacity、bounded G1/G4三组，保持regular/mixed可读ID。
所有dtype、P1/P4（normal另P16）、七种边界形状、strided out、两类queue归属、重复run、
NaN部分和、输入不变、配置隔离、工作空间限制均覆盖；G1/G4强制多轮消费。
原宽输出、半精度fallback和通用fallback参数化加入新名称。CPU策略测试补充合法配置、
old/new专用参数隔离、结构模型、执行入口和实际out guard回归，GPU CI注册检查加入名称。
原runner的marker/compare自动枚举，无新增runner开关，也不改变默认测试交付。

执行顺序：先regular/mixed smoke（包含修复后的旧capacity版本），再normal；性能分别进行
全30矩阵compare及独立S128。P4/8/16固定G中512/G长128；随后固定P8比较G中256/512/1024，
再固定P8/G中512比较G长64/128/256。同批保留旧capacity P4/8/16，避免P混杂。
全部四种regular dtype和四种mixed组合分别测试，non/int32，baseline及perf-metrics保持开启。
完整命令交付在对话中，不在文档重复维护。

评价继续使用 `ms=process_cpu_ms+gpu_ms`；诊断另行执行；精度FAIL的有效速度保留。
统一compare厂商分子，分别计算旧最佳含S128、新默认最佳、显式配置候选最佳，不能将事后配置
最优冒充默认路由收益。先看mip1/wiki目标矩阵，再看全体GM、胜出与回退，TSOPF作为控制。
未经新数据验证，不宣称有界中行已经提速，不修改默认算法。

本轮静态验收：修改Python文件AST通过，`git diff --check`通过；与基线AST逐函数比较，
原48个kernel/runtime函数均保持不变，仅新增 `run_three_tier_bounded`。
未运行功能导入、策略测试、pytest、算子或离线Triton编译，未暂存/提交。
源码接通、硬件正确性与性能收益仍作为三个独立状态记录。
