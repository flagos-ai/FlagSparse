# CSR non SpMV 定向优化候选：设计与独立评审说明

- 编写日期：2026-11-16。
- 目标库/分支：`FlagSparse-main / zyq_sparse`。
- 目标操作：原生 CSR，`y = A @ x`，即 `op="non"`。
- 文档状态：**设计提案，三个候选均未实现、未编译、未进行GPU正确性或性能验证。**
- 面向读者：负责算法评判、Triton实现、平台适配和测试验收的 Agent/开发者。
- 名称均为建议名称，不表示已注册公共API。

## 1. 结论与建议顺序

建议先完成成本诊断，再依次评审两个简单候选：

1. `csr_row_scalar_strip`：每个逻辑元素独占一行，取消行内归约，面向极短行。
2. `csr_row_static_tile`：小行组在固定长度范围内一次加载、一次归约，面向行长集中的短中行。
3. `csr_head_tail_queue`：普通行直接计算，仅为超长行生成分段任务，面向少量长行承担大量NNZ的矩阵；**工程与算法风险高，需通过本文第6节的设计门槛后才能实施**。

这些不是学术原创性声明，而是相对当前库可独立对照的实现候选。不得仅换名称复用旧执行路径，也不得通过修改旧kernel来消除原始对照。

本轮不建议继续叠加 refill、更多预取或更多累加器。先证明目标瓶颈，再增加机制。

## 2. 已有证据及其限制

### 2.1 当前算法的角色

当前CSR non主力包括 `row_tile`、`row_vector`、`row_subgroup_stream`、`legacy_segbin`；半精度还包括 `csr_subgroup_pair_mixed`。

- `row_tile`：多行×行内lane，组内按最长行循环，行内归约。
- `row_vector`：一行一个program，按本行长度循环，最后归约。
- `row_subgroup_stream`：一个program依次处理多个行组。
- `legacy_segbin`：NNZ分块、局部同行分段归约和atomic。
- `csr_subgroup_pair_mixed`：双子块、双FP32累加链。
- `csr_row_refill_reduce`：行完成后动态领取下一行，但增加scan、状态维护和频繁归约。
- split/adaptive/NNZ候选：提高长行或NNZ负载均衡能力，但有计划、部分和及归并成本。

### 2.2 11.15数据提供的观察

完整定义和计算见工作区 `test_mixed_data/11.15data/analysis/`：

- `best_family_summary.csv`：旧算法全集最佳与加入新算法后的全集最佳。
- `best_family_per_matrix.csv`：逐矩阵最优算法及耗时。
- `spmv_csr_non_gm.csv`：non按类型和索引组合汇总。
- `REVIEW.md`：准确性、baseline与数据完整性问题。

代表性观察：int32/int32下regular non的最佳GM对厂商加速比约为FP32 0.768×、FP64 0.792×；mixed宽输出约0.48×，但只覆盖10个矩阵。`mip1` regular约0.30×、`ASIC_680ks` mixed宽输出约0.29×，是优先调查对象。

**不可由这些数值直接推导矩阵结构或硬件瓶颈。** mixed存在未完成测试及FP16输入溢出，部分算法和厂商结果准确性FAIL。性能统计保留有效耗时，不等于正确性已验收。

refill在全部索引组合中仅有同一矩阵的4个non最优条件，收益约0.09%～3.02%，且3条准确性FAIL。这不足以支持扩大动态补位机制。

### 2.3 必须补充的结构与时间诊断

令M为行数、Z为NNZ、`L[r]=indptr[r+1]-indptr[r]`。至少记录：

| 指标 | 定义/用途 |
|---|---|
| 空行率 | `count(L=0)/M`，M=0单独处理 |
| 行长分位数 | P50/P90/P99/max，区分普通行和长尾 |
| 短行比例 | `count(L<=k)/M`，k取1/2/4/8/16/32 |
| 长行NNZ占比 | `sum(L[L>T])/Z`，Z=0单独处理 |
| 行组静态利用率 | `sum(L_group)/(R*K)`，只对全部L<=K的组定义 |
| 行组循环有效比例 | `sum(L_group)/(R*V*ceil(max(L_group)/V))`，分母为0记为空组 |
| x访问特征 | 相邻行列索引重用、工作集及局部性，仅在离线分析/profiler中获取 |

上述统计用于实验筛选，不自动加入每次run，不引入以平均行长为唯一依据的路由。

时间必须区分：完整run、实际kernel设备时间、初始化/转换/计划、CPU提交与等待。事件包围Python执行函数的`gpu_ms`可能包含设备空闲的提交间隙，不能直接称为纯kernel时间。额外直绑kernel诊断不得替代正式完整运行分母。

## 3. 共同实现契约

### 3.1 范围与类型

候选首先只声明non。以后扩展trans/conj必须单独设计/验证，不能因为CSR公共入口支持转置就声称本候选优化了转置。

建议以通用类型规则实现，保持当前新CSR SpMV数值策略：

| A与x输入 | 计算/部分和 | 输出 |
|---|---|---|
| FP16/BF16 | FP32 | 同类型或FP32 |
| FP32 | FP64 | FP32 |
| FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | complex64 |
| complex128 | FP64实虚分量 | complex128 |

不扩展异类型A/x或任意计算精度。最终store前不得窄化部分和；不能将数值策略变更混为算法提速。

### 3.2 接口与生命周期

- 沿用原CSR API、prepare/run、`alg/config`、查询与benchmark；新增候选独立注册，不新增顶层算法函数。
- 保留默认路由、旧算法参数、缓存和prepare生命周期。
- 输入保持原生CSR，不coalesce、不重排、不丢重复项/显式零。
- out按真实stride写回；拒绝与原始输入存储重叠，保持返回对象身份。
- 索引和地址使用足够宽度。标量stride/维度使用兼容Triton标量特化的写法，例如 `tl.cast(stride, tl.int64)`；不能对可能为Python整数的参数直接调用`.to`。
- 只有明确索引兼容错误才允许同算法安全回退，且保留配置、op和精度。
- 后端能力来自main现有能力接口。逻辑张量维度不等于固定物理warp布局，CUDA/ROCm需分别验证。
- 非法显式配置报错；默认参数是实验起点，不是已调优结论。

### 3.3 初始实验参数（待审查）

| 候选 | 起始值 | 首轮消融 |
|---|---|---|
| scalar-strip | R=128、2 warps、stage=1 | 固定warp比较R=64/128/256 |
| static-tile | R=8、K=32、2 warps、stage=1 | 固定R比较K=8/16/32/64，再比较R |
| head-tail | 长行阈值T与分段长度S待结构统计确定 | 先证明目标长行NNZ占比和队列成本，不预设通用默认 |

R/K需为合法二次幂并受后端资源限制；不要把单个program的张量元素数误认为线程数。

## 4. 候选A：csr_row_scalar_strip

### 4.1 算法与正确性

一个program负责连续R行，维护一维张量：`row[R]、pos[R]、end[R]、acc[R]`。每个逻辑元素独占一行。

伪代码（说明数学执行，不是可直接复制的Triton代码）：

```text
row = pid * R + arange(R)
pos = masked_load(indptr[row], row<M)
end = masked_load(indptr[row+1], row<M)
acc = 0
while any((row<M) and (pos<end)):
    valid = (row<M) and (pos<end)
    col = masked_load(indices[pos], valid)
    a = cast_to_compute(masked_load(values[pos], valid))
    xval = cast_to_compute(masked_load(x[col*x_stride], valid))
    acc += a*xval
    pos = minimum(pos+1, end)
masked_store(out[row*out_stride], acc, row<M)
```

复数使用实虚两个acc。尾部和空行的输入加载全部mask；空行acc=0并最终写零。每行从start到end依次执行一次，program行区间互斥，因此无漏项、无重复写、无浮点atomic、无独立输出清零。

### 4.2 与旧算法的实质差别

旧tile/subgroup为一行安排多个逻辑lane，再沿行内维度归约。本方案取消行内维度和浮点归约，扩大同时处理的行数。它不做refill；一行完成后该位置失活，直到整个行组完成。

### 4.3 成本与适用范围

收益假设：长度0/1/2/4的行占比高时，减少行内归约及空lane工作。

新增/保留成本：

- 行组终止判断仍需一次布尔/整数归约，不能称为完全无归约。
- 相邻逻辑元素访问不同CSR行，value/index地址不保证连续，合并访存可能变差。
- 行长变大时，单行累加链串行且x加载依赖更明显。
- 混入少数长行时，仍存在组内等待；本方案不解决长尾。

优先选择极短行且组内离散程度低的矩阵；不因其支持长行就把长行矩阵列为目标。退出条件：节省归约不足以抵消不规则加载，或目标子集完整调用无稳定收益。

## 5. 候选B：csr_row_static_tile

### 5.1 快速分支

每个program负责连续R行，读取行长。若有效行全部满足L<=K，则构造 `[R,K]` 地址张量，一次加载values/indices/x、计算乘积、沿K归约并写出R行。

```text
start,end = row pointers for R rows
if max(masked_length) <= K:
    p = start[:,None] + arange(K)[None,:]
    valid = valid_row[:,None] and p<end[:,None]
    product = load(values,p,valid) * load(x,load(indices,p,valid),valid)
    result = reduce_sum(product, axis=K)
else:
    result = complete_loop_over_all_chunks(start,end,K)
store every valid row
```

所有乘法前显式提升到计算类型；复数显式实虚运算。无效元素为零。快速分支完整覆盖每行，循环分支也必须完整覆盖长行，分支结果只写一次。

### 5.2 长行分支与身份

长行分支是本候选内部的正确性保障，不调用另一个公开算法，不更改元数据中的算法身份。首版允许复用纯数学helper，但不能修改旧row_tile计算体。

动态条件是program一致的标量条件；两分支输出形状/类型一致。该结构是否值得保留需要测量分支开销。若最终与旧row_tile在目标条件生成等价机器码，应认定没有形成有效差异，而不是宣称新优化。

### 5.3 成本模型

定义目标行组的利用率 `U=sum(L)/(R*K)`。收益假设是移除运行时循环、暴露固定小块中的独立加载与计算；不保证减少实际内存事务。

成本包括行组最大值归约、快速/循环分支、padding逻辑元素、较大的活跃张量和归约布局转换。K增大可能降低occupancy或产生spill；不允许通过无限扩大K覆盖最长行。

首轮固定R/warp，仅比较K；记录快速分支命中率可在独立诊断中采集，不能为元数据在正常run额外读回GPU计数。

适合行长集中在一个小范围、U较高的矩阵。长尾混入使组进入慢分支时，收益可能消失。

## 6. 候选C：csr_head_tail_queue（高风险，设计门槛未通过）

### 6.1 目标与不可省略的问题

目标不是“很多长行”，而是多数普通行能够快速完成、少数超长行承担显著NNZ比例。仅建立长行列表、每个worker领取整行不能解决单超长行并行不足；任务必须是分段。

建议使用明确的kernel阶段边界，避免跨program自旋、生产者/消费者同kernel死锁和虚构全局屏障。

### 6.2 可供评审的执行草案

阶段0：初始化整数任务计数器及必要状态。输出不必全清零：普通行与尾部归并分别负责自己的完整输出。

阶段1：融合普通行计算与长行发现。

- 读取原CSR行长；L<=T的行走独立普通行实现并直接写输出，空行写零。
- L>T的行计算q=ceil(L/S)，通过整数atomic一次预留q个连续任务槽位。
- 写入任务描述符 `(row, segment_start, segment_end, partial_slot)`，同时记录该长行的任务起点和q。
- 不做全矩阵排序或三类行列表，不同步读取CPU计数来决定分配。

阶段2：固定数量worker读取最终任务计数，领取分段任务，计算后写独占部分和。只消费阶段1已完成的数据；阶段间依靠同流顺序提交。

阶段3：对每个长行的连续部分和执行有界/分层归约，最后写其输出；普通行不再被写入。半精度仅在最终写回时转换。

**这只是可评审的草案，不是已经闭合的实施方案。** 阶段3如何避免扫描大量普通行、如何设置层级容量，必须在开始编码前明确。

### 6.3 描述符容量与最坏情况

长行数为Q，长行NNZ为Z_tail，任务数D满足：

`D = sum_long ceil(L/S) <= Z_tail/S + Q`。

仅用已有M、Z及阈值T可构造保守容量：

`Q <= min(M, floor(Z/(T+1)))`，
`D_cap = ceil(Z/S) + min(M, floor(Z/(T+1)))`。

该上界可能明显过大，**有界不等于低开销**。必须分别计入描述符、部分和、归并缓冲、计数器及长行元信息；检查整数宽度和工作空间上限。不能用动态计数回CPU之后重新分配来冒充免同步设计。

首版若超出空间上限，可以明确拒绝配置；不能偷偷换算法。若希望分批，则需另行给出批次边界、跨批FP32/FP64累加及容量证明，不能在本草案中宣称已解决。

### 6.4 与现有split/adaptive区别及否决条件

潜在区别是普通行计算与长行发现融合、仅长行产生任务、不全矩阵前缀和和分类列表。并非仅将旧descriptor构建换个函数名。

新增成本：整数atomic、队列状态、描述符写入、多阶段launch、worker调度和部分和归并。一个超长行若在阶段1由单个program串行写大量descriptor，还可能产生新瓶颈。

在以下问题回答之前不建议实施：

1. Q和D_cap是否足够小？是否需要无界的M级数组？
2. 阶段1普通行是否被长行描述符生成拖慢？
3. 阶段3是否又重新产生昂贵的全矩阵计划？
4. 相比现有split完整run，减少的成本是否超过新增整数atomic及launch？
5. 单超长行、全长行、全空矩阵的终止与容量是否成立？

## 7. Triton机制：可以使用什么，不能假定什么

### 7.1 静态形状与编译期分支

利用`tl.constexpr`固定R/K、实数/复数分支与计算类型；避免在实数路径保留无用复数状态。`tl.sum`、scan、gather都可能涉及布局转换，其源码形状不能代替机器码成本。

### 7.2 对齐与连续性

可以单独研究对齐后的稀疏数组加载，但CSR每行起点不保证对齐。`tl.multiple_of`和`tl.max_contiguous`是事实提示，不是运行时修复。不能对任意行起点、任意`indices`或`x[indices]`伪造对齐/连续性承诺。

对齐读取若扩展了范围，必须严格mask，不得读到数组外；多读的元素成本必须统计。

### 7.3 缓存与预取

A/value/index通常顺序读取，x是不规则gather，两者应分别分析。NVIDIA PTX缓存修饰符不能未经验证推广到HIP。增加`num_stages`或先写两次load不等于硬件异步拷贝；必须检查真实加载等待、寄存器、spill及执行时间。

### 7.4 mixed的收益上限

半精度降低value及部分x的存储字节，但int32索引仍占4字节、累加仍FP32、宽输出仍FP32。不能预设FP16/BF16获得两倍速度。直接加载原始输入、在消费时提升、避免整份高精度副本和无效输出复制，是应审计的实现细节；不能靠提前窄化换取性能。

首轮不引入Tensor Core、Gluon、平台专用汇编或格式转换。

## 8. 测试、计时与验收

### 8.1 保持原交付范式

实施时修改既有kernel/config/runtime及原CSR accuracy/策略测试，不新增Python框架；原API、benchmark compare与`run_flagsparse_pytest.py`接通。CUDA计算节点继续通过现有`tools/run_cuda_tests.sbatch`进入环境，DCU沿用原runner。本文不提供尚未注册算法的可执行命令。

### 8.2 正确性重点

- 六种dtype、四种半精度输入/输出组合、合法混合索引。
- 空矩阵、空行、矩形、重复坐标、未排序列和显式零。
- scalar-strip：全空组、单行长尾、R边界、长度0/1/2/4及更长行。
- static-tile：K-1/K/K+1、全快分支、混入长行、R尾部、循环分支覆盖。
- head-tail：T/S边界、单超长行、全长行、任务计数及容量边界、归并多层、无任务。
- NaN预填out、strided x/out、返回身份、输入不变与别名拒绝。
- 强抵消及动态范围；宽输出确实保留计算精度，不能先转半精度再转FP32。
- 配置隔离、未知后端能力、安全索引回退。

保持共享容差。输入溢出、参考溢出、有限值准确性失败、编译/执行错误分别记录。只有明确能力缺失允许SKIP。

### 8.3 计时与性能判据

正式计时保持 `ms=process_cpu_ms+gpu_ms`。完整性能采样与分段诊断分开。所有每次必须发生的初始化、转换、队列、计算及归并计入run；不能移到prepare冒充收益。

固定矩阵、dtype/输出、op、索引、配置和设备：

`old_best(m) = min(time of all old supported algorithms)`；
`new_best(m) = min(old_best(m), time of new candidates)`；
`gain(m) = old_best(m)/new_best(m)`；
`GM_gain = exp(mean(log(gain(m))))`。

厂商加速比另用`vendor_ms/new_best(m)`；没有vendor时间不伪造。保留精度FAIL的有效性能并独立标识，但正确性未通过不能推荐部署。缺失新候选的条件不能作为完整配对成绩。

先小矩阵、再一个真实矩阵、最后完整30矩阵。报告目标子集与全体GM/AM、最佳占比、单算法相对旧最佳的回退、准确性状态及重复采样稳定性。全算法最优集合包含旧算法，因此整体最佳在同批样本中不应变慢；跨批噪声或输入/条件变化必须单独排查。

### 8.4 机制验证

| 假设 | 需要的证据 | 否定信号 |
|---|---|---|
| scalar减少归约成本 | 归约指令与完整时间下降 | 加载事务/等待上升抵消收益 |
| static暴露独立工作 | 循环控制减少且无明显spill | 与旧kernel等价或寄存器成本更高 |
| head-tail减少计划开销 | 全流程计划/提交成本下降 | descriptor/队列/归并比原计划更贵 |
| mixed临时量更合理 | 没有全量转换副本，寄存器/内存证据支持 | 临时量增加或编译器已自动优化旧路径 |

## 9. 独立Agent评审清单与交付要求

请审查者逐候选给出“接受/修改后接受/否决”，并回答：

1. 是否与当前注册算法具有实质差异？是否只是参数特化或生成等价代码？
2. 索引集合是否完整互斥？空行、尾部、复数及宽输出是否正确？
3. Triton逻辑张量能否合法表达？是否错误假设物理lane映射或异步流水？
4. 哪些成本减少，哪些增加？目标矩阵结构是否已有数据证明？
5. 是否保留旧数值策略、prepare生命周期、输出/回退契约？
6. 是否有GPU→CPU读回、隐藏格式转换、无界空间、跨program死锁或重复清零？
7. 是否应先做kernel诊断，而不是立刻增加算法？
8. 最小可证伪实验是什么？何种结果应停止投入？

建议返回格式：`候选 / 结论 / 关键证据 / 阻塞问题 / 必要修改 / 最小实验`。head-tail的任务布局和归并未闭合，不能仅凭概念认可就判为可实施。

评审记录：待填。源码实现：未开始。硬件正确性：未验证。完整调用收益：未验证。

## 10. 参考资料与借鉴边界

1. [CSR-Adaptive](https://computermachines.org/joe/publications/pdfs/sc2014_csr-adaptive.pdf)：借鉴根据行结构分工的思想；不假设其预处理成本能在当前按次执行中摊薄。
2. [CSR5](https://arxiv.org/abs/1503.05032)：借鉴规则tile与分段归约的负载均衡思想；其格式/描述符机制不能直接冒充零准备成本的原生CSR。
3. [Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)：mask与缓存修饰符语义；后者涉及NVIDIA PTX，不等于通用后端保证。
4. [Triton max_contiguous](https://triton-lang.org/main/python-api/generated/triton.language.max_contiguous.html)：连续性提示必须满足真实输入性质。

官方main文档可能与计算节点Triton 3.6不同；实施前必须核对目标版本源码/编译支持。文献中的设备、精度、格式及计时边界不同，不引用其加速比作为本库预期收益。
