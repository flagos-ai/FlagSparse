# CSR SpMV 短行/中长行与 CSC SpMM non 定向优化方案

- 日期：2026-11-16。
- 目标：`FlagSparse-main`，`zyq_sparse`。
- 状态：短行A/B与CSC D已完成源码接入和静态检查；中长行C待验证门槛。未运行Triton编译、pytest、算子或性能测试。
- 数据依据：[11.16 newdata 分析](../../../test_mixed_data/11.16data/newdata/analysis/ANALYSIS_REPORT.md)。其中部分 mixed 时间来自四位小数 stdout 恢复；profiler 文件尚不能支持纯 kernel 结论。
- 与旧文档的关系：细化短行候选，增加无动态计划的固定行分段和原生 CSC 列分段/输出分片；本轮不实施旧草案中的动态 head-tail 队列。

## 1. 目标与决策原则

目标是让新候选补齐现有算法族的性能缺口；不要求单个算法适合全部矩阵，不以超过某个较弱算法作为成功标准。

固定矩阵、输入/计算/输出类型、op、layout、N、索引、平台和计时边界：

```
old_best = min(本批所有旧算法的时间)
new_best = min(old_best, 新候选时间)
family_gain = old_best / new_best
```

同时报告候选自身相对 old_best 的回退、目标子集 GM、30矩阵 GM/AM 和最优占比。全体最优是逐矩阵 oracle，不冒充当前默认路由的成绩。独立重复测量判断小幅优势是否稳定。

本批优先顺序：

1. CSR SpMV 的短行控制/归约成本。
2. CSC SpMM non 的列内负载和输出散射成本。
3. CSR SpMV 中长行中，能够证明存在串行尾部的固定分段候选。

普通中长行先调已有 row_vector 的合法配置，不强行为参数调整起新名字。

## 2. 已有证据与需要避免的误判

| 对象 | 本批证据 | 不能直接推出的结论 |
|---|---|---|
| CSR SpMV 极短行 | roadNet-TX 约99.5%行≤4；ASIC约85.5%；wheel约99.9% | 相邻行一定同长、跨行读取一定连续 |
| CSR SpMV 集中短行 | ecology最大行长5；NACA最大10 | 任意更大tile都能改善带宽 |
| CSR SpMV neos3 | 平均4.01，最长6480，但>1024行的NNZ仅约0.3% | 应为全矩阵建立长行计划 |
| CSR SpMV mip1 | 平均156、p99=713，最长66395；wide最佳0.2968ms，对厂商0.236× | 大多数行需要跨program拆分、最长行已被证明是运行临界路径 |
| CSC SpMM non | wide mip1约0.192×，trdheim约0.210×；tile_panel处理诊断占比中位约1% | 主要问题是prepare；或者已经证实atomic竞争占满全部时间 |
| CSC列分布 | c8_mat11列p99=2091、最大2417；trdheim最大150；ecology最大5 | 所有差矩阵都能由长列分段解决 |

源码复核的两个约束：

- `_spmv_csr_kernels.py` 的 `row_tile_real_kernel` 已有 `FIXED_STEPS`。静态候选必须与合法、同参数旧实现比较；如果编译结果等价，只保留配置建议，不注册重复算法。
- `spmv_csc.py` 的 `_csc_columns_atomic_kernel` 已在列内循环外加载 B 面板。新 CSC 方案不将这项已有复用算成新增优化。

## 3. 候选A：csr_row_scalar_strip

### 3.1 目标与区别

目标是长度0–4占多数、相邻行组长尾较少的场景。旧tile/subgroup给每行多个逻辑lane并作浮点归约；本候选每个逻辑元素独占一行，使用 `[R]` 累加状态，取消行内浮点归约。

它不做refill、不维护领取状态、不生成owner-map或运行计划。逻辑元素不是对物理CUDA线程/AMD lane的一对一承诺。

### 3.2 完整执行

每program拥有连续R行，读取各行start/end。每轮各有效行读取当前位置的value/index/x并累加，位置推进1。直到所有行完成，按真实out stride写回。

```
pos = start; acc[R] = 0
while any(valid_row & (pos < end)):
    active = valid_row & (pos < end)
    acc += product(masked_load(A[pos]), masked_load(x[I[pos]]))
    pos = min(pos + 1, end)
store(out[row], acc, valid_row)
```

这里是算法伪码，不是可直接运行的Triton程序。所有两级间接加载均带mask；无效值明确为0。空行最终写0，program输出区间互斥，无独立全输出清零、无浮点atomic。复数使用实虚两个累加器。

### 3.3 成本、参数与风险

- 减少：行内浮点归约、多lane短行的无效计算。
- 仍有：组级终止判断的整数/布尔归约。
- 增加或可能恶化：跨行分散加载、每行串行依赖、长行污染整个组。

起点R=64、2 warps、1 stage；候选R=32/64/128，warp由后端合法集合校验。不是使用R=128固定覆盖所有极短行矩阵。

离线先计算候选R下：

`scalar_efficiency = sum(row_lengths)/(R * max(row_lengths))`。

全空组另记，不作0/0。它只衡量逻辑工作利用率，不是带宽或occupancy预测。对neos/wheel，必须看相邻组内长尾是否破坏该比值，不能只看全局短行占比。

长行仍完整执行，但不承诺快；首版不再增加额外尾部队列来挽救此候选。

## 4. 候选B：csr_row_static_tile

### 4.1 同时服务集中短行与部分中行

该候选用一个固定二维块 `[R,K]` 计算若干行。有效行均满足L≤K时，一次加载、一次乘法、一次K维归约，直接输出；不执行运行时NNZ循环。

若组内存在L>K，进入候选自己的完整分块循环。两分支都产出同形状 `[R]`、同计算类型结果；不能丢长行尾部，也不调用其他公开算法。

### 4.2 不同目标使用不同参数实验

| 场景 | 首轮参数 | 目标 |
|---|---|---|
| ecology、NACA、CurlCurl等集中短行 | R=8，K=8/16/32，2 warps | 少循环控制，完整暴露小块独立加载 |
| 普通中行 | R=1/2，K=128/256，2/4合法warps | 单次覆盖一行，避免多chunk依赖 |

实部FP64、复数使用更多寄存器，不共享“最优R/K”结论。显式参数不裁剪。地址计算保留所需宽度。

### 4.3 为什么只作为有条件接入的候选

旧row_tile的固定步数或旧row_vector配置也可能形成近似/等价实现。必须先比较：

1. 同R/K/warp、同精度下的旧实现。
2. 固定块快分支的编译结果与旧循环是否还有差别。
3. 快分支命中率、padding比例、寄存器、spill及完整时间。

若旧实现已消除相同循环，该候选不再另注册，改为旧算法显式配置实验。不会为了交付“新算法数量”保留同义实现。

其收益来源只能是控制及编译后执行组织变化，不是少读必需的A/x，也不是“形状更大就更合并”。K过大会增加padding和资源占用。

## 5. 候选C：csr_row_fixed_split_reduce

### 5.1 要解决的问题

针对一行需要很多chunk、单program尾部可能拖慢运行的场景；不是对所有平均行长156的行统一上复杂计划。

它与现有split/adaptive/NNZ的区别是：**固定二维网格和固定部分和位置，无GPU计数、前缀和、描述符压紧或CPU读回**。仍原生CSR，owner天然是row，不二分定位每个NNZ。

首版参数：`segments_per_row=2`、`block_nnz=128`、`split_threshold=256`、合法2 warps、stage=1；H=1/2/4、B=64/128为消融，阈值需至少保证长行有多个chunk。默认只是起点。

### 5.2 行内分工与完整性

对行长L和chunk长度B，令Q=ceil(L/B)，固定H个program槽位：

```
chunk_begin(h) = floor(Q*h/H)
chunk_end(h)   = floor(Q*(h+1)/H)
```

每个h计算上述chunk区间。区间覆盖[0,Q)、彼此不交；最后一个chunk按行尾mask。与先均分L再各自补齐B相比，该划分不会重复引入多份分段尾部padding。

- L≤threshold：只有h=0计算整行并直接写out，其余h不读A/x、不写out。
- L>threshold：各h写独占 `partial[row,h]`，无浮点atomic；空分段写0。
- 第二kernel再次读取行长，只为长行归约固定H项并写out；短行结果不覆盖。
- 输出为空时直接返回。空行由h=0写0；临时部分和的短行位置不用初始化，也不能被归约kernel读取。

数值临时空间约 `M*H*bytes(compute_element)`。复杂类型包含两个分量。默认上限256MiB；超限明确报错，不隐式换算法。H=1作为独立实现控制组，可取消临时缓冲和第二kernel。

### 5.3 成本与继续条件

收益：长行的串行chunk链被分给多个program，省去动态计划构建。

成本：更多program、重复行指针读取、M×H部分和空间、第二次行长扫描和归约launch。短行无用h槽位仍有调度成本，不能称作免费。

需要满足：

`减少的原计算临界路径时间 > 新增调度 + 部分和访问 + 归约时间`。

M足够大时原row_vector可能已有充分并行，因此增加program不保证加速。先用同一block_nnz调优旧row_vector，若已经消除缺口就停止候选C；不能把B变化的收益归因于分段。

目标实验首先是mip1、c8_mat11、smt、TSOPF；只对确实存在长循环尾部的子集继续投入。mip1中>1024 NNZ占比仅2.1%，若它并非运行临界路径，拆分可能无益。

## 6. 候选D：csc_col_partition_panel_atomic

### 6.1 两个因素分开验证

保留原生CSC non，用一个算法内的正交配置区分：

- S=`segments_per_column`：一个列分给几个program。
- H=`output_shards`：输出散射到几个计算精度累加区。

默认先S=2、H=1；测试S=1/2/4和H=1/2/4。S=1,H=1是新实现控制组，不另注册算法。若H>1不赢H=1，不宣传“分片有效”。

### 6.2 原生列段计算

每program处理 `(column, segment, output_panel)`，维护稀疏块 `[K]` 与输出贡献 `[K,BN]`。每个列段独立按实际长度循环，不再等待同一旧R列组的最长列。

令列chunk数Q=ceil(Lcol/K)，按 `floor(Q*s/S)` 到 `floor(Q*(s+1)/S)` 分配chunk，保证每个NNZ仅处理一次。每段先读取该列的B面板，在段内复用，然后按行索引散射。

```
row = indices[p]
contribution = A[p] * B[column, output_column]
atomic_add(Z[shard, row, output_column], contribution)
```

不创建col_ids、owner-map、排序或转CSR。重复坐标全部贡献；空行由初始化覆盖。B/out均按实际stride寻址，支持opB的有效视图，不重复应用转置。

起点 `block_nnz=32, block_n=16, num_warps=4, num_stages=1`；复数128可采用较小BN的后端内置候选。旧算法的R=2、K=16与新参数不同，因此先做同K/BN/warp控制实验，再分离R组等待与S分段的收益。

### 6.3 输出分片

S个列段得到确定的分片编号，例如 `h=(column*S+segment)%H`，不依赖动态队列。

- H=1：初始化计算精度输出并直接atomic。
- H>1：初始化Z[H,M,Nwave]，scatter完成后沿H归约写输出。
- 同流依次提交init/scatter/reduce，不使用跨program自旋或全局屏障。
- FP16/BF16一律FP32累加；FP32宽输出在H=1时可直接使用合法out。半精度输出必须等所有贡献结束才窄化。
- FP32/FP64/复数分别沿用CSC现有分量策略；复数实虚atomic分开，最终组合。

分片**不减少atomic次数**。实数仍约Z×N次，复数是对应分量更新；期望只是把热点分散到多个地址。不能声称它解决所有访存成本，也不把旧B面板复用算作新增收益。

### 6.4 受限空间与面板波次

当不能直接使用out时，数值临时空间为：

`workspace = H * M * Nwave * bytes(compute_element)`。

最终归约直接写out，不再分配第二份完整计算输出。按256MiB默认预算选择Nwave，最少1列；连1列都放不下则明确报错。最终输出自身的存储与数值临时workspace分别记录。

Nwave不足显式BN时仍保留BN，严格mask波次尾部，不静默改用户tile配置。每个波次完成init/scatter/reduce后复用同一缓冲。H=1且可直接写计算精度out时无需该临时缓冲。

分波会重读部分A/index并增加launch，必须计时。特别是大M、小N场景，H>1容易得不偿失，不自动启用。

### 6.5 为什么可能补齐当前最优

- c8/TSOPF等列较长，S可能降低单列串行尾部，并消除多列组互相等待。
- trdheim最大列长150，未必需要S；若主要限制来自跨列输出竞争，H才可能有用。
- ecology列长最多5，S通常徒增空任务；作为回退控制，不是此候选主要目标。

分片继续条件：`scatter时间减少 > 额外清零 + H维归约 + 分波新增开销`。S继续条件同样使用完整run，不能只比较某个scatter kernel。

若两者都没有收益，停止此候选，不通过改用CSR或大量预处理掩盖失败。

## 7. 类型、操作与库内接入契约

本轮新候选首先仅声明opA=non；CSC SpMM覆盖opB=non/trans/conj及现有合法布局。CSR转置生命周期和转置算法本轮不改。未声明操作显式请求时报错，compare排除并说明原因。

| 输入A与RHS | CSR SpMV计算 | CSC SpMM计算 | 输出 |
|---|---|---|---|
| FP16/BF16 | FP32 | FP32 | 同类型或FP32 |
| FP32 | FP64；不把旧row_tile专用后端例外推广到新候选 | FP32 | FP32 |
| FP64 | FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | FP32实虚分量 | complex64 |
| complex128 | FP64实虚分量 | FP64实虚分量 | complex128 |

不引入异类型A/RHS、任意计算精度、TLE或Tensor Core，不以精度改变冒充算法优化。

- 独立实现和显式alg/config，不修改旧kernel或默认路由；静态候选按本轮确认单独注册为实验对照；其与旧tile的增益必须在相同R/K、精度和计时边界下验证，不预先认定结构创新。
- 复用原类型解析、prepare/run、设备能力、输出检查、同算法索引回退及公共导出。
- 原始输入别名、内部重叠、out身份、strided输出和index地址宽度保留；无效地址必须在load/store处mask，不能用where假装阻止加载。
- 普通Triton，合法二次幂逻辑shape；不是手写CUDA lane布局。不默认开启平台特有cache modifier，不捏造连续性提示。
- 后端分别检查实际FP64/atomic能力及合法warp集合；CUDA/ROCm配置分别验收，未知能力不冒充已支持。
- 矩阵结构模型离线分析；run不加入额外统计扫描以选择算法。本轮不接调优器、不扩大缓存或图绑定。
- 新专用参数只属于各自算法，不能让compare把同一config套给所有名称。

## 8. 落地顺序与验证

### 阶段0：补证据并固定对照

补齐可打开的profile和原始数据。离线增加R=32/64/128的行组最大值/利用率、R=1/2/8的静态命中率，以及CSC列组等待比例。统计不进入正式运行分母之外的隐藏计划。

对CSR row_vector做B=64/128/256与合法warp的有限显式实验。确认短行是否控制成本、中长行是否长尾、CSC是否scatter占主导；不能仅凭完整事件时间就认定HBM瓶颈。

### 阶段1：低状态候选

实现A；B以独立快/慢分支实验实现保留，与旧tile做同配置对照。接入原CSR准确性文件、规格和benchmark；用旧实现同参数及不同参数两个层次对照。

### 阶段2：CSC列分段

实现D的H=1，先验证S=1/2/4，隔离固定列分段的效果。再增加H>1与受限波次；分别证明计算语义、workspace界限和收益。

### 阶段3：固定行分段

仅在中长行诊断支持该方向后实施C；先H=1控制，再H=2/4。不得先搭完整动态队列或多级计划框架。

### 阶段4：原测试交付

只在现有kernel/config/runtime、准确性/策略测试、文档及原runner接入；不新增Python测试入口。沿用marker、quick/normal、共享golden/容差和JSON。

必须覆盖：

- 六种dtype及四种half组合；真实stride、out身份、NaN预填、输入不变和别名拒绝。
- 空矩阵、空行列、连续空指针、重复项、未排序索引、矩形、尾部和抵消。
- A：全空组、R尾部、单长行污染、0/1/2/4长度。
- B：K-1/K/K+1、组内一条长行、快/慢分支，半精度宽输出无中间窄化。
- C：threshold及chunk边界、空segment、单行跨H、短行不读未初始化partial、预算不足。
- D：S/H边界、同列重复项、跨列热点、均匀目标行、N尾部、多波次、Nwave<BN、opB与布局。
- 能力不足才SKIP，编译/配置/执行失败必须FAIL，不放宽容差。

计时保持 `ms=process_cpu_ms+gpu_ms`；无CPU计划则CPU process为0。初始化、部分和、归并、转换和分波全部进run。完整采样与诊断独立；同时报告gpu_ms吞吐和prepared_run_ms，逻辑带宽不称HBM实测带宽。

真实性能首测：30矩阵non、int32，SpMM N=32；先FP16/BF16→FP32和regular FP32/FP64，再同类型half/复数及其他N、opB、布局、索引。CUDA保留cuSPARSE，DCU无支持baseline时只比较自身。继续原run_flagsparse_pytest.py，CUDA按现有sbatch脚本进入环境。

建议继续投入门槛为目标子集相对old_best的GM稳定≥1.10，并确有全体最优增益；不是收益承诺。几次重复都在噪声内、或收益被init/reduce抵销，就停止扩展。准确性未通过不推荐部署，但按既定规则保留FAIL性能记录。

## 9. Triton依据与未验证项

- [tl.load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)：mask控制真实加载，显式other=0；不依赖where保护越界。
- [tl.static_range](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html)：提供编译期展开表达，但代码变大是否有利仍须检查。
- [tl.atomic_add](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)：按目标指针元素类型进行atomic，检查后端支持；使用设备范围原子并依赖同流kernel边界完成跨program排序。
- [官方softmax教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：借鉴编译后寄存器/共享内存和occupancy检查，不照搬其算法或参数。

官方main文档不代替目标Triton 3.6与各设备的编译验证。当前只能证明设计上的索引覆盖、输出所有权与空间界限；不能宣称编译已通过、访存已合并或性能已改善。


## 10. 源码交付记录（2026-11-16）

基线为 `zyq_sparse` / `9a6d9945828ab125bcfe467f59cd96884a90cf95`。
开始时已有 `tests/benchmark_utils.py`、`tests/ci/test_runtime_policies.py` 和本优化文档目录的未提交修改，均保留；本轮不修改厂商baseline、统一runner、采样工具或原有默认绑定分支。不暂存、不提交。

| 新名称 | 实际执行 | 默认config | 状态 |
|---|---|---|---|
| `csr_row_scalar_strip` | `[R]`独立行状态；每轮每行最多一个NNZ，循环至组内全部完成；无行内数值归约 | R=64、warp=2、stage=1 | 源码接入，硬件未验证 |
| `csr_row_static_tile` | `[R,K]`；组内max≤K只加载一块，否则K步长完整循环；最终每行归约 | R=8、K=16、warp=2、stage=1 | 实验对照，硬件未验证 |
| `csc_col_partition_panel_atomic` | 每列固定S个chunk区间；直接CSC散射，H个输出分片可选；输出列波次限空间 | S=2、H=1、K=32、BN=16、warp=4、stage=1、256MiB | 源码接入，硬件未验证 |

CSR配置是**扁平字典**：`rows_per_program/num_warps/num_stages`，static额外`block_nnz`。CSC配置是`segments_per_column/output_shards/block_nnz/block_n/num_warps/num_stages/workspace_bytes`。这些新字段不跨算法继承。complex128的CSC默认BN=8；N较小时内置BN取覆盖N的较小二次幂，显式配置不裁剪。

所有新名称仅支持opA=non。六种dtype与四种half组合沿用第7节，CSC opB三种操作均通过原入口处理一次。没有新顶层函数；原查询、prepare/run、高层API及两级导出可调用。CSR新路径不进入旧ROCm row_tile精度例外，也不进入旧gather或缓存绑定。

### 实现差异、正确性与成本

- scalar-strip保存每行位置和计算精度累加值，循环控制仍有一次组内active归约；它减少的是数值归约和二维累加状态，不是消除所有同步，也不能解决单长行污染。
- static-tile快路径只消除循环控制；慢路径仍处理整行。它与既有row_tile非常接近，必须测量对照，不能仅因新名称将收益归为新算法。
- CSC长度L对应Q=ceil(L/K)个chunk。分区s的区间为`[floor(Q*s/S), floor(Q*(s+1)/S))`，覆盖互斥且完整，空分区不读取B。分片为`(column*S+s)%H`。重复坐标逐贡献atomic，complex实虚部分别累加。
- CSC H=1且输出就是计算精度时，先清零合法out/result并直接atomic；零额外数值工作空间。H>1或half输出使用计算精度缓冲`[H,M,W]`，W由预算算出且允许W<BN，最终归约直接按out stride写回。预算至少能容纳一列，否则报错；不静默减少S/H。
- CSC数值临时空间为`H*M*W*sizeof(compute_element)`，complex按完整复数元素计数。最终输出不是临时workspace；没有双缓冲、owner-map、排序或动态描述符。分波重读A及B/重新初始化和最终写回的成本计入run，不能将较小workspace理解为免费。
- CSR初始化由每行最终store完成；CSC空行由atomic目标清零保证。x/B/out使用真实stride，复数视图保留逻辑步长，不用reshape复制结果。
- `ms=process_cpu_ms+gpu_ms`；新结构计划CPU工作为0。CSR行循环、CSC初始化/scatter/归约/转换均属compute；必要输入视图物化仍在原run边界。完整计时与诊断分开，保留吞吐/逻辑带宽和FAIL性能记录。

### 测试与静态审查

原 `test_spmv_csr_accuracy.py`、`test_spmm_csc_accuracy.py` 已加入类型/操作覆盖、空维度、尾部、宽输出、重复项、strided out、原始输入不变与安全索引回退。短行候选的fallback用例由规格产生合法op；不把non算法强行参数化为trans/conj。scalar的超长行能力以小规模257-NNZ样本及既有边界覆盖，不把分段专用257-segment压力用例套给短行候选。

CSC新增S=1/2/4、H=1/2/4、W=3<BN、尾波次、复数opB=conj以及不足一列workspace报错覆盖。仅明确后端能力缺失可SKIP。原策略测试增加配置隔离、能力未知、非法参数及无结构计划的源码契约；GPU CI继续收集原marker，没有增加第二个入口。

已做15个修改Python文件的AST语法检查、`git diff --check`及旧JIT函数AST对照；`_spmv_csr_kernels.py`的18个、`spmv_csc.py`的13个、`spmm_csc.py`的14个旧JIT声明/计算体均无改写。静态支持表新增每候选12个non记录（6种输入dtype×2种index），全部标为UNVERIFIED；宽输出组合在规格和准确性用例中另行覆盖。公共算法常量、规格、静态支持表和operator registry同步。benchmark的compare由注册表枚举，runner透传进入现有脚本；无须在runner复制算法名单。另补齐CSC regular脚本原先未下传的`--config`：仅单个显式算法可用，传给prepare，实际配置与元数据写入CSV。half原生测试必须用`--type-combos`，避免旧regular脚本的历史FP16模拟分支。

**未完成的验收：** CUDA/ROCm编译、准确性、性能及profiler验证。本地没有运行这些测试；静态通过不代表Triton lowering或硬件行为已经验证。

## 11. 计算节点命令与消融顺序

以下从仓库根目录、已分配的CUDA计算节点执行。先原runner quick准确性；通过后将mode改为normal。登录节点使用`sbatch --partition=gpu_a800 --gpus=1 tools/run_cuda_tests.sbatch ...`提交同样参数，不能在登录节点直接bash运行。

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csc --phase accuracy --mode quick \
  --results-dir pytest_results_short_csc_accuracy
```

随后先用**一个真实矩阵文件**作下列性能命令的`--benchmark-input`，确认新名称有结果且无异常，再运行整个目录。regular与mixed分开。现行`--type-combos`仅接受四种half组合，因此regular用`--dtypes`，不能给它传FP32/FP64/复数三元组；现行`--perf-metrics`也仅用于显式half组合。CUDA不关闭vendor：regular沿用各脚本既有vendor路径（CSC的CUDA路径经CuPy包装），mixed使用现有直接Generic接口。本轮不更换baseline或改写regular的历史测量框架。

```bash
MATRIX_DIR="$HOME/run/matrix"
test -d "$MATRIX_DIR" || exit 1
RESULT_ROOT="$PWD/targeted_short_csc_results"

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --dtypes float32,float64,complex64,complex128 --ops non --index-dtypes int32 --timing" \
  --op-benchmark-args "spmm_csc=--op-b non --layout row --dense-cols 32" \
  --results-dir "$RESULT_ROOT/regular_non"

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 --ops non --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics" \
  --op-benchmark-args "spmm_csc=--op-b non --layout row --dense-cols 32" \
  --results-dir "$RESULT_ROOT/mixed_non"
```

DCU在已配置的计算节点环境使用相同runner参数，把`bash tools/run_cuda_tests.sbatch --repo "$PWD" --`换成`python run_flagsparse_pytest.py`；当前DTK不支持的厂商组合可在benchmark-args追加`--no-vendor`，仅关闭baseline。CUDA不要追加此参数。

专用配置只传单个显式算法；例如以下H=2对照，S固定为2：

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmm_csc.py -- \
  "$MATRIX_DIR" --alg csc_col_partition_panel_atomic \
  --config '{"segments_per_column":2,"output_shards":2}' \
  --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
  --ops non --op-b non --layout row --dense-cols 32 \
  --index-dtypes int32 --indptr-dtypes int32 --warmup 10 --iters 50 \
  --timing --perf-metrics --csv-csc "$RESULT_ROOT/csc_s2_h2.csv"
```

先固定H=1改变S=1/2/4，再固定S改变H；不同时改变K/BN。CSR scalar固定R对比旧subgroup/static，再分别改变R；static固定R/K与旧row_tile对照。中长行只对现有row_vector做block=64/128/256验证，达到第8节门槛前不实现固定行分段新候选。扩展CSC的opB、layout和N时复用原透传：`--op-b non,trans,conj --layout all --dense-cols 8,32,128`；新算法opA仍non。

返回数据按日期保存，报告每矩阵old_best/new_best、全体与目标子集GM/AM、候选最优占比及回退；H>1若不胜H=1，应判定分片无收益，而不是只与更弱旧算法比较。


### English implementation summary

Three explicit candidates are registered: `csr_row_scalar_strip`, `csr_row_static_tile`, and `csc_col_partition_panel_atomic`. All accept only opA=non in this stage; CSC SpMM accepts all existing opB modes and legal dense/output strides. Six floating/complex input types and the four half/BF16 output combinations are covered by source-level test additions. Old kernels and default selection remain intact. The static-tile candidate is experimental and must be compared with the existing row tile at identical tile sizes and arithmetic precision.

CSC partitions column chunks without an owner map. Optional output shards use bounded output-column waves; all initialization, repeated sparse reads, atomic accumulation, shard reduction and final conversion remain inside the run. Metadata records S/H, actual configuration, output buffer identity and peak numeric workspace. No hardware correctness or performance claim is made: only AST/diff checks and unchanged-old-kernel source comparisons have run locally.


旧kernel文件基线SHA256（HEAD源码文本、换行规范化为LF，用于后续审查；不是二进制编译摘要）：

```text
_spmv_csr_kernels.py  069c9b5cf2e156078dce252a6266f85e43aa787fee34f93aa7161c0751ec5abd
spmv_csc.py           887e73671cb9db56a55bc7c3c1e6a9aaa001cb01c35aecc22a6d9bbad4fac3c5
spmm_csc.py           d0653e1a774dae8287d9ed77bcacc6a819e339d0ab318ea4fc48e2d8aa84b79f
```
