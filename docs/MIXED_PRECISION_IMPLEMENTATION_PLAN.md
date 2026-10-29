# SpMV/SpMM 混合精度补齐与优化实施计划

日期：2026-10-20  
目标库：FlagSparse-main，`zyq_sparse` 分支  
状态：规划文档；本文不代表功能已实现、已编译或已通过实机验证。

## 1. 目标、术语与边界

让用户通过原公共 API、prepare/run、原准确性和性能入口，明确选择算法及合法的类型组合，并获得一致的执行行为和结果记录。

混合精度采用厂商通行定义：输入、计算或输出使用不同精度。例如 FP16 输入、FP32 计算、FP16 输出也是混合精度。区分：

- **算法内置策略**：当前算法已固定的输入、计算和输出类型关系。
- **类型组合能力**：算法能接受哪些输入类型、计算类型与输出类型组合。
- **mixed 模块**：`mixed_spmx.py` 的类型判断、专用内核及后端兼容代码；它不是混合精度概念本身。
- **q4**：现有指定组合的测试组织方式，不是量化 INT4，也不是独立算子库。

本计划以 CSR/COO/CSC 的六个 SpMV/SpMM 算子为第一交付范围。其他格式的能力也要盘点，但 BSR/BELL/SELL 的新类型扩展单独排期，不凭六个算子的结果声明全库支持。

约束：

1. 以 main 的代码风格、默认路由、生命周期、后端抽象和测试交付方式为准。
2. 原默认调用保持兼容；不擅自改变 FP32/FP64/复数既有计算精度。
3. 指定算法后不得因类型组合静默切换为另一个算法。
4. 新原生 COO/CSC 路径不借转换格式伪装原生执行。既有兼容路径保留身份与限制，不能作为新原生算法验收结果。
5. 优先扩展现有文件，不另建测试框架或 benchmark 入口。本文新建文档不授权立即改算子代码。
6. 不顺带引入跨调用缓存、全局 workspace 池、调优器或 Tensor Core 路径。
7. 当前环境只允许源码审查、AST/静态检查和 diff 检查；算子、pytest、功能测试、离线编译及性能验收均在计算节点执行。
8. 不暂存、提交、重置现有修改。

厂商组合与版本限制参考：[厂商参考记录](MIXED_PRECISION_VENDOR_REFERENCE.md)。文档中的厂商支持和本库实机支持必须分开。

## 2. 当前状态与需要解决的问题

现有普通注册算法已包含固定混合策略，例如 FP16 输入、FP32 累加、FP16 输出。`mixed_spmx.py` 另行提供宽输出、整数及实数乘复数等指定组合；公共入口通过类型判断进入专用实现，并非从普通算法注册表统一选择。

目前因此存在以下断点：

- 单一 `supported_dtypes` 无法表达 A、x/B、计算和输出四种类型。
- 显式算法与 `out_dtype` 不统一；默认入口能完成的组合不代表所有算法支持。
- prepared 与直接调用的类型、输出校验和计算边界可能不同。
- q4 准确性和性能各有组合清单，不是统一的能力声明。
- COO/CSC SpMV 的显式 `--alg` 分支结束后不执行旧 q4 追加；CSR SpMM 默认 `test_spmm.py` 与扫描入口 `test_spmm_csr.py` 也不等价。
- 直接 cuSPARSE C API baseline 已有辅助模块，但直接 HIP C API 的对等链路未接通。
- 注册声明、实际执行和硬件验证状态没有形成完整的一一对应表。

## 3. 目标类型、操作与布局

### 3.1 类型组合分批

以下为目标，不是现有支持承诺。每个算法只有在实现完成后才增加能力声明。

| 批次 | A | x/B | 计算/部分和 | 输出 | 目标 |
| --- | --- | --- | --- | --- | --- |
| P0 | FP16 | FP16 | FP32 | FP16 | 六个算子，补齐目标算法 |
| P0 | BF16 | BF16 | FP32 | BF16 | 六个算子，补齐缺口 |
| P1 | FP16 | FP16 | FP32 | FP32 | 六个算子 |
| P1 | BF16 | BF16 | FP32 | FP32 | 六个算子 |
| P2 | INT8 | INT8 | INT32 | INT32 | 六个算子，单独验收整数策略 |
| P2 | INT8 | INT8 | FP32 | FP32 | 六个算子，明确先转 FP32 再乘加 |
| P3 | FP32 | FP64 | FP64 | FP64 | 优先三个 SpMV |
| P3 | FP32 | complex64 | FP32 实虚分量 | complex64 | 优先三个 SpMV |
| P3 | FP64 | complex128 | FP64 实虚分量 | complex128 | 优先三个 SpMV |
| P3 | complex64 | complex128 | FP64 实虚分量 | complex128 | 优先三个 SpMV |

P3 的 SpMM 扩展另作支持评估，不由 SpMV 实现自动宣布支持。FP8、INT4、复数半精度、scale/zero-point 量化不属于本轮范围。

### 3.2 操作与布局

- SpMV：`opA=non/trans/conj`。
- SpMM：目标覆盖 `opA × opB` 的九种组合：NN、NT、NH、TN、TT、TH、HN、HT、HH。
- `conj` 表示共轭转置；不能只取共轭。实数 T/H 数值等价，接口参数仍分别覆盖。
- SpMM dense 输入支持原 row/col 布局；合法 strided view 和 out 沿用原契约，不能未经验证扩张任意 stride 支持。
- B 的布局和 `opB` 分别记录，不把列主序视为已经转置。
- 缺少 `op_b` 的入口采用向后兼容的关键字扩展；不得移动旧位置参数。
- 转置处理复用各格式现有生命周期。CSC 的原生列计算不为统一接口改用 CSR；需要逐次发生的处理不得藏到测试外。
- 后端、索引、安全回退仍受现有能力约束；不因类型扩展就声明 TLE 或其他专用实现支持全部组合。

## 4. 分步实施

### 步骤 0：建立源码与测试基线

工作：

1. 记录分支、commit、工作区差异；只读取，不改动已有修改。
2. 从六个算子注册表枚举全部名称，包括旧算法、新融合算法与专用路径。
3. 逐名称追踪高层入口、prepare、run、最终 kernel/兼容函数。
4. 记录实际类型转换位置、累加器、atomic 输出、部分和和最终输出类型。
5. 盘点 q4、普通 accuracy、benchmark、runner 各自实际覆盖内容。

产物：逐算法支持表，至少包含：

`算子/格式/算法/输入类型/计算类型/输出类型/opA/opB/layout/后端/索引限制/执行函数/测试位置/验证状态`。

状态区分：仅声明、源码已实现、编译通过、准确性通过、性能已测。不把注册表等同于实机支持。

完成条件：每个注册名称都有实际去向；不再使用“部分算法支持”作为能力记录。

### 步骤 1：补齐 P0 固定混合策略

工作：

1. 检查 FP16/BF16 在各算法入口、加载、运算和输出中的一致性。
2. 加载后提升到 FP32；乘法也应在正确类型执行，不能先发生低精度乘法再提升。
3. atomic 路径使用 FP32 工作输出，最终转换一次；不因输出 FP16 就改用低精度 atomic。
4. split/adaptive 的部分和、层级归约和跨批累加保持 FP32。
5. 保留算法结构、默认参数和旧高精度路径；TLE 等专用算法逐一核实编译/硬件条件。
6. 在原 accuracy 文件补对应 dtype/op 测试。

完成条件：能力表中新增的 P0 组合均有真实计算路径及测试；不可用的专用算法明确标注原因。

### 步骤 2：统一类型解析，不执行计算

在现有公共 helper/mixed 模块中复用并改造类型规则，避免新增平行框架。

类型解析的逻辑输入：

`A.dtype、x/B.dtype、out_dtype、out.dtype、算子类别`。

逻辑输出：

`a_dtype、rhs_dtype、compute_dtype、output_dtype、complex_component_dtype`。

规则：

1. `out_dtype` 与 `out.dtype` 不一致时报错。
2. 未显式要求输出时保留原默认值，包括已有 INT8 默认输出。
3. 解析器只检查数学类型组合并返回信息，不分配张量、不发起 kernel、不选择替代算法。
4. 不把类型组合合法性与某算法的支持能力混成一层。
5. 第一阶段不暴露任意 `compute_dtype` 用户参数；由支持组合确定计算精度，避免无依据改变数值策略。

完成条件：高层调用与 prepare/run 得到相同类型解释；冲突在一致的位置被报告。

### 步骤 3：扩展算法规格与能力查询

工作：

1. 将单一 dtype 清单扩展为可表达四种类型的规则或组合集合。
2. 支持检查同时考虑格式、算法、opA/opB、布局、后端和索引限制。
3. 算法名称规范化、查询、compare 枚举和实际执行共同使用该规则。
4. 保留旧查询函数调用方式；按需新增可选过滤参数，不破坏两级公开导出。
5. 显式不支持的组合报出算法名及具体原因，不静默路由到 mixed。
6. 普通默认调用保持原分派，暂不开展“最优算法自动选择”。

示例目标：

`csr_row_panel + FP16/FP16/FP32/FP32 + non/trans + row` 能被独立查询；未实现前必须返回不可用。

完成条件：查询支持与实际执行条件一致；相同组合不会被 compare 错误枚举。

### 步骤 4：接通 P1 宽输出与原算法

工作：

1. 高层入口完成类型解析后，将类型信息交给所选算法。
2. 输出根据目标 dtype 分配；接收用户 out 时校验形状、类型、设备、重叠和 stride。
3. kernel 的输入存储类型、计算类型与输出类型分离。
4. 直接算法在最终 store 写入目标类型；split/atomic 路径在全部计算结束后再做输出处理。
5. 工作空间上限按实际计算类型重新计算；包括复数分量及归约缓冲。
6. prepared 不错误固定依赖 RHS 的类型/布局。可绑定的信息按原契约绑定，运行时变化的信息在 run 检查；不能复用不同组合的专用计划。
7. 索引回退保留算法、配置、op、类型组合，不因回退改变计算精度。
8. 保留旧 mixed 默认路径；为其元数据记录明确 implementation/route，避免冒充新算法。

目标调用示意：

```python
flagsparse_spmm_csr(
    values_fp16, indices, indptr, B_fp16, shape,
    alg="csr_row_panel", out_dtype=torch.float32,
)
```

该调用必须执行 row panel 方法，而非换成 `spmm_csr_mixed` 的逐行内核。

完成条件：高层与 prepared/run 的输出、精度及算法身份一致；禁止“先 FP16 输出，再转 FP32”的伪宽输出实现。

### 步骤 5：补齐操作、布局及兼容路径

工作：

1. 对 P0/P1 逐格式覆盖 SpMV 三种 op、SpMM 九种 op 组合。
2. B 转置优先使用现有 stride 机制；复数共轭若需物化或内核符号处理，显式记录其执行边界。
3. A 的转置处理沿用当前格式实现，不为类型统一而改写全库转置算法。
4. 修正使用原形状校验转置输出、使用错误行列统计等接口问题。
5. 对旧混合兼容分支记录原生/委托/转换身份；保持兼容但不混入原生算法成绩。

完成条件：每个声明组合形状正确，非对称复数数据能区分 T/H；输入不被修改。

### 步骤 6：统一原测试入口，承接 q4 覆盖

准确性：

1. 在六个原 `tests/pytest/test_*_accuracy.py` 内加入类型组合参数化。
2. 使用原 marker、QUICK_MODE、共享 golden、容差和结果记录。
3. quick 保留每算法声明组合的代表用例；normal 扩展索引、布局、边界，避免无意义全笛卡尔积。
4. q4 清单逐项映射到新用例后再消除重复；保留兼容入口，不立即删除历史测试。

性能：

1. 在原 benchmark 中增加明确的输入/输出类型组合选择，不只用一个 `--dtype` 表达全部信息。
2. 建议关键字概念为 RHS dtype、output dtype 或显式组合列表；最终 CLI 命名须与现有解析风格对齐，不能提前声称已有这些开关。
3. `--alg compare` 对指定类型组合枚举支持算法。普通路径与 q4 路径不再因提前 return 导致覆盖缺失。
4. 显式算法组合未支持应输出原因；精度 FAIL 保留有效耗时与加速比，不中断后续矩阵。
5. 接口、编译和执行异常输出上下文与 traceback；CSV 逐条保存。
6. 总 runner 沿用原参数透传；CSR SpMM 默认脚本与显式算法脚本都应获得一致的组合能力。
7. 避免普通测试之后无条件追加一套与用户 dtype 选择无关的 q4 行；统一后使用明确组合清单去重。

完成条件：用原 runner 就能测试算法×类型组合；无需用户另找 q4 才能测宽输出。

### 步骤 7：直接厂商 C API baseline 与计时审计

工作：

1. 复用 `tests/cusparse_generic_baseline.py`，扩展所需组合和参数传递。
2. 增加直接 HIP C API baseline 适配，明确调用 hipSPARSE 还是 rocSPARSE；不把 hip-python 或 CuPy 路径标为此项完成。
3. 读取实际库版本、设备、算法及 API 返回状态，结合版本规则判断可用性。
4. baseline 使用同设备、同格式、同 op、同输入/计算/输出类型；不能先把 FP16 全部转为 FP32 再称 native FP16 baseline。
5. 支持表不明确的交叉组合在计算节点探测并记录；接口故障与“不支持”分别报告，不吞掉所有异常。
6. 厂商不支持时记录 N/A 原因，仍测试 FlagSparse。

计时：

- 沿用 main 各入口约定，不把完整首次调用与 prepared 稳态结果混在一起排名。
- 统一算法对比的准备/运行边界；逐次必要转换、清零、归约、输出处理必须在运行时间内。
- 分段诊断单独执行，不替换完整采样值；已有总时间公式与采样过滤口径保持一致。
- 审计 q4 当前 call/prepared 差异和采样辅助函数，接入公共统计机制后记录口径版本。
- 不将设备等待、配置字典组装作为 CPU 算法处理时间。
- 记录实际输出/计算类型及计划复用情况；厂商描述符准备与本库 prepare 范围分别说明。

完成条件：同条件数据能公平比较；不可用 baseline 有可复核原因。

### 步骤 8：P2 整数与 P3 异类型输入

整数：

1. INT8→INT32 使用相应整数乘加与部分和，不能套浮点缓冲。
2. INT8→FP32 明确先提升到 FP32 再乘加，保持现有 mixed 策略。
3. 定义溢出契约并设计长行/极值用例；不得用小随机数通过替代边界验证。
4. 检查后端整数 atomic/归约能力；不可用时明确限制，不偷偷改精度。

异类型输入：

1. A 与 RHS 分别传递存储类型，移除仅适用于同类型的假设。
2. 实数×复数分别累加实虚部；conj 只作用于相应被指定操作的输入。
3. FP32×FP64、complex64×complex128 使用目标计算精度加载和累加。
4. 第一批只做 SpMV；SpMM 评估通过后再扩展注册与测试。

完成条件：各组合有独立策略、支持规则和正确性测试，不是最终强制 cast。

### 步骤 9：性能优化，保持数值契约

按消融逐项推进：

1. **融合加载转换**：比较完整 FP32 副本与寄存器提升；减少临时空间和数据流量。
2. **输出直接写回**：直接算法写入合法 out，避免额外分配/复制；不提前降低部分和精度。
3. **类型感知配置**：按计算类型、输出宽度和复数分量选择 tile、warp、stage；使用现有后端能力接口。
4. **复用稀疏加载**：SpMM 面板复用 A/索引；评估寄存器占用与并行度，不仅看单 kernel 时延。
5. **atomic 合并与无 atomic 归约**：复用既有 segmented/stream/split 思想；完整运行收益包含计划成本。
6. **受限 workspace**：按真实元素宽度控制峰值，评估输出列/segment 分批成本。

暂不纳入 Tensor Core、结构化稀疏、跨调用新缓存或自动调优；需要时独立立项。

完成条件：每个优化有未优化对照、相同输入和数值策略，报告完整调用收益及回退矩阵。

## 5. 验证与交付要求

### 策略与接口

- 类型冲突、算法不支持、未知后端、prepared 不匹配、配置隔离。
- 默认返回值、out_dtype、op_b、out、两级导出及 C API 旧内核 ABI 无回归。
- 类型转换不影响索引安全回退的算法身份。

### 数值与边界

- 空矩阵、零维度、空行列、矩形、重复项、无序索引。
- tile 尾部、极短/超长行列、多级归约、小工作空间分批。
- FP16/BF16 舍入与动态范围、正负抵消；不放宽共享容差掩盖失败。
- out 预填非零/NaN，验证完整覆盖、对象身份、输入不变。
- 九种 SpMM 操作与 row/col 布局用非对称输入验证；复数实虚部均非零。

### 性能

- 固定随机种子与原矩阵数据策略，不为减少 FAIL 擅自缩放输入。
- 所有 30 个矩阵的有效时间都参与性能记录，精度 FAIL 单独标记。
- 固定 dtype 四元组、op、布局、N、索引后统计单算法、旧最佳及全算法最佳。
- 算术/几何平均分别标明；缺失、异常、N/A 不当成零，也不悄悄剔除后声称满覆盖。
- CUDA/ROCm 各自记录设备和软件版本；其他平台独立记录验证范围。

沿用现有计算节点入口：

```bash
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase both --mode quick \
  --benchmark-input ../../matrix \
  --results-dir pytest_results_mixed
```

此命令当前只覆盖现有测试内容。步骤 6 完成后，新组合才自动接入；新增组合 CLI 的准确命令届时同步文档，不能用尚不存在的开关作为当前交付。

## 6. 里程碑

| 里程碑 | 包含步骤 | 必须交付 |
| --- | --- | --- |
| M0 | 0 | 逐算法真实支持表及已有测试映射 |
| M1 | 1～3 | FP16/BF16 固定策略、统一类型解析和规格查询 |
| M2 | 4～7 | 宽输出、操作/布局、原测试接通、直接 baseline 与计时审计 |
| M3 | 8 | 整数与异类型输入的独立验收 |
| M4 | 9 | 消融与完整运行性能报告 |

每个里程碑分别报告“源码接通”“计算节点编译/准确性”“性能实测”。源码完成不能替代硬件验收；失败和未验证组合不得标为支持已确认。


## 7. 首轮实施进度（2026-10-21，尚未完成）

以下仅记录当前源码接入范围，不代表 CUDA/ROCm 已编译或通过准确性测试。
本轮目标仍为六个算子的全部非 TLE 算法；未完成的旧算法不能从目标清单中删除。

统一类型解析 `mixed_spmx.resolve_registered_types` 只校验类型并返回类型四元组，
不会启动 mixed 专用 kernel。显式算法继续走自身实现。当前接入的组合均为
A/RHS 同为 FP16 或 BF16、FP32 计算、同类型或 FP32 输出；异类型输入不在本轮范围。

| 算子 | 已接通上述四种组合的注册路径 | 操作 | 尚待补齐 |
| --- | --- | --- | --- |
| SpMV CSR | legacy_rowpar、legacy_segbin、legacy_bucket_vector、row_tile、row_vector、row_split_reduce、row_adaptive_split、row_subgroup_stream | non/trans/conj | 特殊后端兼容路径、缓存和回退复核；实机验证 |
| SpMV COO | coo_rowrun、coo_atomic、coo_segmented_atomic、coo_rowrun_subgroup、coo_segmented_stream_atomic | non/trans/conj | 旧 prepared 兼容对象的宽输出、实机验证 |
| SpMV CSC | csc_col_subgroup、csc_col_vector、csc_col_split_reduce；csc_col_tile_atomic | 前三者 trans/conj；atomic 为 non | spmv_csc_base；默认委托路径单独标识 |
| SpMM CSR | csr_row_tile、csr_row_kparallel、csr_row_panel、csr_split_nnz_reduce、csr_adaptive_tile_split | opA/opB 各 non/trans/conj | csr_base、csr_base_accuracy、spmm_csr_alg1、spmm_csr_alg2、spmm_csr_alg2_accuracy、spmm_csr_alg2_accuracy_hp |
| SpMM COO | coo_rowrun、coo_atomic、spmm_coo_alg1、coo_segmented_panel_atomic、coo_rowrun_panel、coo_segmented_stream_panel_atomic | opA/opB 各 non/trans/conj | 后端兼容实现身份及完整边界复核；实机验证 |
| SpMM CSC | csc_col_panel、csc_col_kparallel、csc_col_split_reduce；csc_col_tile_panel_atomic | 前三者 opA=trans/conj，atomic 的 opA=non；opB 均三种 | spmm_csc_base |

SpMM 保留现有 dense stride/layout 支持；新增测试覆盖转置视图和 strided out，
两种布局的完整参数化及 normal 边界仍需补齐。索引继续遵循原算法限制，
例如 legacy_bucket_vector 的执行列索引必须为 int32；本次不扩大索引支持范围。

源码实现要点：

- CSR SpMV 三条 legacy 路径向现有 kernel 传递 FP32 out；segbin 的 atomic 缓冲保持 FP32。
- COO SpMM 三条 legacy 路径新增内部目标输出参数，计算结果在最终输出时才转换，
  不修改 prepared 上代表原始输入类型的字段。原 canonical/coalesce 的 prepare 生命周期保留。
- 新 CSR split/adaptive 的部分和与分批缓冲对 FP16/BF16 均保持 FP32。
- 新 COO/CSC 路径在 run 内提升半精度输入，最终输出转换、初始化和归约留在 run 中。
- CSR 显式算法的 opB 在 run 中处理为视图，复数共轭由执行路径物化；旧默认路径不重写。
- 六个算法查询入口增加 RHS/输出类型筛选；尚未实现的宽输出不在查询结果中冒充支持。
- 六个原 accuracy 文件已增加类型组合、高层/prepare-run 一致性、op 路由、NaN out 和类型冲突用例；
  沿用原 marker。策略用例添加到原 test_runtime_policies.py。测试尚未运行。
- cuSPARSE ctypes helper 已改为按需绑定符号，缺少 SELL 等接口不再阻断无关的乘法接口；
  直接 hipSPARSE C API 适配尚未实现，原 hip-python 未删除。

接下来的实施顺序：

1. 补齐表中 CSR SpMM/CSC legacy 路径，审计各自 FP32 累加及宽输出，不切换算法。
2. 完成全部 prepared/config/op 冲突和原始存储重叠检查，补齐 normal、空维度及小工作空间用例。
3. 原 benchmark 接入 --type-combos、--op-b、类型组合枚举和 CSV 字段，保持 q4 兼容规则；
   这些新 benchmark 参数当前还不能作为已交付命令使用。
4. 完成直接 hipSPARSE、资源释放/异常分类/实际流、版本记录及 baseline 独立正确性检查。
5. 联动 runner、公开支持矩阵和文档；完成静态审查，再交计算节点 quick/normal 验收。

当前未执行任何算子、pytest、功能导入或离线编译；仅进行 AST、源码与 diff 检查。
未暂存或提交。性能、准确性和支持组合的实机状态全部为未验证。


### 后续实施批次：旧算法入口补齐（2026-10-21）

本节更新第 7 节的进度记录，仍不代表实机验证通过。

- CSR SpMM 的六个 legacy 名称已增加 FP16/BF16 注册及宽输出执行路径。
  run 内为原算法物化 FP32 输入，继续调用所选算法自己的分桶、分段与归约实现。
  半精度 accuracy/HP 组合固定 FP32 算术；已有其他类型的 accuracy/HP 策略保持不变。
- CSC SpMV/SpMM base 已增加半精度 FP32 运行缓冲及最终输出转换。
  显式 base 半精度不进入旧 mixed 模拟路径；SpMV 的新增半精度 prepared 不建立 CUDA CSR delegate。
  旧未指定算法的默认混合精度分派仍保留。
- COO SpMV 旧 prepared 可接收半精度输入及 FP32 宽输出，转换和写回纳入原运行计时。
- CSC base 索引重试仅捕获明确索引兼容错误，不把普通执行错误作为 int64 回退触发条件。
- 原 accuracy 文件增加上述 legacy 算法、SpMM row/col 布局和 COO 旧 prepared 用例；
  JSON 参数中显式记录 A/RHS/计算/输出类型与 opA/opB。

静态结果：19 个改动 Python 文件 AST 通过，git diff --check 通过，原 Triton JIT 函数签名未变。
没有执行算子、pytest、功能导入或离线编译，没有暂存或提交。

仍待完成：

1. 全部兼容路径、原始输入存储重叠、分段诊断范围和 normal 边界审计；
   当前新增用例不等于完整边界验收。
2. 六个 benchmark 的 --type-combos、--op-b、比较枚举、CSV 和 q4 兼容分支，以及 runner 联动。
3. 直接 hipSPARSE C API baseline、资源异常清理、版本/能力记录和厂商独立正确性检查。
4. 最终支持矩阵、计算节点命令及 CUDA/ROCm 编译、准确性和性能验收。

在第 2、3 项完成前，不宣称混合精度性能测试链路已经交付。

### 本轮源码交付更新（2026-10-22）

本节替代上述历史进度中的待办状态；CUDA/ROCm 实机验收仍未执行。

- 六个原 benchmark 已接入 `--type-combos`，SpMM 同时接入 `--op-b`。
  四种组合经原注册 prepare/run 执行，不调用 q4 或 mixed 专用计算 kernel。
  `compare/all` 根据类型、操作、布局及后端枚举；CSC 算法仍保持各自的 opA 范围。
- 原 runner 在 CSR SpMM 收到显式算法或类型组合参数时选择 `test_spmm_csr.py`；
  默认交付仍使用原脚本。显式算法或类型组合不追加 q4，旧默认调用保留追加行为。
- 旧 CSR SpMM、CSC base 和 COO prepared 宽输出已补齐；原始输入引用参与 out 别名检查。
  FP32 输出不能先经过半精度输出缓冲。必要转换与最终写回已纳入运行及分段诊断。
- Generic helper 增加显式 ROCm 后端，直接调用 libhipsparse，保留原 CUDA/q4 调用和 hip-python。
  使用原格式、原类型、实际流和独立描述符/workspace；构造失败清理已创建资源。
  缺库、缺符号或明确 NOT_SUPPORTED 记录 N/A；其他接口/执行错误保留 traceback。
  厂商 PASS/FAIL 与有效耗时独立记录，不因厂商不可用停止 FlagSparse 测试。
- 六个原 accuracy 文件扩展四种组合、操作、SpMM 布局/opB、strided out、原始输入别名、
  stream carry 和 split 小工作空间宽输出用例；normal 扩展索引组合。
  后端能力缺失使用专用异常表示，不能将一般 NotImplementedError、编译或执行错误一律 SKIP。
- CSV 逐行写入类型四元组、操作、算法、元数据及厂商原因；精度 FAIL 保留有效速度。
  显式组合统计沿用现有采样工具，`ms=process_cpu_ms+gpu_ms`，诊断另行执行。
  prepare 不在采样内，结果标为 `prepared_run`，不是首次调用端到端耗时。

#### 计算节点命令

以下从 FlagSparse-main 根目录执行。当前机器没有运行这些命令。

```bash
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase accuracy --mode quick --results-dir pytest_results_mixed_quick

python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase accuracy --mode normal --results-dir pytest_results_mixed_normal
```

这两个命令沿用原 marker，包含原准确性用例和新增组合用例，不只执行 mixed。
性能扫描在 Linux 计算节点使用：

```bash
COMBOS='float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32'
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase performance --mode normal --benchmark-input ../../matrix \
  --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--type-combos $COMBOS --alg compare --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing" \
  --op-benchmark-args "spmm_csr=--op-b all --layout all --dense-cols 1,8,16,32,64,128,256" \
  --op-benchmark-args "spmm_coo=--op-b all --layout all --dense-cols 1,8,16,32,64,128,256" \
  --op-benchmark-args "spmm_csc=--op-b all --layout all --dense-cols 1,8,16,32,64,128,256" \
  --results-dir pytest_results_mixed_perf
```

不要再同时添加 `--dtype/--dtypes`；显式组合和显式 dtype 会报冲突。
可以先将矩阵目录换成单个文件、减少 N 和 iters 来检查计算节点链路。
COO 两种索引类型分别代表 row/col；厂商 COO 不支持混合索引时仅 baseline 为 N/A。

直接检查一个原脚本的调用示例：

```bash
python tests/test_spmm_csr.py --synthetic --alg compare \
  --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
  --ops all --op-b all --layout all --dense-cols 7,33 --timing \
  --csv-csr mixed_spmm_csr.csv
```

性能分析应读取各算子的逐行 CSV，按完整类型四元组、opA/opB、布局、N 和索引分组。
runner 原默认交付汇总仍按默认交付投影，不代表显式扫描的全部组合或最佳算法结果。
不得将 exit code 0、注册声明或本地 AST 通过视为准确性通过。

#### 验证状态与后续

本轮仅完成源码联动和静态核查；支持矩阵新增组合标为 UNVERIFIED。
需要 CUDA、ROCm 分别完成编译、quick/normal 准确性及 30 矩阵性能验收，
尤其检查 BF16、直接厂商接口、宽输出抵消样本、stream 和分层归约。
保留既有容差和 FAIL 记录，不承诺所有厂商版本支持全部组合。
INT8、异类型输入、TLE、其他格式及性能消融仍属于后续阶段。

### 实测问题修复与逐算子确认（2026-10-29）

2026-10-27 的 quick/normal 性能结果只覆盖首个矩阵，不能作为全矩阵性能验收。
本次修复不改变默认算法、容差、prepare 边界或低精度计算策略：

- 显式类型组合 baseline 在现有 Generic helper 的独立子进程运行，传输 CPU 数据，
  不共享 GPU 指针。进程启动、数据传输及描述符准备不计入事件耗时。
  相同矩阵/类型/op/布局/索引/输出 stride 的厂商结果在算法间复用；不复用执行计划。
- 记录厂商退出码、超时及独立 stdout/stderr。`--vendor-timeout` 默认 180 秒。
  普通失败、原生 abort 或段错误作为 vendor ERROR 返回，默认继续算法扫描；
  `--fail-fast` 仍明确终止。库/组合不支持仍为 SKIP，不伪造厂商成绩。
- CSV 旁的 `.vendor_logs/` 保存 worker 日志和调用厂商前的 FlagSparse JSON 快照。
  快照标为 PENDING，不是额外的完成结果行；用于异常中断后的恢复和定位。
- CSV 增加原值最大绝对值及原值、转换后输入、RHS、参考和输出的非有限值计数。
  非有限结果保持 FAIL 和有效时间，不缩放矩阵、不扩大容差；误差不可定义时留空。
- COO 实数 segmented scan 改用段首标志的结合式归约，避免无序重复行错误合并。
  CSC 复数乘积 helper 使用互斥编译分支及一致返回形状。
- CSR compare 根据声明过滤非法列索引组合；半精度 pytest 同步能力筛选。
  独立 split 不再要求无关的 adaptive 阈值顺序，adaptive 仍检查此约束。
  CSC fallback 测试注入明确索引不支持错误，保留普通运行错误不得回退的测试。

进程隔离不保证能恢复驱动级设备重置；若设备整体失效，应停止该节点任务并恢复设备。
本次只做 AST/diff 和源码审查，没有执行算子、pytest、功能导入或离线编译。
厂商的具体类型支持及已出现的 VMFault 根因仍需在节点通过独立日志确认。

先在计算节点只确认 SpMV CSR，使用有限值合成矩阵：

```bash
COMBOS='float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32'
python tests/test_spmv_csr.py --synthetic --alg compare \
  --type-combos "$COMBOS" --ops non --index-dtypes int32 \
  --warmup 1 --iters 3 --timing --csv-csr spmv_csr_mixed_check.csv
```

先检查每算法的 status、vendor_status/vendor_reason、非有限值计数与 worker 日志。
厂商 SKIP/ERROR 不应把 FlagSparse PASS 改成 FAIL/ERROR。确认 non 后再改 `--ops all`，
随后扩大合法索引范围和真实矩阵。准确性回归仍使用原 runner：

```bash
python run_flagsparse_pytest.py --ops spmv_csr --phase accuracy --mode quick \
  --results-dir pytest_results_spmv_csr_mixed_check
```

通过后再依次确认其他五个算子，不立即重复完整六算子大扫描。
