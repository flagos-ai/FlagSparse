# SpMV/SpMM 混合精度补齐与优化实施计划

## 2026-11-03：四组合优化及 NNZ 均衡源码交付

优化前基线：`zyq_sparse` / `c76e20dd7857d0950f4d856d492d74c5e11f12ee`。
本节记录源码实现；CUDA/DCU 编译、准确性与性能均待计算节点验证。
不以此前 small synthetic 数据宣称 30 矩阵提速。

- 新增 `csr_nnz_tile_reduce`（SpMV）及 `csr_nnz_panel_reduce`（SpMM），
  仅声明 FP16/BF16 同类型输入、FP32 计算、同类型或 FP32 输出。
  原 API、prepare/run 与 compare 由注册表接入，不切换默认算法。
- 原生 CSR 固定 NNZ tile，GPU upper-bound 定位行；完整行直接写输出，
  跨 tile 行输出独占边界记录，经 GPU 前缀压紧、分层归约和顺序跨批归并。
  不使用浮点 atomic，不建立跨调用 owner-map/cache。
- workspace 默认 256 MiB，包含 FP32 行组累加器及部分和/归约同时存活缓冲；
  按面板、行组、tile 分批。`workspace_peak_bytes` 是保守数值空间上界，
  描述符另计估计上界；不包含最终用户输出、输入及 PyTorch allocator 保留空间。
  计数读回和设备等待在完整 run 内，CPU 算法处理时间不额外计入这些等待。
- SpMV 显式配置使用平坦字段 `block_nnz/num_warps/num_stages/reduce_block_size/workspace_bytes`。
  SpMM 复用 `block_k` 表示 NNZ tile 长度、`block_n` 表示输出面板宽度，其他字段沿用现有配置。
  两者默认分别为 128 和 32×16、4 warps、1 stage、256 路归约。
- CSC 直接路径改为加载时 FP32 转换；归约可直接写合法 out，non 散射仍使用 FP32 缓冲。
  CSC split 保留原物化转换及 FP32 缓冲，避免改变既有归约策略。
- COO SpMV 新算法去除半精度输入全量 FP32 副本；COO SpMM 的 compute-dtype out
  可直接充当累加输出，保留 materialize/load 显式消融。
- 原四组合宽输出测试增加新注册名称，增加空行、跨 tile、多层归约、512 字节
  workspace、面板尾部、strided out、重复调用测试。原离线编译检查增加新 JIT，
  本地不执行该检查。默认 runner 和公开函数名不变，无新增 Python 文件。

### 计算节点验收命令

在库根目录执行；CUDA/DCU 使用相同命令。先准确性，再真实矩阵。

```bash
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase accuracy --mode quick --results-dir mixed_nnz_accuracy_quick

python run_flagsparse_pytest.py \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase accuracy --mode normal --results-dir mixed_nnz_accuracy_normal

COMBOS=float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32
MATRIX_DIR=../../matrix
mkdir -p mixed_nnz_results
for kind in spmv spmm; do
  for fmt in csr coo csc; do
    extra=()
    if [ "$kind" = spmm ]; then extra=(--op-b non --layout row); fi
    python "tests/test_${kind}_${fmt}.py" "$MATRIX_DIR" \
      --type-combos "$COMBOS" --alg compare --ops non \
      --index-dtypes int32 --indptr-dtypes int32 \
      "${extra[@]}" --no-vendor --warmup 10 --iters 50 --timing \
      "--csv-${fmt}" "mixed_nnz_results/${kind}_${fmt}.csv"
  done
done
```

SpMM 未指定 dense-cols 时为 N=32。厂商验收在支持组合的 CUDA/新版 HIP 环境
删除 `--no-vendor`，记录 vendor 状态/版本；旧 DTK 不可用不阻断 FlagSparse。
全覆盖将 `--ops non` 换为 `--ops all`；SpMM extra 换为
`(--op-b non,trans,conj --layout all --dense-cols 1,8,16,32,64,128,256)`，
索引改成 `int32,int64`。建议分批运行并保存独立 CSV。

仅测试新算法及强制分批的示例：

```bash
python tests/test_spmv_csr.py --synthetic --type-combos "$COMBOS" \
  --alg csr_nnz_tile_reduce --ops all \
  --config '{"block_nnz":8,"reduce_block_size":4,"workspace_bytes":512}' \
  --no-vendor --warmup 1 --iters 3 --timing --csv-csr mixed_nnz_results/spmv_smoke.csv

python tests/test_spmm_csr.py --synthetic --type-combos "$COMBOS" \
  --alg csr_nnz_panel_reduce --ops all --op-b non,trans,conj --layout all \
  --config '{"block_k":8,"block_n":4,"reduce_block_size":4,"workspace_bytes":512}' \
  --no-vendor --warmup 1 --iters 3 --timing --csv-csr mixed_nnz_results/spmm_smoke.csv
```

消融通过相同显式算法分别传 `--config`，不能与 compare 共用配置：
CSR panel 使用 `tile_n=16/32/64`、`panel_accumulators=1/2/4`；
COO stream panel 使用 `conversion_mode=load/materialize`、
`blocks_per_program=1/2/4/8`、`local_reduce=segment/none`，每次只改变一项。
每次结果记录四元组、op、布局、N、索引、算法及配置；精度 FAIL 保留有效耗时。
性能统计分别比较同算法优化前后与旧/新最佳，不把无 baseline 的自身改进称为厂商加速比。


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

## Mixed 第二轮：panel 成块加载与 NNZ fixed 计划（2026-11-06）

基线：`5f35de149d7fce4f3f6d6817ea2a23fc20850d25`，分支 `zyq_sparse`。
开始时已有的 `tools/run_cuda_tests.sbatch`、`docs/CUDA_SBATCH.md` 修改保留。
优化前数据：工作目录 `test_mixed_data/11.06data/pytest_results_mixed_all_vendor`；
prepared_run、median ±10% mean，三个SpMV各30矩阵，SpMM CSR仅6矩阵，不能视为六算子完整基线。

### 新配置及边界

| 算法 | 配置 | 默认 | 实现 |
|---|---|---|---|
| csr_row_panel | panel_load_nnz=1/4/8 | 1 | `[R,K]`成块加载value/index，逐位置广播到面板，不保留三维累加器 |
| csr_nnz_tile_reduce / csr_nnz_panel_reduce | plan_mode=compact/fixed | compact | fixed首尾槽位、局部行定位和固定容量归并树 |
| row_subgroup_stream | 原lane、行块数配置 | 不变 | 已直接半精度加载、FP32累加和最终out写回，本轮增加配置回归 |
| COO stream | 原blocks_per_program/local_reduce；SpMM另有conversion_mode | 不变 | 已支持合法FP32 out直接atomic，本轮增加NaN/strided out身份回归 |

四种组合：FP16/FP16、BF16/BF16输入，FP32计算，输出同输入或FP32。不新增API/算法名、不切默认。
成块加载不保证机器码访存向量化，需实机检查加载事务、寄存器和spill。
fixed每次run重建，当前tile批次行界在最多四个输出面板间复用；单行tile广播owner，多行tile局部二分。
固定首尾槽保留有序owner，未用数值槽写零。跨界记录按块分段归约，完整行写FP32累加器，跨块首尾递归归并。
不使用浮点atomic、动态边界压紧或有效计数读回。fixed的reduce_block_size为4至256的二次幂；2不能保证容量收缩。
数值workspace默认256MiB，包含行波次/面板累加和归并双缓冲；描述符单独统计。小空间分批，最终才转半精度输出。
必要的行波次指针读回仍在run中。boundary_slot_capacity为峰值容量，不冒充有效记录数。
完整计时保持ms=process_cpu_ms+gpu_ms；诊断另测。初始化、计划、归并和转换均在原run边界内。

### 原测试入口及消融

在已配置CUDA容器/虚拟环境或HIP计算节点执行，当前开发环境不运行：

```bash
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmm_csr,spmv_coo,spmm_coo \
  --phase accuracy --mode quick --results-dir pytest_results_mixed_p2_accuracy
# quick后将mode改为normal，并更换results-dir。
```

原mixed用例增加fixed及panel加载配置，覆盖四种类型、操作、布局、合法索引和宽输出。
NNZ专用测试覆盖compact/fixed、小workspace、多层归并、空矩阵、长行及strided out。
COO carry用例保留块数1/2/4/8和segment/none，增加NaN预填strided out；策略测试覆盖非法配置与配置隔离。
源码接通不等于硬件验证，CUDA/ROCm编译、精度及性能均待计算节点验收。

全算法基线仍使用默认配置：

```bash
MATRIX_DIR=~/run/matrix
: "${MATRIX_DIR:?set MATRIX_DIR}"
test -d "$MATRIX_DIR" || exit 1
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmm_csr,spmv_coo,spmm_coo \
  --phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
  --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--type-combos float16:float16:float32,bfloat16:bfloat16:float32 --alg compare --ops non --index-dtypes int32 --indptr-dtypes int32 --timing" \
  --op-benchmark-args "spmm_csr=--op-b non --layout row --dense-cols 32" \
  --op-benchmark-args "spmm_coo=--op-b non --layout row --dense-cols 32" \
  --results-dir pytest_results_mixed_p2_compare
```

新配置用单算法显式消融，不将专用config传给compare：

```bash
for mode in compact fixed; do
  python tests/test_spmv_csr.py "$MATRIX_DIR" \
    --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --alg csr_nnz_tile_reduce --config "{\"plan_mode\":\"$mode\"}" \
    --ops non --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --csv-csr "spmv_nnz_$mode.csv"
  python tests/test_spmm_csr.py "$MATRIX_DIR" \
    --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --alg csr_nnz_panel_reduce --config "{\"plan_mode\":\"$mode\"}" \
    --ops non --op-b non --layout row --dense-cols 32 \
    --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --csv-csr "spmm_nnz_$mode.csv"
done
for load in 1 4 8; do
  python tests/test_spmm_csr.py "$MATRIX_DIR" \
    --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --alg csr_row_panel --config "{\"panel_load_nnz\":$load}" \
    --ops non --op-b non --layout row --dense-cols 32 \
    --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --csv-csr "spmm_panel_load$load.csv"
done
```

厂商默认开启；当前HIP明确不支持时可加`--no-vendor`，FlagSparse继续，结果仅称自身改进。
后续补同类型输出、全部op/opB、两种布局及N=8/128。固定其余因素后分别消融面板加载块、宽度、累加器、subgroup行块/lane、COO块数/归约/转换。
CUDA/ROCm分别记录软件版本、覆盖、精度和profiler；保留FAIL，报告单算法及逐矩阵最佳的算术/几何平均和回退。

### 2026-11-09 实机回归修复

11.09测试暴露两处本轮实现问题：

- csr_row_panel：static_range中重复注解slot为tl.constexpr，导致load1/4/8全部编译失败。现直接在分支使用k % ACCUMULATORS，保留原累加器分配和数值策略。
- SpMV NNZ：公共元数据硬编码compact的boundary_record_count，fixed报KeyError，compact又遗漏plan_mode。现按实际plan_mode返回boundary_record_count或boundary_slot_capacity，不将容量冒充有效计数。

SpMV实现版本升为7、SpMM升为6。默认配置、计时公式、容差及workspace策略不变。
原pytest增加compact/fixed、四种类型组合、return_time/return_meta/timing八种组合回归，检查输出身份、精度、计时公式及统计字段。
已有panel加载尾部和宽输出用例继续覆盖默认load1与load4/8；CI策略用例增加static_range内constexpr重复注解的静态检查。
本地仅AST和diff检查，未执行pytest、内核编译或算子。实机验证状态仍为待验证。

已在计算节点时，先使用原容器脚本定向验证：

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csr --phase accuracy --mode quick \
  --pytest-args '-k "nnz_tiles_batched_boundaries or nnz_return_contract or nnz_panels_bounded_workspace or panel_sparse_load_tails or registered_wide_output"' \
  --results-dir pytest_results_mixed_p2_fix_accuracy
```

定向通过后使用同一入口去掉--pytest-args做完整quick回归，再执行前文compact/fixed与panel1/4/8性能对照，写入新的结果目录。
FP16真实矩阵输入溢出、neos3/net150等数值FAIL需继续保留；本修复不通过放宽容差掩盖它们。


## Independent mixed candidates: source baseline (2026-11-09)

Base commit: `cbbac89a607cb3af8d292eafb9e567e0875131c0`, branch `zyq_sparse`. Existing unstaged repairs are retained. No runtime validation performed. Existing computational function SHA256 snapshots (AST source, excludes decorators) follow for static regression review.

<details><summary>Pre-change function hashes</summary>

```json
{
  "_spmv_csr_kernels.py": {
    "uses_native_rocm_fp32_accumulation": "d95608efbde99bddbd2f6f2ffbd9d33c2da5c4fbd1fb91f755611d83ee7edc6e",
    "_product": "efbadca5fc4d8ecba88a5bb3d036c3b993ad6d642b696436a8dd586cbc819fa9",
    "_store_result": "2bcffbc131448dcb923811f762f5e80f47175527203991d3aed5775e76f4bca3",
    "row_tile_kernel": "d3d0ac8aa0c3964ec21f361eae58a6c78bf9fbe680a57d426fddd8ad1a342ef9",
    "row_subgroup_stream_kernel": "6a83be100afc3d79ec67066e4ac7fe0ea9d3e857c05cfb176082bc5f60ec0d62",
    "row_tile_real_kernel": "bfc65589119dd76d6036502935520ac98f0abc28d6c584c6bd3e4b7a59529323",
    "row_vector_kernel": "c1f132155b2bd90ade944bcb8f2f7a633461f81c58455d62b8f21145523a8bcb",
    "bucket_rows_kernel": "ddea690075cd7d49fe8788fbd22f526b39748e9faba78b1fa0067a0ac7cc43f4",
    "classify_kernel": "b3a56f08eab148067e4f5c34a3a6cc8cbb5cd2c361f6c90d77d16f744d8d2197",
    "compact_rows_kernel": "8f0d3577261bee0de52a268521378480fd4d263d8c3262fc3fc023e636ae9b4f",
    "_owner": "b3f8870337cc682a646be428611c458853589927d293b6631d0e96bf726bbacc",
    "descriptors_kernel": "2dd67b991433bfa8f5c8944c1f0671aa9865b612740a3c945582f73d60d26eb5",
    "segment_kernel": "f63194457e83bd6462cd7b37feffa4e262c000d665b1da23b90f126cf330b7fa",
    "reduce_level_kernel": "6ee168ec7160ac3096f472642261b29a3ca16e186e5a77bc0af7e0fa48f2b667",
    "finish_kernel": "4954cdc541f7dbfbfc9b5990592335db8471cae78f6c8dbdd7b1d902f2bee2d1",
    "build_plan": "c0411159c957fb688bb7e27e934b23882e993aa829eba7f7251fa8d6e3fdaeb1",
    "compute": "13d35aecbf81ffe17f756d78ba28104904af288537d1d5820e260e95e8b7c9a1"
  },
  "_spmm_csr_kernels.py": {
    "rows_kernel": "90043703907034d4204caed6c0fd40a62635bd7d863db4a740000e24197ec918",
    "row_panel_kernel": "e7d2b40c447ccdf6e5e73a6ca9e33577f1cbb1e804ec5c147f36eceac86ef7a1",
    "reduce_kernel": "2ac8db6916b3f54a1e6e1c64c9d2d6a62c0c823662aa0f330cae4e0ad2a2e474",
    "finish_kernel": "2adf8385f9358a9005a644eb53e1957c3c81a8c8d1024ed0c096e47262783bab"
  },
  "spmv_coo.py": {
    "_spmv_coo_dtype_error_message": "124cc7d5c6db7fe2a15bf14ebf23c9cfbca444d0a317882fb2e2e1ff8734011d",
    "_normalize_spmv_coo_op": "41d0becbd081e51fd364bb2774ff51b4e0bc679452a9ea97a45d5d634519adab",
    "_spmv_coo_op_to_name": "a431a400ce09498c1a727f0a7bfb418eae9700ce728d8eb4fe1aac7228c5bf7e",
    "_spmv_coo_op_transposes": "89d6bf23e833f6da1586eab11d4629b7bc1d06657daa0f5a8df2c85a488a9d5a",
    "_normalize_spmv_coo_index_fallback_policy": "b5acf334972a7c390ae26df34a2d5bab0f7f5c22ccc1ed64752679dfc47d76da",
    "_spmv_coo_seg_f32": "bb0e43bb2fea63c7deb77533a624f80bed08ecce786da534ef83efa0f1e89161",
    "_spmv_coo_seg_f64": "81eea03ecd960acbbda24442f824f7209ff24da91c5c224c03566c935ee60952",
    "_spmv_coo_seg_complex": "0123fdc539b222a1d66cc6f1d2b3de0d026a389493b49437cc2cd9e950c9cc99",
    "_spmv_coo_atomic_f32": "757630bf4db990afbe1cc4b4e8771a8130307b6df30a12c1cc1c6a205be802cb",
    "_spmv_coo_atomic_complex": "ca8f5f33255e7d0224af3361e43448bd9651acba04cc03efa46cc8278890f662",
    "_spmv_coo_atomic_f64": "b0e5e8691be8981bd45e53c23a680b5c584a0e6cc0d3beedc86791f6d4a13a40",
    "_spmv_coo_run_add": "77d300f95942fd73d2047cf108bbfe174495ce3aeda738e8954341e1876d2c4e",
    "_spmv_coo_atomic_runs_real": "b583eb71c1da12efa45d7d3411e93141bc2a17d344e4a36c4aea0252f72f201c",
    "_sort_coo_lex_inplace": "8b35931cb8fb4bb9607a28626dcef48980dcf121123989dbb88e2a8edb9b7afb",
    "_seg_starts_from_sorted_rows": "43b9c9fe60ede56e713719da3a3f6511487a3240e10f76c799846b3a3cb88285",
    "_prepare_coo_tensors": "8975485d865a865aee958a95862cce30be59040473da3310d3cfc52697cac48e",
    "prepare_spmv_coo": "cff7dc70d266a7aac04e6c02d3868e309174370766a9f167a2fc0cf7d9e8030d",
    "_prepare_spmv_coo_launch_from_raw": "7824d343a4c20f74961c598b49fc47eb3474cf6dc9de9e4ffb4ec6c02cdfd432",
    "_resolve_spmv_coo_launch": "fc9109d9c5cb1bf4b88868ddaec4c7f6a83caeae97c152984ba45b1cc2b994dc",
    "_validate_x_coo": "604694b30b6f394906f3682ef8003aa11c61b8dc77d85a7092d2efcf140ae6f9",
    "_triton_spmv_coo_kernel": "755a031ba2f1c3b77d737e742b00428c65120f2acf665395ed8d5fbe54d2861b",
    "_spmv_coo_uses_int64_indices": "c57272fcc84227c01eeb8bb1d6dc06b6b2ebb5c696abb9928dbdeb54eb8e9353",
    "_spmv_coo_int32_fallback_blocker": "53b8dc4cc566c2ad88b3802bd3524232b4c4a8b66c04d6c788c7c6cb154ff8c7",
    "_spmv_coo_prepared_with_int32_indices": "2048bb05e6c18d483daf65c0a24a8b911ec004e1588e72439a2a39d5f78971ed",
    "_run_spmv_coo_prepared_with_fallback": "da0f5c320504f7adfdb75a05f78943c0b892bffefcdc166fb8849bea7fbe4811",
    "_ascend_spmv_coo_index_add": "39f94975703190d860e14790f0d4d5ce8c30c85a45b14f1b0f34413e6336c4af",
    "_cached_rocm_coo_launch": "4340f4de1e8012fa1e31d3f3e37b43ddde71b2674728a32313771c08848f0f5c",
    "_resolve_spmv_coo_kernel_launch": "1a0f8c6b62a871965c35717d89f6224629dd8815b4e726195530aec01188990b",
    "flagsparse_spmv_coo": "9a2fa7eca59a58ffd348b881b9518b5313ae6326bf373cd90830c08ced941b7c",
    "_coo_scan_pair": "44e95e764e35925dfbc92d00e58873042b452cadd10b4efaa93bdf7290a56a0a",
    "_coo_product": "56e3d3b9d744a64981ad2510c84491f57badf3bebdd288a0bccc598e65810a43",
    "_coo_segmented_panel_kernel": "761a9b0e51a75072b7453158f89c49fd5ac148a61a8cfd68231d460ce65cf0c3",
    "_coo_stream_panel_kernel": "9b834f0c9643d1b62a90d789b81b9985f1b6187a70cba0ee14eff962dc81aa95",
    "_coo_rowrun_group_kernel": "ceaf1d44350efa81e4006d17558c73dc136257010cf95d471610b8422a3ca298",
    "_resolve_coo_config": "7abf42fba66866b510e93ead37a66607fa002c772e735d21704de8a882337585",
    "_coo_sorted_runs": "8d29af726c9e25ce8d4681b62a905f4674111eb00d3c52ca442b13683aa265ee",
    "_launch_coo_extension": "379ffb96cd48c3f6347ff8038dc9721cd6da5bb69d09f005bb17c1df787d1066",
    "get_spmv_coo_algorithm_spec": "c3bf0d5448a022d5a208a5cae1e55145f5177ca39c4017e26ba9407bff881d00",
    "list_spmv_coo_algorithms": "38b5dea5501bdbadb7a42746634635a909d94f7b8d68487c42bd8b2858a5dbc9",
    "_coo_measure_run": "fe4209b6efeb3005b296297badff4a025da997f674c35d2f8a63fecc72823106",
    "flagsparse_spmv_coo_run": "375f08289a612cf4a94645af665e263817c3f8cb6072a119939f052631513db3",
    "_coo_index_compatibility_error": "6ca1156cb605fc51f1c148a2ffdc9df8982919b2dfa305b9cc2543bd67572e97"
  },
  "spmv_csc.py": {
    "_spmv_csc_dtype_error_message": "7ef3fbcb65f21475c6eb94cf9d87a7d102b182daf36415c5881e0af0045719a3",
    "_normalize_spmv_csc_op": "fa09cafb139a5b569dea6ef882f00128142546c2a14c1fad33c75959a1e71c34",
    "_spmv_csc_op_to_name": "a606e0795fee9a7a07aff4c69d2f345045a5738e92fef2bd23e6361cbab8c13c",
    "_spmv_csc_op_transposes": "7ceb9ba527ac250b3eff93ff2195a335697d34b491901ec30ea6b3887ea47784",
    "_normalize_spmv_csc_index_fallback_policy": "f2f31cc1928a5996371c89da0933c701f19d22d7634306d19412d4cda61dcc99",
    "_spmv_csc_non_real_kernel": "ed7b4e43cfbe77a632926fb9143f84708f7907e102e1dcba2e9e8b3cce504eea",
    "_spmv_csc_non_complex_kernel": "3edfc4bae8a32ce65c867e40dc9fcff0bf04cc3c4a76d1eb1bdc93542e9fd307",
    "_spmv_csc_trans_real_kernel": "dab40cfccb2dbe872e34613e9d9e87c2e89d8fa625629759beafa7b94ce667de",
    "_spmv_csc_trans_complex_kernel": "408b9d8ccd48d008d185c37da79f0facf06b2f745a6e5e0a15d543fc17e5c1b9",
    "_prepare_spmv_csc_matrix": "9caf1232e0c047e24fac8afdbbcb26ec7897dcc597b6be5b289b07542db0ece9",
    "prepare_spmv_csc": "f9762ec3c8e86758069cac50bf636a814e7a68ef618ed2e1ba1276aaed8cffa4",
    "_validate_spmv_csc_x": "c1c195421627b8bcca42434f5df180d483e583be008e088613376ef649a4513d",
    "_spmv_csc_non_real_nnzpar_kernel": "e5d8889688a2746abea781689cddfcdec0a69130f007dc8838f8589596dcb933",
    "_spmv_csc_non_complex_nnzpar_kernel": "31662456933c1801ec8c7a10fc11d61744ed7bcb12f7b6528e48f712e4644a21",
    "_ascend_spmv_csc_torch": "deb371db6c6579f6e140a2ce5ec94e431727015d24c8838b2aa1a43cf84a04d5",
    "_triton_spmv_csc_kernel": "7b59531ea543405a118079eff30051f761acabb16682a3b50fdab331e3384403",
    "_spmv_csc_uses_int64_indices": "b4c11fa3f13fc165355854f9e774ae96353332bad95d4d92c54f012118981f6e",
    "_spmv_csc_int32_fallback_blocker": "1424b4bfe6bfafd9c611a2dee39b78100c4e20ff3d58acf340c5d22385bcad33",
    "_spmv_csc_prepared_with_int32_indices": "290f9a58198b9a7cab7d815d8116c5c7a21b4420ec8ccb5c3dea0d8edcfd9ce4",
    "_run_spmv_csc_prepared_with_fallback": "336c7613b237eae63f5c8ad49d174049a5ed476990633516f826091ea58b696e",
    "_spmv_csc_via_mixed": "c22d028687aff70b14fddfa97c9637fdd1c11d296a21198d592da9ac773fc6b8",
    "flagsparse_spmv_csc": "b2d09c97badb6e1d367eb3651969bd02268509954a6db3b7c39a3c3d3dacf99e",
    "_csc_product": "eadecc05328451470c7d3bfeb4d3f72d726fd7ca6d59f1ac71317e17d5e70cb8",
    "_csc_columns_reduce_kernel": "068c8c14dd3ab1e75b3719f7aea2742995d8b0dbf61c6b13cda47c8f3f030ec0",
    "_csc_columns_atomic_kernel": "a38a0330a32dee0c2d33124afda9e3ffb2974447865ddaf04c065acbd4c8d1ed",
    "_csc_vector_kernel": "9bb0c089c0e01cbf4635e8c055700c89524b542d8450053542a8d587e7788085",
    "_csc_split_store_kernel": "4d3927a0f1fe79e8aa57a9ff76851c191af976a7873398e00700e17dd545457a",
    "_run_csc_split": "337cccd9e9b4910a449566d332aca4dccfd6288150c5fb482e4073f51656c7f1",
    "_launch_csc_columns": "6d47e56e60011bb1f91e002a0b21883be96fd5e29da868274a536f714b83f064",
    "_csc_spec": "823a5c5029faa38db7c2f4b405fff695e150098b9b37a9d5f3468e8971031353",
    "_prepare_csc_registered": "588a4ef0fabc61f3736c6885fbd06312c07070011ae0e6cc9c60f280d10cb7b2",
    "get_spmv_csc_algorithm_spec": "80d47a88c121d65e58f7c8cff3d4b51b12a857365e87690af892b12a6773cf40",
    "list_spmv_csc_algorithms": "99784f7100dfbc7ff28e8966023db5430defa7daaed877b79d81ba7889a6ac2a",
    "_run_csc_registered": "0810bda220a5b0481efc9208d134629693b37402477705a2fa1e874220ac666c",
    "flagsparse_spmv_csc_run": "48f849a466d314d173aa964892dc82501f81e9f3809ff1d612ae628b2c2c5286"
  }
}
```
</details>


## 六个独立 mixed 候选：实现与验收（2026-11-10）

本次保留旧计算 kernel、默认配置、默认路由及 NNZ compact/fixed。新增名称只声明
FP16/BF16 同类型输入、FP32 计算、同类型或 FP32 输出；不是旧算法别名。
源码和静态检查不代表 CUDA/ROCm 编译、正确性或性能已通过。

| 算子 | 新名称 | 旧对照 | opA |
|---|---|---|---|
| SpMV CSR | `csr_subgroup_pair_mixed` | `row_subgroup_stream` | non/trans/conj |
| SpMM CSR | `csr_panel_staged_mixed` | `csr_row_panel` | non/trans/conj |
| SpMV COO | `coo_stream_prefetch_mixed` | `coo_segmented_stream_atomic` | non/trans/conj |
| SpMM COO | `coo_stream_panel_prefetch_mixed` | `coo_segmented_stream_panel_atomic` | non/trans/conj |
| SpMV CSC | `csc_subgroup_pair_mixed` | `csc_col_subgroup` | trans/conj |
| SpMM CSC | `csc_panel_staged_mixed` | `csc_col_panel` | trans/conj |

SpMM 保留三种 opB 和原布局契约。CSC non 不使用这些新名称，不构造 CSR 委托。
公共 API、两级导出、六个 benchmark 和统一 runner 沿用原入口；compare 从注册表枚举。
专用配置只能配合一个显式算法，不能传给 compare。静态支持表的新增行标为 UNVERIFIED。

### 实现差异与可证伪假设

- Pair：每轮两个连续 V 宽子块，先加载两组稀疏及 dense 值，再分别 FP32 累加；
  结束合并双链并归约。`prefetch_chunks=1` 是单链消融。对象组顺序执行，不同时保留全部组的累加器。
  索引集合互斥且完整；额外寄存器与尾部工作可能使短行变慢，不能预先宣称异步流水。
- Staged panel：L=1/4/8，A=1/2/4，超块 U=max(L,A)=lcm(L,A)。固定 A 时加载 L 不改变
  每项的累加器归属。`value_stage=storage` 抽取后提升，`compute` 整个小块先提升。
  保持 A 个 [R,BN] FP32 累加器；不引入 [R,BK,BN] 累加器。
  tl.gather 的布局转换和寄存器占用必须实测，半精度存储不等于寄存器自动减半。
- COO prefetch：一次预取 1/2 个块的坐标、value、dense 输入，再逐块 scan/carry。
  carry 仅跨相邻同 owner 边界合并；不排序，不合并非连续同 owner。none 特化不含 scan/carry。
  SpMM load 模式按有效 B stride 直读，materialize 在 run 内转换作为消融。
- 所有中间累加、atomic 和 carry 为 FP32；窄输出最后转换一次。FP32 out 可以直接累加，
  但旧 COO 已有该机制，不把它重复算作新收益。

Pair 起点 V=8、2 warps、4 顺序对象组、2 子块；CSR 使用嵌套配置
`{"csr_subgroup_pair_mixed":{"prefetch_chunks":2,"lanes_per_row":8,"row_tiles_per_program":4}}`。
CSC 使用平铺配置，字段为 lanes_per_column、columns_per_program、column_tiles_per_program、prefetch_chunks。

Panel 起点 CSR R=4/BN=32，CSC R=4/BN=16，L=4/A=2，2 warps。
字段 panel_load_nnz、panel_accumulators、value_stage；CSR 使用 tile_rows/tile_n，CSC 使用 columns_per_program/block_n。
COO 起点 SpMV T=128，SpMM T=32/BN=16，blocks_per_program=2、prefetch_blocks=2，4 warps。
所有 stage=1；显式非法配置报错，未知能力不冒充已支持。

### 计算节点命令

以下均在已经分配的 CUDA 计算节点、仓库根目录执行。脚本负责容器和虚拟环境。
登录节点提交时用 `sbatch --partition=gpu_a800 tools/run_cuda_tests.sbatch --repo "$PWD" -- ...`；
bash 本身不申请计算节点。ROCm 使用其已有环境直接执行同一 runner 参数，不使用 CUDA 容器包装。

先运行原六算子 quick 准确性（包含新算法的全类型/op 测试及专门边界）：

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
--ops spmv_csr,spmm_csr,spmv_coo,spmm_coo,spmv_csc,spmm_csc \
--phase accuracy --mode quick \
--results-dir pytest_results_mixed_independent_accuracy
```

normal 准确性使用相同命令，将 mode 改为 normal，另设 results-dir。
快速定位新增覆盖可额外传：
`--pytest-args '-k "independent_mixed_boundaries or registered_half_combinations or registered_wide_output"'`。

真实矩阵四组合比较，SpMM 默认 N=32；CSC 必须包含 trans/conj 才有新候选：

```bash
MATRIX_DIR="$HOME/run/matrix"
test -d "$MATRIX_DIR" || exit 1
COMBOS='float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32'
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
--ops spmv_csr,spmm_csr,spmv_coo,spmm_coo,spmv_csc,spmm_csc \
--phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
--benchmark-warmup 10 --benchmark-iters 50 \
--benchmark-args "--type-combos $COMBOS --alg compare --ops all --index-dtypes int32 --indptr-dtypes int32 --timing" \
--op-benchmark-args "spmm_csr=--op-b non --layout row" \
--op-benchmark-args "spmm_coo=--op-b non --layout row" \
--op-benchmark-args "spmm_csc=--op-b non --layout row" \
--results-dir pytest_results_mixed_independent_compare
```

上述命令保留 vendor。厂商不支持时记录原因，FlagSparse 继续。当前不支持组合的 DTK
可显式在 benchmark-args 加 --no-vendor；自身收益不能称为厂商加速比。
扩展布局/opB时对 SpMM 透传 `--op-b all --layout all --dense-cols 8,32,128`。

单算法消融示例（各组合单独记录；修改 L 后分别写入不同 CSV）：

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmm_csr.py -- \
"$MATRIX_DIR" --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
--alg csr_panel_staged_mixed \
--config '{"tile_rows":4,"tile_n":32,"panel_accumulators":2,"panel_load_nnz":4,"value_stage":"storage"}' \
--ops non --op-b non --layout row --dense-cols 32 \
--index-dtypes int32 --indptr-dtypes int32 --warmup 10 --iters 50 --timing \
--csv-csr mixed_staged_L4_storage.csv
```

消融依次固定其他参数比较：L=1/4/8及value_stage；pair子块1/2再V/G；
COO预取1/2，再segment/none，再SpMM load/materialize。blocks_per_program必须能被prefetch_blocks整除。
不同时改变多个因素后将收益归给一个因素。

### 统计和验证边界

保留 main prepare/run 边界，ms=process_cpu_ms+gpu_ms，完整采样与分段诊断分开，
沿用中位数±10%过滤。CSV保留实际配置/stride/算法及准确性状态，FAIL的有效性能不剔除。
固定类型四元组、opA/opB、layout、N、索引和平台，报告各算法、旧最佳、新最佳、胜出比例、
算术/几何平均及逐矩阵回退。未覆盖矩阵不能与旧完整30矩阵结果混称同批收益。

本地只检查AST、diff及旧kernel源码摘要，不执行上述命令或导入算子。
计算节点需要额外核对寄存器、spill、布局转换、加载事务、atomic及kernel数量；
只有完整调用实测有收益，才能推荐新配置，默认路由仍不切换。


## Optional mixed benchmark work metrics (2026-11-10)

The existing generic performance framework primarily reports latency and speedup.
The six existing CSR/COO/CSC SpMV/SpMM scripts now accept `--perf-metrics` through
`tests/benchmark_utils.py`, with pure metric helpers in `tests/utils.py`.
This opt-in switch currently requires `--type-combos`; requesting it on a legacy
branch fails explicitly instead of silently omitting metrics. No new entry point,
algorithm, kernel, default route or vendor policy is introduced by this switch.
The original runner forwards it in `--benchmark-args`.

### Definitions and interpretation

`ms = process_cpu_ms + gpu_ms` remains the denominator. Prepare is excluded in
this registered mixed benchmark; required per-run work remains inside the run.
Diagnostics are separate and are never substituted for the complete run.

Let Z be stored NNZ (including duplicates and explicit zeros), W be 1 for SpMV
or effective output-column count N for SpMM, and O be the output row count after
opA. Model `spmx_logical_access_v1` uses:

- Useful FLOPs = `2 * Z * W` (one real multiply and add per contribution).
- CSR sparse bytes = `Z * (sizeof(A) + sizeof(index)) + (M+1) * sizeof(indptr)`.
- CSC sparse bytes uses `(K+1)` pointers for original shape M by K.
- COO sparse bytes = `Z * (sizeof(A) + sizeof(row) + sizeof(col))`.
- RHS gather bytes = `Z * W * sizeof(RHS)`.
- Output write bytes = `O * W * sizeof(output)`.
- Logical bytes = sparse bytes + RHS gather bytes + output write bytes.
- `useful_gflops = useful_flops / (ms * 1e6)`; `logical_gbs` uses the same divisor.
- `gnnz_s = Z / (ms * 1e6)`, regardless of W; arithmetic intensity is useful
  FLOPs / logical bytes. Zero-sized output reports zero work; invalid/nonpositive
  time yields missing rates with a reason.

These are decimal giga-units. The logical model uses original input storage and
index widths, and is identical for different algorithms under the same condition.
It is **not measured DRAM/HBM traffic, a compulsory-traffic lower bound, or peak
bandwidth utilization**. It excludes cache reuse, repeated panel loads, padding,
atomic read/modify/write, temporary buffers, transpose/plan traffic and conversions.
SpMM may reuse B substantially, so a high logical GB/s can exceed physical bandwidth
without indicating a measurement error. Do not label this field effective hardware
bandwidth. Useful GFLOP/s likewise excludes extra executed arithmetic and padding.
For fixed inputs these rates rescale latency; they alone do not identify a bottleneck.

Optional CSV fields include the byte components, `gpu_useful_gflops` and
`gpu_logical_gbs` using only full-run GPU time, plus GPU sample count, kept count,
min/median/max and coefficient of variation (population stddev / mean, all samples).
The sample filter is unchanged: mean inside median +/-10%, with median fallback
if no sample survives. `sample_kept=0` records that fallback, not an empty timing.
FAIL retains valid rates and its separate accuracy status. Interface/runtime ERROR
must not be ranked as a usable result merely because an earlier timing exists.

Input fingerprints hash quantized CPU values and int64-normalized coordinates /
pointers; RHS fingerprints hash the effective RHS before opB storage transformation.
They are generated outside timing. CSV records seed, device, software versions,
source commit and dirty flag. Match fingerprints, dtype quadruple, opA/opB,
layout, N, indices, actual algorithm/config and prepare/run boundary before comparing
DCU and A800. A dirty flag alone does not identify a patch: retain the exact source
diff with each experiment. Matrix fingerprints intentionally differ across formats.
Report platform ratios from matched latency pairs, then summarize their geometric
mean; do not divide aggregate GFLOP/s values for unrelated matrices.

For actual memory traffic/occupancy/register/spill diagnosis, use hardware profiling
in a separate run: NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
and AMD [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/what-is-rocprof-compute.html).
AMD tool availability does not establish support on a particular DCU/DTK build;
confirm the local profiler and counters. Profiled/replayed time is not substituted
for normal benchmark time. No peak-bandwidth constants are assumed by this interface.

### Compute-node commands

In an **already allocated CUDA compute node**, use the existing container wrapper.
Set and validate the real matrix path in this shell. This includes CSC trans/conj
so the new CSC candidates are actually exercised. SpMM remains N=32, row layout,
opB=non; `--ops all` is opA. The two wide-output combinations are the first
performance priority, not the complete four-combination correctness suite.

```bash
MATRIX_DIR="$HOME/run/matrix"
test -d "$MATRIX_DIR" || exit 1
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csr,spmv_coo,spmm_coo,spmv_csc,spmm_csc \
  --phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args "--type-combos float16:float16:float32,bfloat16:bfloat16:float32 --alg compare --ops all --index-dtypes int32 --indptr-dtypes int32 --perf-metrics --timing" \
  --op-benchmark-args "spmm_csr=--op-b non --layout row --dense-cols 32" \
  --op-benchmark-args "spmm_coo=--op-b non --layout row --dense-cols 32" \
  --op-benchmark-args "spmm_csc=--op-b non --layout row --dense-cols 32" \
  --results-dir pytest_results_mixed_metrics_cuda
```

This command keeps the vendor baseline enabled. For a DCU environment whose vendor
support has already been checked as unavailable, use the same runner after activating
the existing HIP environment, and explicitly disable only the baseline:

```bash
MATRIX_DIR=/path/to/matrix
test -d "$MATRIX_DIR" || exit 1
python run_flagsparse_pytest.py \
  --ops spmv_csr,spmm_csr,spmv_coo,spmm_coo,spmv_csc,spmm_csc \
  --phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args "--type-combos float16:float16:float32,bfloat16:bfloat16:float32 --alg compare --ops all --index-dtypes int32 --indptr-dtypes int32 --perf-metrics --timing --no-vendor" \
  --op-benchmark-args "spmm_csr=--op-b non --layout row --dense-cols 32" \
  --op-benchmark-args "spmm_coo=--op-b non --layout row --dense-cols 32" \
  --op-benchmark-args "spmm_csc=--op-b non --layout row --dense-cols 32" \
  --results-dir pytest_results_mixed_metrics_dcu
```

For all four tuples replace the type list with
`float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32`.
For a small interface/compilation check before the matrix set, invoke an existing
script via the wrapper, for example `--entry tests/test_spmv_csr.py -- --synthetic
--type-combos float16:float16:float32 --alg compare --perf-metrics --warmup 2 --iters 5`.
Repeat with each of the six script names; add `--ops all` to include CSC trans/conj.
Do not pass a fictitious `synthetic` filename to `--benchmark-input`.
Use the original accuracy phase first:

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmm_csr,spmv_coo,spmm_coo,spmv_csc,spmm_csc \
  --phase accuracy --mode quick --results-dir pytest_results_mixed_accuracy_cuda
```

### Source audit and validation boundary

Static review covers six CLI hooks into the shared mixed branch; runner forwarding
and CSR SpMM script selection; query/high-level/run output-type arguments; SpMM
opB handling; independent mixed kernel dispatch; FP32 atomic/partial buffers and
wide output; CSC new-candidate trans/conj restrictions; configuration isolation;
separate full-run/diagnostic measurement; vendor failure isolation and FAIL timing
retention. Existing accuracy tests cover the four tuples, API/run equivalence,
operations, tails, strided out, aliases, input immutability and configuration cases.
The existing benchmark policy test file gains metric-formula, empty/invalid-time,
sampling-window and six-script opt-in coverage; no new test entry point is created.

This is **source integration and static audit, not hardware acceptance**. No operator,
pytest, functional import or offline compilation was run locally. AST and diff
checks cannot establish Triton lowering success, device capability, numerical
correctness or speed. CUDA and DCU must each run the original quick accuracy and
synthetic benchmark first; inspect ERROR/SKIP reasons and CSV coverage, then run the
matrix set. Do not infer successful coverage solely from runner exit code.


### Timing clarification and measured reference normalization (2026-11-10)

The timing audit found that legacy CSR SpMM alg1/alg2 and COO SpMM alg1 construct
CPU bucket descriptors **inside** the outer start/end event interval. Their CPU
measurement also includes count/offset readback. Event elapsed time is a device
timeline interval, not a sum of active kernel durations: submission gaps can be
included. Consequently `process_cpu_ms + gpu_ms` can double-count part of the CPU
interval. Never estimate an overlap correction by subtracting CPU time from events.
New mixed candidates report zero CPU processing, so this particular sum does not
add a second CPU term for them. This does not prove their events exclude idle gaps.

Keep the existing `ms` sum and vendor speedup for historical compatibility. With
`--perf-metrics` additionally measure `prepared_run_ms` in a **separate** loop:
synchronize before starting the host clock, call the same prepared algorithm with
internal timing/metadata/diagnostics disabled, synchronize after the call, stop the
clock. Apply the same median-window filter. This includes the run API/host work and
completion synchronization, excludes prepare and the pre-start synchronization,
and measures single-call completion latency (not pipelined batch throughput).
There is no separate CPU term to add to this measurement. GPU samples, wall samples
and phase diagnostics are separate populations; their values need not sum or match.
The metric switch therefore adds another `iters` calls, not extra work inside the
original event samples. CUDA/ROCm compilation remains outside sampling.

- Prefer `gpu_useful_gflops` / `gpu_logical_gbs` for the GPU event interval view;
  they are not pure GPU arithmetic efficiency or active-kernel rates.
- Prefer `prepared_run_ms`, `prepared_useful_gflops`, `prepared_logical_gbs` for
  single prepared-call completion efficiency.
- `cpu_gpu_overlap_risk` is true when any reported CPU sample is nonzero; it is
  a conservative warning, not a measured overlap duration. Do not assume the
  historical `ms` is wall-clock latency. The original speedup column is unchanged.
- The optional terminal columns now show Wall, GPU-GF/s and GPU-LGB/s. Previous
  section descriptions of the optional terminal columns are superseded here;
  existing CSV `useful_gflops` and `logical_gbs` still use the historical ms sum.

Optional reference arguments (decimal GB/s and GFLOP/s):

```text
--reference-bandwidth-gbs VALUE
--reference-bandwidth-source "BabelStream version/backend; device; Triad; FP32; array size; iterations; statistic; clock/ECC; date/log"
--reference-peak-gflops VALUE
--reference-peak-source "same-device FP32 scalar/SIMT FMA peak; clock; theoretical or measured; source"
```

Bandwidth value and source must be supplied together. Compute peak and source are
optional, but require the bandwidth reference to construct the logical Roofline.
All values must be positive and finite; the options require `--perf-metrics`.
No A800/BW200 bandwidth is guessed or hardcoded. Measure each device separately
using [BabelStream](https://github.com/UoB-HPC/BabelStream), with arrays larger than
cache and comparable test settings. If the tool prints decimal MB/s, divide by
1000 for this argument. Record the operation (e.g. Triad), precision, data size,
statistic and full log; do not silently treat its best/min-time rate as an average.
The storage format of the sparse test need not equal BabelStream's data type, but
its choice is part of the reference definition and must be disclosed.

For Q logical bytes, F useful FLOPs, historical total time T, GPU interval Tg:

- `logical_bw_ratio = logical_gbs / reference_bandwidth_gbs` (requested original ratio).
- `gpu_logical_bw_ratio = gpu_logical_gbs / reference_bandwidth_gbs` (GPU-oriented view).
- Model intensity `I = F/Q` in FLOPs/byte.
- `logical_roofline_gflops = min(reference_peak_gflops, reference_bandwidth_gbs * I)`.
- `gpu_logical_roofline_ratio = gpu_useful_gflops / logical_roofline_gflops`.

Missing references leave derived fields empty, never zero or fabricated. Ratios
are not clipped to one. They are **logical-model normalization**, not hardware HBM
utilization or a rigorous upper bound: cache reuse and modeled repeated gathers
can produce ratios above one. These references require no profiler, but a logical
Roofline cannot by itself establish the physical memory/compute bottleneck.
The present mixed kernels use FP32 scalar/SIMT arithmetic, not Tensor Cores. Do not
use advertised FP16 Tensor Core throughput (or sparsity-doubled throughput) as their
compute ceiling. If no matching FP32-path peak is established, omit peak arguments
and use only measured bandwidth normalization.

Append reference options inside the existing `--benchmark-args` string from the
commands above; supply device-specific measured values, not the same number for
A800 and BW200. Retain all raw CSV fields and separate accuracy status. Compare
prepared wall times separately from historical event-plus-CPU speedups.

Sources: [CUDA event semantics](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EVENT.html)
and [synchronized host timing](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).
Local verification remains source/AST/diff only; the new measurements and reference
normalization have not been hardware-validated on either platform.
