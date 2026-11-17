# SpMV 独立候选与低精度 baseline 排查落地计划

日期：2026-11-17。目标库：FlagSparse-main / zyq_sparse。

本轮已完成下述三个候选及 baseline 诊断的源码接入，GPU 编译、正确性与性能尚未验证。起点commit为 `d7633c1474e6cd913526963d8ca476e8351b5030`，计划编写前工作区和暂存区无修改；实施时保留已有计划文档修改。只进行源码、CLI参数和文档静态检查，不执行算子、pytest、功能导入或离线编译，不暂存或提交。

依据：[11.17数据分析](../../../test_mixed_data/11.17data/analysis/ANALYSIS_REPORT.md)、[BF16 baseline复核](../../../test_mixed_data/11.17data/analysis/BF16_BASELINE_REVIEW.md)。本计划按FlagSparse开发SOP组织；计时采用用户已确认的完整run公式，覆盖SOP中分段之和的旧约定。

## 1. 固定交付范围

先交付三个**新注册名称、独立kernel**，不修改旧算法计算体、默认配置、默认路由、缓存/图绑定名单及C API旧JIT签名。

| 优先级 | 算子/新名称 | 首轮op | 核心差异 | 旧对照 |
|---|---|---|---|---|
| P1 | SpMV CSR `csr_row_scalar_bound` | non | 每program一次计算最大行长，去掉scalar每轮active归约 | csr_row_scalar_strip、row_subgroup_stream、row_tile |
| P1 | SpMV CSC `csc_col_scalar_bound` | trans/conj | 原生短列scalar，无行内lane归约、排序或转置重建 | csc_col_subgroup、csc_col_vector及base |
| P2 | SpMV CSR `csr_longrow_queue_atomic` | non | 普通行独占输出，只有长行进入GPU任务队列，跨program计算 | row_vector、legacy_segbin、row_split_reduce、row_adaptive_split |

两条CSR新路径首轮仅non，明确避免把短行策略包装为转置优化。CSC直接覆盖trans/conj。后续CSR转置散射、COO有序边界归并不在本轮实施；CSC长列队列在第7节门槛通过后另行立项，不预注册空实现。

所有候选覆盖六种输入：FP16/BF16/FP32/FP64/complex64/complex128，A与x同类型。半精度支持同类型或FP32输出；不增加异类型输入、INT8或任意计算精度接口。

| 输入 | CSR新算法计算及部分和 | CSC新算法计算及部分和 | 输出 |
|---|---|---|---|
| FP16/BF16 | FP32 | FP32 | 输入类型或FP32 |
| FP32 | FP64，保持现有新CSR策略 | FP32，保持CSC策略 | FP32 |
| FP64 | FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | FP32实虚分量 | complex64 |
| complex128 | FP64实虚分量 | FP64实虚分量 | complex128 |

有计算体变化时新增算法；仅R/K/warp参数消融不另起同义名称。全部只通过显式alg或compare调用，默认行为不变。只修改现有Python文件，不建新运行框架、调优器或跨调用计划缓存。

## 2. 短行与原生短列实现

### 2.1 CSR scalar_bound

每program拥有R个连续行，加载start/end，执行一次 `steps=max(end-start)`，随后按k从0到steps−1读取位置 `start+k`，在乘法前转换到计算精度，维持[R]累加器。最终按真实out stride写回。

与旧scalar相比，消除每轮R维active计数归约和pos饱和更新；不增加预取、双累加或动态补位。每行累加顺序与旧scalar一致。超长行仍完整执行，不能以优化短行为由截断或静默换算法。

初始配置：`rows_per_program=64, num_warps=2, num_stages=1`。显式R候选32/64/128；先保持旧参数只比较计算体，再改变R。默认warp依据main能力集合合法解析，非法用户配置报错。

边界要求：尾行mask、空组steps=0且写零、索引和dense加载均mask、地址乘法保持所需宽度。实数编译分支不保留复数计算；复数两个分量独立累加。FP32宽输出不得经过半精度缓冲。

### 2.2 CSC scalar_bound

按原生CSC列指针实现相同思想，每program拥有R列。输出 `y[col]`，读取 `x[row_indices[p]]`；trans不共轭，conj只取A的共轭，不改变x。不调用CSR算子，不构造owner-map或转置矩阵。

初始配置：`columns_per_program=64, num_warps=2, num_stages=1`，R候选32/64/128。沿用CSC的计算类型策略；不得直接复用CSR的FP32→FP64数值分支。可以共享纯数学/校验helper，不能通过转格式复用算子。

先测短列trans/conj，不能从CSR non的收益直接推定CSC收益。长列完整运行但不作为其性能目标。

## 3. 长尾：GPU长行任务队列

### 3.1 固定结构和配置

将长行阈值L与任务段长S区分，先固定 `long_row_threshold=1024, segment_nnz=1024`；初始生产者R=8、短/中行K=32，消费者 `block_nnz=128`，stage=1。生产者/消费者warp分别使用现有合法集合中的保守值；不把短行候选或未知能力直接当作队列配置。

专用扁平config为：

```
long_row_threshold=1024
segment_nnz=1024
rows_per_program=8
short_block_nnz=32
block_nnz=128
producer_num_warps=4
num_warps=4
worker_programs=128
num_stages=1
workspace_limit_bytes=268435456
```

这些是待验证起点，不能标为已调优。L初测1024/4096，S初测512/1024/2048，分开改变；L/S不要求相等。worker_programs是固定有限消费者数量，显式合法正值；不依赖新增跨平台SM数量接口。

首版验证规则：L、S、short_block_nnz、block_nnz、R为合法正二次幂，S是block_nnz的整数倍；R取4/8/16，worker_programs取32/64/128/256，stage固定1。生产者和消费者分别校验warp/线程/张量大小限制。显式非法配置报错；workspace为正整数，容量计算不得溢出。

### 3.2 四个顺序阶段

1. **计数器初始化。** 创建/清零GPU的长行计数、job计数、领取游标，均在run内。
2. **普通行计算与job生产。** 每行判定长度。长度≤L的行完整计算后直接写out；组内循环上限只对这些普通行取max，排除长行。长度>L的行不写最终输出，而是领取唯一长行slot，清零该slot的计算精度累加值，并生成长度≤S的job。
3. **长行job消费。** 所有生产结束后，有限worker通过整数atomic领取job。job保存long-slot与NNZ起点，row从slot-owner取得，终点由min(start+S,row_end)得到。每段先局部归约，再对slot做一次实分量atomic，复数两次。避免逐NNZ二分owner。
4. **最终写回。** 消费kernel全部结束后，按长行slot写对应out，最后才窄化。普通行输出与长行输出互斥。

四阶段在同一实际执行流上顺序launch。禁止消费者与生产者在同一kernel自旋等待；禁止假设普通Triton有跨program全局屏障。领取阶段读GPU计数，不执行 `.item()` 或CPU同步分配。

生产者可以用组内整数prefix一次预留一组slot/job区间，避免每job一次生产atomic；job描述符分小块写入，不能按最大长行段数生成无界constexpr张量。即使生产顺序不确定，每个slot、job及最终输出也必须唯一。

### 3.3 容量、数值及性能风险

仅使用已知M和nnz推导容量：

```
long_capacity = min(M, floor(nnz / (L+1)))
job_capacity  = floor(nnz / S) + long_capacity
```

证明：每条长行长度≥L+1；各行ceil(length/S)之和不超过floor(total_long_nnz/S)+长行数。所有计算使用Python整数及足够宽的设备整数，不先截断int32。

workspace至少计入：job的slot/start描述符、long-owner、计算精度long-accumulator、计数器及实际临时缓冲。描述符和数值空间分别记录，总分配受256MiB默认上限约束。超限明确报错，不静默少分配、减小S或换算法。没有长行的实际计数可以为0；容量上界不是实际数量，不为统计再读回计数。

半精度atomic为FP32；CSR FP32路径为FP64 atomic；complex128为FP64分量atomic。需要显式检查整数队列atomic和相应浮点atomic能力；未知即不可用。初版选用明确支持的计数宽度实现，不以“支持int64寻址”代替“支持int64 atomic”。

首版计数器使用int32 atomic：必须证明long_capacity、job_capacity以及消费者最终游标预留余量均≤INT32_MAX，否则明确报配置/容量错误；不做未经验证的窄化。消费者耗尽时的额外领取次数上界为worker_programs，应计入游标范围检查。NNZ起止、指针和stride地址仍使用int64。能力对象增加明确的int32-atomic声明，不自动借用FP32-atomic字段。

没有完整输出的额外清零；普通空行直接写零，长行slot只初始化一次。零维度和nnz=0有正确空/零输出路径。所有必要初始化、生产、消费、归并与转换包含在完整run。

风险包括四次launch、描述符生成、整数领取竞争与长行slot竞争。它区别于refill的单program长行，区别于legacy_segbin的逐NNZ定位，也区别于全矩阵NNZ边界归并；但并非天然更优。首轮必须与旧最佳对照，不只和scalar对照。

## 4. main接口及文件落点

| 现有位置 | 计划变更 |
|---|---|
| `_spmv_csr_config.py` | 两个CSR名称、专用配置、类型/op/能力声明；队列整数atomic能力；禁止专用config串用 |
| `_spmv_csr_kernels.py`、`spmv_csr.py` | 独立kernel及独立执行分支；计时/元数据；不进入旧缓存或直接绑定分支 |
| `_spmx_column_config.py`、`spmv_csc.py` | CSC新名称、独立列scalar；trans/conj能力筛选及直接写out |
| `tests/test_spmv_csr.py`、`test_spmv_csc.py` | 原compare按规格枚举新名称，unsupported op明确排除 |
| 原CSR/CSC accuracy与CI策略文件 | 合法op参数化、混合精度、索引/out、队列生命周期和配置测试 |
| `benchmark_utils.py`、`cusparse_generic_baseline.py`、`test_spmm_csc.py` | 第5节诊断、厂商参数和结果记录修正 |
| `run_flagsparse_pytest.py`、原GPU CI入口 | 保持默认脚本；原参数透传可调用新算法与诊断；不增加第二套runner |
| 两级公开导出、`conf/operators.yaml`、支持矩阵及相关文档 | 核查原API无需新增顶层函数；同步算法规格与真实支持范围 |

公共函数签名、返回结构不变。prepare不缓存队列；run每次重新构建。保留原始输入重叠检查、strided out、返回身份、类型冲突、prepared/op冲突，以及同算法安全索引回退。所有fallback完整保留alg/config/op/输出类型。

元数据至少记录算法版本、实际config及来源、计算/输出类型、backend/target、原生格式、queue容量、分配峰值、stride及转置策略。实际GPU计数与容量不同，不伪造精确job数量。

计时固定 `ms=process_cpu_ms+gpu_ms`；完整样本不插分段事件，诊断另行执行。正常CPU算法处理为0。队列生产融合了普通行计算，首版将整个融合kernel标为compute，并额外记录 `fused_plan_compute_ms` 诊断；不能把含普通行乘加的整段时间误标纯process。计数初始化可归process，长行消费和最终写回归compute。总时间不由phase之和替代。

## 5. Baseline排查：先定性，再改实现

### 5.1 已知问题及首要修复

本批CSC BF16→BF16 baseline 29/30 FAIL，mip1约239ms、最大绝对误差293；这是需要排查的记录，不能当作已证实正确的39倍优势。原始FAIL性能继续保留。

先修现有shared benchmark的速度记录条件：只要求native/vendor时间为有限正值，不要求双方PASS；状态和误差独立保存。修复旧 `csr_subgroup_pair_mixed` strided-out回归，仅修契约，不改计算精度或容差。

检查准确性golden依赖已量化输入；补最大相对误差、归一化误差、失败元素数及有限性字段，不直接用最大绝对误差判定原因。计时必须仍使用同原始CSC、同输出类型；FP32结果再转BF16只能作为诊断，不能替代原生BF16-output baseline。

### 5.2 原测试入口中的诊断接口（源码已接入）

在原 `tests/test_spmm_csc.py` 及共享helper中增加：

- `--vendor-diagnostic ones_fanin,rectangular`：生成小型确定性样本，经原FlagSparse与baseline执行链，不新增Python测试入口。
- `--diagnostic-lengths 128,256,512,1024,2048,4096`：仅与ones_fanin使用。
- `--vendor-alg default,csr_alg1,csr_alg2,csr_alg3`：CUDA显式候选；按官方头文件核对映射及真实支持，不拿CUDA枚举套HIP。
- `--vendor-preprocess off,on`：off保留既有CUDA默认；on只对支持且有意义的组合执行。缺失/不支持单独记录，不偷偷按off执行。

厂商扫描对每个算法/预处理组合产生独立结果行，FlagSparse结果可复用一次已完成测量并显式标记。厂商结果缓存键扩展为输入身份/形状、dtype四元组、op、stride、vendor_alg、preprocess、backend，不能仍仅以output_stride缓存。

一次运行中传入不识别参数属于ERROR；厂商明确不支持属于SKIP/N/A；vendor正确性FAIL仍保存时间；运行异常打印上下文和traceback，保留已有CSV。厂商失败不阻止FlagSparse完成。

### 5.3 诊断样本与裁决顺序

1. ones_fanin：CSC形状1×L，每列一个值1，B全1，N=1及32；期望结果L。保持compute=FP32，比较FP16/BF16同类型输出及FP32输出。能定位累加停滞，不能单独证明描述符方向正确。
2. rectangular：非对称稀疏矩形、非重复的dense值，分别non/trans/conj、row/col，检查形状、ld和操作映射。
3. 输出初态：zero/NaN预填；beta=0重复运行应独立于C初态。预填在专门准确性诊断的测量外，不能永久预清零掩盖厂商问题。
4. mip1、ecology1、c8_mat11、TSOPF真实矩阵：同一A/B、四组合；保存可复现输入指纹和异常输出样本。
5. default及合法ALG1/2/3，适用时preprocess：准备耗时、稳态耗时分开；不以未验证参数自动替换旧baseline。
6. 最小独立C/C++ Generic API复现：优先扩展已有C API测试target，使用同一CSC/stride/操作/类型/算法。接通后核对构建与运行命令再写文档，不在本计划虚构现成可执行目标。
7. profiler记录库kernel时间/数量、GPU空隙；记录实际libcusparse绝对路径/hash、版本、workspace、设备和时钟。必要时另一库版本复核。

判断规则：C++与ctypes不一致先查包装；两者一致但窄输出异常查厂商路径；只有某alg/版本异常才报告该范围；若只有过滤均值异常则查采样。不得直接将全部FAIL归因于容差，也不得仅凭源码无明显错误宣称baseline无问题。

## 6. 原pytest、benchmark及runner验收

### 准确性

- 在原 `test_spmv_csr_accuracy.py`、`test_spmv_csc_accuracy.py` 内按规格生成新算法合法op，沿用marker、quick/normal、golden、共享容差与JSON。
- quick覆盖六dtype和四种半精度组合；CSR non、CSC trans/conj。代表性合法索引，普通/strided out，空维度/空行列、重复项、未排序索引、复数、NaN预填、对象身份与输入不变。
- 队列专用：全普通行、全长行、单超长行、T边界、S尾部、容量上界、多个生产者、job领取耗尽、空队列、低workspace报错、长/短输出互斥。测试不能用GPU计数回CPU的调试路径冒充正式run。
- normal扩四种索引组合、长归约、抵消与FP32宽输出保真；后台能力缺失才SKIP，编译/配置/运行异常必须FAIL。
- 原SpMM CSC accuracy中加入baseline小样本，CUDA缺库/明确不支持可SKIP；baseline数值错误必须作为对应测试失败记录。不能让测试只调用FlagSparse却声称检查了vendor。
- 扩原CI配置、CLI/public API、baseline contract/gate测试，检查新参数和生命周期；不新增重复收集入口。

### 性能

先小样本，再目标子集，再30矩阵。regular与mixed分目录；旧/new及baseline保持相同输入、精度策略和准备/运行边界。四种mixed均回归，宽输出优先优化；CSC另测trans/conj，不与non混合GM。

记录旧最佳、新最佳、单算法GM/AM、胜出比例和回退。单次oracle增加候选后不会下降，需至少5轮随机交错确认，不能只取最好一轮。保留FAIL有效耗时，baseline失败不传播成全部算法失败。

## 7. 分阶段交付与停止条件

| 阶段 | 交付 | 完成/停止条件 |
|---|---|---|
| 0 | 固定旧kernel AST摘要、默认配置、commit/diff；baseline诊断和记录修复 | 所有参数链路静态可达；厂商异常可解释地记录，不要求所有vendor组合PASS才开发FlagSparse |
| 1 | CSR scalar_bound + CSC scalar_bound | 独立计算体、全部声明dtype/op有用例；计算节点小样本通过后才跑性能 |
| 2 | 目标子集性能与profile | scalar目标子集GM≥1.10×旧最佳作为继续投入参考；无净收益则保留实验，不扩大默认 |
| 3 | CSR longrow_queue_atomic | 先容量/唯一性/同步审查，再小样本；目标子集GM≥1.20×且至少两矩阵稳定胜出才扩展 |
| 4 | 30矩阵及CUDA/ROCm分别验收 | 不合并平台统计；逐项报告源码、正确性、性能三种状态 |
| 后续门槛 | CSC长列队列、CSR转置散射或COO有序归并 | 先取得对应操作的旧最佳/成本证据，不自动进入本轮交付 |

门槛是研究投入标准，不是收益承诺。旧static只做同参数对照/配置消融，本轮不修改其计算体。无收益候选不切默认。

## 8. 计算节点命令：现在即可运行

以下CLI已按当前源码核对，**本地未执行**。默认CUDA baseline启用，没有 `--no-vendor`。`--synthetic` 是现有通用小样本；精确 ones_fanin 诊断见9.3。

已获得Slurm计算节点时，在仓库根目录执行bash；不要在登录节点用bash运行GPU测试。登录节点可先用已有方式 `srun -n 1 -p gpu_a800 --gpus=1 --pty /bin/bash`，再cd回仓库。

### 8.1 环境与输出目录

```bash
cd ~/run/zyq/flagsparse-zsparse-2  # 改为当前实际仓库
REPO="$PWD"
MATRIX_DIR="$HOME/run/matrix"
RESULT_ROOT="$REPO/$(date +%m.%d)data/spmv_next_baseline"
test -n "${SLURM_JOB_ID:-}" || { echo '请先进入已申请的计算节点'; exit 1; }
test -f "$REPO/run_flagsparse_pytest.py" || exit 1
test -d "$MATRIX_DIR" || { echo "矩阵目录不存在: $MATRIX_DIR"; exit 1; }
mkdir -p "$RESULT_ROOT"
```

### 8.2 baseline通用小样本：4组合、non、N32

```bash
bash tools/run_cuda_tests.sbatch --repo "$REPO" \
  --entry tests/test_spmm_csc.py -- \
  --synthetic --alg csc_col_tile_panel_atomic \
  --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 \
  --ops non --op-b non --layout row --dense-cols 32 \
  --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 3 --iters 10 --timing --perf-metrics --vendor-timeout 300 \
  --csv-csc "$RESULT_ROOT/baseline_synthetic.csv"
```

故意只选一个现有FlagSparse算法，避免baseline排查重复跑全算法；不会追加q4。无需fail-fast，以便保留全部vendor状态。

### 8.3 四个真实矩阵：直接cuSPARSE对照

```bash
for m in mip1 ecology1 c8_mat11 TSOPF_FS_b300_c1; do
  test -f "$MATRIX_DIR/$m.mtx" || { echo "缺少矩阵: $m"; exit 1; }
done
bash tools/run_cuda_tests.sbatch --repo "$REPO" \
  --entry tests/test_spmm_csc.py -- \
  "$MATRIX_DIR/mip1.mtx" "$MATRIX_DIR/ecology1.mtx" \
  "$MATRIX_DIR/c8_mat11.mtx" "$MATRIX_DIR/TSOPF_FS_b300_c1.mtx" \
  --alg csc_col_tile_panel_atomic \
  --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 \
  --ops non --op-b non --layout row --dense-cols 32 \
  --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 5 --iters 20 --timing --perf-metrics --vendor-timeout 300 \
  --csv-csc "$RESULT_ROOT/baseline_real.csv"
```

保留CSV旁的`.vendor_logs`、progress与stdout/stderr。单个数百ms厂商调用×warmup/iters需要数秒到十余秒；不要只凭短暂无打印判断挂死。超时仍按ERROR记录，不当作“不支持”。

### 8.4 准确性回归：包含已接入新算法与独立 vendor 检查

```bash
bash tools/run_cuda_tests.sbatch --repo "$REPO" -- \
  --ops spmv_csr,spmv_csc,spmm_csc --phase accuracy --mode quick \
  --results-dir "$RESULT_ROOT/accuracy_quick"
```

quick通过后把mode改normal、结果目录改`accuracy_normal`。新候选由原 marker 收集，不新增 runner。SpMM CSC 中的 `native_vendor_fanin` 是独立厂商准确性用例，厂商数值错误会 FAIL，不能当作 FlagSparse 内核失败；报告中用例名及 properties 会记录厂商结果。若先只验新算法，--ops 选 spmv_csr,spmv_csc。外部 CSR 矩阵 pytest 依赖 `FLAGSPARSE_SPMV_CSR_MTX_DIR`，wrapper 已显式透传该变量；仅传 benchmark-input 不会启用外部矩阵 pytest。

## 9. 新算法实现后的原入口验收命令

新候选已由 compare 按声明的合法操作枚举。先通过8.4的小样本，再选一个真实矩阵检查；通过后执行下面的完整矩阵集扫描。

### 9.1 regular：CSR non与CSC trans/conj

```bash
bash tools/run_cuda_tests.sbatch --repo "$REPO" -- \
  --ops spmv_csr,spmv_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --dtypes float32,float64,complex64,complex128 --index-dtypes int32 --indptr-dtypes int32 --timing" \
  --op-benchmark-args "spmv_csr=--ops non" \
  --op-benchmark-args "spmv_csc=--ops trans,conj" \
  --results-dir "$RESULT_ROOT/regular_candidates"
```

### 9.2 mixed：四组合，保留吞吐与逻辑带宽

```bash
bash tools/run_cuda_tests.sbatch --repo "$REPO" -- \
  --ops spmv_csr,spmv_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics" \
  --op-benchmark-args "spmv_csr=--ops non" \
  --op-benchmark-args "spmv_csc=--ops trans,conj" \
  --results-dir "$RESULT_ROOT/mixed_candidates"
```

regular入口当前不普遍支持perf-metrics，故不错误添加该参数。后续normal补两种指针/索引及其他合法布局；不改变当前benchmark默认索引含义。

### 9.3 精确累加诊断：四组合、厂商算法及预处理对照

```bash
# 原入口的新增参数已接通；此命令保留 cuSPARSE baseline。
bash tools/run_cuda_tests.sbatch --repo "$REPO" \
  --entry tests/test_spmm_csc.py -- \
  --vendor-diagnostic ones_fanin,rectangular \
  --diagnostic-lengths 128,256,512,1024,2048,4096 \
  --vendor-alg default,csr_alg1,csr_alg2,csr_alg3 --vendor-preprocess off,on \
  --alg csc_col_tile_panel_atomic \
  --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 \
  --ops non --op-b non --layout row --dense-cols 1,32 \
  --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 3 --iters 10 --timing --perf-metrics --vendor-timeout 300 \
  --csv-csc "$RESULT_ROOT/vendor_exact_diagnostic.csv"
```

诊断模式在普通输入校验前提供确定性样本；支持N列表，厂商算法/预处理组合逐条写CSV。default/on、csr_alg2/on 明确报告该预处理选择不适用；不偷偷按off执行。ALG3/CSC及低精度支持取决于目标库版本，原生API拒绝会保留status/reason；未知invalid-value不能泛化成“不支持”。

登录节点提交时，可将命令首部换为 `sbatch --partition=gpu_a800 --ntasks=1 --gpus=1 tools/run_cuda_tests.sbatch --repo "$REPO" ...`；若站点提交策略拒绝，沿用已验证的srun申请方式，不在登录节点改用bash执行。

## 10. 实施记录与静态验收

源码已接通三个新名称：`csr_row_scalar_bound`、`csr_longrow_queue_atomic`、`csc_col_scalar_bound`。六种 dtype、四种半精度输入输出组合、合法操作及 strided out 通过原 API/prepare/run/compare 路由；没有新建 Python 文件或顶层算法 API。公开导出原有查询/run 函数无需增删；支持表新增48条 UNVERIFIED 声明，原准确性 marker 与 GPU CI 扫描已联动。

实现细节相对设计的明确调整：

- 队列job实际存储 `(slot,start,end)` 三个 int64 字段，额外保存end以免消费者再次依赖owner和CSR行指针读取；描述符按24字节/job计入上限。
- long-slot累加缓冲按容量在初始化阶段一次清零，生产者不再逐slot清零。这样每个slot仍只初始化一次，长/短输出互斥，初始化时间纳入完整run及process诊断。
- `job_capacity=0` 在long_capacity=0时成立，无消费者/finish launch；即使容量非零而实际没有长行，也只在设备侧检查计数，不读取CPU。
- 用户请求的 `workspace_limit_bytes` 是描述符、数值缓冲与12字节计数器的合计上限，不只是数值空间。元数据容量为上界，不冒充有效任务数。
- 首轮L/S为正二次幂，S必须为consumer block_nnz的倍数；R=4/8/16，workers=32/64/128/256。循环和地址保持int64，三种计数器为已验证容量内的int32 atomic。
- 修复既有 `csr_subgroup_pair_mixed` 的strided out校验遗漏，未改变其计算体。

baseline记录修复：有效正耗时可计算速度比，native/vendor PASS/FAIL分别保留；新增最大相对误差、按既有容差归一化最大误差、失败元素数及非有限计数。参考仍基于已量化输入的CPU FP64结果，转换到请求输出dtype；共享容差不变。relative分母为max(abs(ref),atol,tiny)，normalized分母为max(atol+rtol*abs(ref),tiny)。

诊断子进程记录default/ALG1/2/3、实际preprocess、库版本、workspace、准备墙钟时间及稳态事件时间；zero/NaN/repeat三种输出初态保存至CSV旁 `.vendor_logs/*.diagnostic.pt`。宽输出再窄化仅是准确性对照，不计为原生窄输出 baseline。诊断保存量化A/B、golden和输出，可直接供独立复现使用；实际加载库路径/hash写入CSV。库身份无法读取时记录原因，不替换数值结果。

CUDA显式枚举核对依据：[NVIDIA HPC SDK 25.9 CUDA Interfaces](https://docs.nvidia.com/hpc-sdk/archive/25.9/pdf/hpc259cudaint.pdf)，CSR ALG1/ALG2/ALG3为4/6/12；HIP仍只使用自身默认枚举和原preprocess规则，不沿用CUDA专用编号。

厂商扫描缓存键显式包含输入身份、类型、形状、操作、stride、索引类型、backend、vendor_alg和preprocess；每个输入条件单独缓存结果。多vendor行复用一次native测量，`flagsparse_measurement_reused`标识该事实。诊断数据必须按vendor_alg_requested/vendor_preprocess_requested分组，不把不同厂商算法混为一个基准GM；runner细节键已包含这两个字段。

静态验收：22个修改Python文件全部通过AST解析，git diff --check通过，sbatch脚本保持LF换行；46个实施前已有JIT函数AST摘要完全不变，新增5个JIT函数（三个算法，其中queue含生产/消费/写回三个kernel）。本地未执行pytest、算子、功能导入或Triton编译，没有暂存或提交。默认选择、旧缓存/图绑定及C API JIT签名保持原样。

尚待计算节点：CUDA/ROCm真实编译、准确性、30矩阵收益与profiler；若诊断样本确认厂商异常，后续第5.3节独立C/C++复现及跨库版本对照仍需实施和验证。本次没有声称已修复cuSPARSE内部数值行为，也没有提供不存在的C++可执行命令。

## 11. 补充诊断及消融命令

### 11.1 矩形样本的操作/布局检查

```bash
bash tools/run_cuda_tests.sbatch --repo "$REPO" \
  --entry tests/test_spmm_csc.py -- \
  --vendor-diagnostic rectangular --alg spmm_csc_base \
  --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 \
  --vendor-alg default --vendor-preprocess off \
  --ops all --op-b all --layout all --dense-cols 1,32 \
  --index-dtypes int32 --indptr-dtypes int32 \
  --warmup 3 --iters 10 --timing --perf-metrics \
  --csv-csc "$RESULT_ROOT/vendor_rectangular_ops.csv"
```

### 11.2 配置消融：单次只改变R、L或S

在小样本通过后使用相同矩阵集；宽输出优先，后续重复同类型输出与regular类型。

```bash
mkdir -p "$RESULT_ROOT/ablation"
COMBOS=float16:float16:float32,bfloat16:bfloat16:float32
for r in 32 64 128; do
  for alg in csr_row_scalar_strip csr_row_scalar_bound; do
    bash tools/run_cuda_tests.sbatch --repo "$REPO" --entry tests/test_spmv_csr.py -- \
      "$MATRIX_DIR" --alg "$alg" --config "{\"rows_per_program\":$r}" \
      --type-combos "$COMBOS" --ops non --index-dtypes int32 --indptr-dtypes int32 \
      --warmup 10 --iters 50 --timing --perf-metrics \
      --csv-csr "$RESULT_ROOT/ablation/${alg}_R${r}.csv"
  done
  bash tools/run_cuda_tests.sbatch --repo "$REPO" --entry tests/test_spmv_csc.py -- \
    "$MATRIX_DIR" --alg csc_col_scalar_bound --config "{\"columns_per_program\":$r}" \
    --type-combos "$COMBOS" --ops trans,conj --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csc "$RESULT_ROOT/ablation/csc_scalar_R${r}.csv"
done
# 固定S=1024比较L；再固定L=1024比较S。另保留compare中旧最佳作对照。
for config in '{"long_row_threshold":1024,"segment_nnz":1024}' \
              '{"long_row_threshold":4096,"segment_nnz":1024}' \
              '{"long_row_threshold":1024,"segment_nnz":512}' \
              '{"long_row_threshold":1024,"segment_nnz":2048}'; do
  tag=$(printf '%s' "$config" | tr -cd '0-9,')
  bash tools/run_cuda_tests.sbatch --repo "$REPO" --entry tests/test_spmv_csr.py -- \
    "$MATRIX_DIR" --alg csr_longrow_queue_atomic --config "$config" \
    --type-combos "$COMBOS" --ops non --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csr "$RESULT_ROOT/ablation/queue_${tag}.csv"
done
```

DCU在既有环境中使用同一脚本/runner及参数，例如把wrapper首部替换为 `python tests/test_spmv_csr.py` 或 `python run_flagsparse_pytest.py`；不传CUDA专用vendor算法。当前HIP低精度vendor不支持时只在该平台显式加 `--no-vendor`，不能把它复制到CUDA对照命令。

### 11.3 外部矩阵准确性

```bash
export FLAGSPARSE_SPMV_CSR_MTX_DIR="$MATRIX_DIR"
bash tools/run_cuda_tests.sbatch --repo "$REPO" -- \
  --ops spmv_csr --phase accuracy --mode normal \
  --pytest-args "-k external_matrix_regressions" \
  --results-dir "$RESULT_ROOT/external_accuracy"
unset FLAGSPARSE_SPMV_CSR_MTX_DIR
```

wrapper现在显式透传该变量。默认矩阵目录仍需可在容器内访问；外部矩阵准确性保留原回归清单，不能把它等同于全部30矩阵的mixed扫描。全部mixed真实矩阵由9.2完成，并保留CPU参考准确性记录。
