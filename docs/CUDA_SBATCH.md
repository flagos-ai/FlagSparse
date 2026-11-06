# CUDA 容器测试作业

统一使用 `tools/run_cuda_tests.sbatch`，无需每个算子复制环境脚本。
在 **库根目录** 提交，Slurm 分配计算节点后依次执行：

1. 登录 Bash 初始化模块环境，`module load apptainer`。
2. `apptainer exec --nv --cleanenv`（无该命令时使用 singularity），绑定 `/data`、库目录及 venv。
3. 容器内激活 `/data/home/scwb320/run/gcx/vpy/bin/activate`。
4. 设置 `TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real`；设置 PYTHONPATH 优先导入本库源码。
5. 使用激活环境的 Python 运行原测试入口，不安装依赖、不改变测试计时或默认算法。

默认镜像是 `$HOME/run/images/flagtree.sif`。无需先执行交互式 `env1.sh`，也无需在登录节点激活 venv。
venv 应已基于镜像 Python 创建，且已具备测试依赖；脚本不创建或修复 venv。
venv 校验比较目录身份，允许 `/data/home/...` 和 `/data/run01/...` 等软链接别名；
不要求激活脚本内保存的路径字符串与物理路径完全相同。
Slurm 的 `CUDA_VISIBLE_DEVICES` 显式传入 cleanenv 容器，不固定为 GPU 0。
包装脚本同时设置专用分配标记，runner 的 `--gpus` 此时表示**分配列表中的序号**：
默认 0 使用分配的第一张卡；申请两张卡后 `--gpus 0,1` 使用两张卡。
子进程保留对应物理编号或 UUID，在其单卡视图中使用 device 0。
未使用该包装脚本时，runner 原有 GPU 编号规则保持不变。

## 提交方式

```text
sbatch [调度资源参数] tools/run_cuda_tests.sbatch [环境/入口参数] -- [原测试参数]
```

默认资源为单节点、单任务、1 GPU、8 CPU、2 小时。分区、账户、内存未写死；
按集群要求在**脚本名之前**添加 `--partition=实际分区`、`--account=实际账户`、
`--mem=...`、`--time=...` 等。集群若使用其他 GPU 申请约定，应按管理员要求调整 `--gres`。
同一作业内算子顺序测试；性能比较不要在同一张 GPU 上并行启动多个作业。

### 六个算子的 quick 准确性

```bash
sbatch tools/run_cuda_tests.sbatch -- \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase accuracy --mode quick
```

只测试一个算子改为 `--ops spmv_csr`；更完整的准确性测试改为 `--mode normal`。
这是原 pytest 用例集合，包含已接入的 mixed 用例，不是仅测 mixed。

## 真实矩阵集：普通类型与 mixed 分开提交

先区分两种测试任务：

| 目的 | runner 参数 | 实际数据与输出 |
|---|---|---|
| 边界、接口、各算法准确性回归 | `--phase accuracy` | 原 pytest 用例；quick/normal 控制覆盖规模 |
| 真实 `.mtx` 矩阵准确性 **及** 性能 | `--phase performance --benchmark-input 路径` | 原 benchmark 逐矩阵计算 reference、检查结果并计时 |
| 两者都跑 | `--phase both --benchmark-input 路径` | 先 pytest，再矩阵 benchmark；两份结果分别查看 |

**测试真实矩阵不要只传 `--phase accuracy`。** `--benchmark-input` 是性能阶段的输入，
不是为所有 pytest 文件设置外部矩阵集。性能阶段也做正确性检查，不是只计时。
以下命令不使用 `--synthetic`，输入为指定目录中的 `.mtx`；`quick` 不表示只取一个真实矩阵。
请用独立目录放置本轮的 30 个矩阵，并在结果中核对覆盖数量及失败/缺失原因。

下文“普通/非 mixed”指使用 `--dtypes` 的已有同类型输入输出测试分支，
不是承诺每个算法的内部计算类型都与输入相同；原有 FP32→FP64 累加等策略保持不变。
四种半精度组合则使用 `--type-combos` 明确输入和输出，**不能同时传 `--dtypes`**。

### A. 普通 FP32/FP64：六算子、真实矩阵、全部适用非 TLE 算法

在库根目录执行，先将 `MATRIX_DIR` 改成真实路径。此处默认 `non`、int32、
SpMM row 布局和 N=32；厂商 baseline 启用。

```bash
MATRIX_DIR=/data/你的矩阵目录
sbatch --time=08:00:00 tools/run_cuda_tests.sbatch -- \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
  --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--dtypes float32,float64 --alg compare --ops non --index-dtypes int32 --timing" \
  --op-benchmark-args "spmm_csr=--layout row --exclude-tle" \
  --op-benchmark-args "spmm_coo=--layout row" \
  --op-benchmark-args "spmm_csc=--layout row"
```

这会分别进入六个原格式测试脚本；CSR SpMM 因显式 `--alg compare` 进入
`tests/test_spmm_csr.py`，而不是默认交付的 `tests/test_spmm.py`。
`compare` 枚举该 dtype/op/后端实际支持的算法，不重复运行 auto。
新 NNZ 均衡算法只声明半精度组合，因此**不会出现在这组 FP32/FP64 扫描中**。
本例排除 CSR SpMM TLE；要扫描包含 TLE 的原算法集合，去掉 `--exclude-tle`，
是否可运行仍取决于集群中 TLE 环境与算法自身的支持范围。

扩展普通测试：

- 复数也测：将 `float32,float64` 改为 `float32,float64,complex64,complex128`。
- A 的转置/共轭转置也测：将 `--ops non` 改为 `--ops all`。
- SpMM 两种存储布局：将三项 `--layout row` 都改为 `--layout all`。
- 两种索引：将 `--index-dtypes int32` 改为 `--index-dtypes int32,int64`。
- SpMM N 扫描：普通分支目前只接受单个 N，分别提交 `--dense-cols 8`、
  `--dense-cols 32`、`--dense-cols 128` 等作业；该参数加在三个 SpMM 的
  `--op-benchmark-args` 中，不要传给 SpMV。

普通分支当前不接受独立 `--op-b trans/conj` 扫描，也不能直接复制 mixed 分支的
N 逗号列表。普通算法是否支持某项能力与当前 benchmark 是否提供该 CLI 是两回事。

### B. 四种 mixed：六算子、真实矩阵、全部适用算法

| `--type-combos` 条目 | A 和 x/B | 计算/部分和 | 输出 |
|---|---|---|---|
| `float16:float16:float16` | FP16 | FP32 | FP16 |
| `float16:float16:float32` | FP16 | FP32 | FP32 |
| `bfloat16:bfloat16:bfloat16` | BF16 | FP32 | BF16 |
| `bfloat16:bfloat16:float32` | BF16 | FP32 | FP32 |

下面不添加 dtype 参数；SpMM 默认 N=32。mixed 注册扫描包含适用的新 NNZ 均衡算法，
本轮组合不包含 TLE。显式选择分支不会额外追加旧 q4 测试。

```bash
COMBOS=float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32
MATRIX_DIR=/data/你的矩阵目录
sbatch --time=08:00:00 tools/run_cuda_tests.sbatch -- \
  --ops spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc \
  --phase performance --mode quick --benchmark-input "$MATRIX_DIR" \
  --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--type-combos $COMBOS --alg compare --ops non --index-dtypes int32 --indptr-dtypes int32 --timing" \
  --op-benchmark-args "spmm_csr=--op-b non --layout row" \
  --op-benchmark-args "spmm_coo=--op-b non --layout row" \
  --op-benchmark-args "spmm_csc=--op-b non --layout row"
```

矩阵路径相对库根目录，需在容器可见路径中，推荐 `/data/...`。
SpMM 不指定 dense-cols 时使用默认 N=32。CUDA 厂商 baseline 保持启用；如要单独检查
FlagSparse，可在 `--benchmark-args` 中增加 `--no-vendor`。
扩展操作时将 `--ops non` 改为 `--ops all`，各 SpMM 的 op-b 改为 `non,trans,conj`。
资源时限仅为示例，请根据矩阵规模调整。

扩展 mixed 全操作/布局时，保持上述提交命令，把对应参数替换为：

```bash
  --benchmark-args "--type-combos $COMBOS --alg compare --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing" \
  --op-benchmark-args "spmm_csr=--op-b non,trans,conj --layout all --dense-cols 1,8,16,32,64,128,256" \
  --op-benchmark-args "spmm_coo=--op-b non,trans,conj --layout all --dense-cols 1,8,16,32,64,128,256" \
  --op-benchmark-args "spmm_csc=--op-b non,trans,conj --layout all --dense-cols 1,8,16,32,64,128,256"
```

这是**替换片段，不是独立命令**。CSR/CSC 中两项索引参数分别控制列/行索引及指针；
COO 的第二项对应第二个坐标数组。全组合乘积很大，先完成默认 non/N=32 扫描，
再按算子、操作或类型组合拆分提交；每次作业结果目录独立。

### C. 只测一个算子，或只测一个真实矩阵

上面两类命令都只需修改：

- `--ops spmv_csr`：仅测 SpMV CSR，同时删除三个 SpMM 专属的
  `--op-benchmark-args`。runner 会拒绝给未选中算子传参数。
- `--ops spmm_csc`：仅测 SpMM CSC，只保留它自己的 `--op-benchmark-args`。
- `--benchmark-input /data/矩阵目录/2cubes_sphere.mtx`：先单矩阵检查链路；
  确认后换回矩阵目录，不要用单矩阵结果代表整套矩阵性能。
- `--phase both --mode normal`：同一作业内额外运行完整 pytest 回归；
  dtype/type-combos 参数是 benchmark 透传，不会把 pytest 同步限制成这些类型。

例如，只检查一个真实矩阵上的 SpMV CSR 四组合：

```bash
sbatch tools/run_cuda_tests.sbatch -- \
  --ops spmv_csr --phase performance --mode quick \
  --benchmark-input /data/矩阵目录/2cubes_sphere.mtx \
  --benchmark-warmup 2 --benchmark-iters 5 \
  --benchmark-args "--type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 --alg compare --ops non --index-dtypes int32 --indptr-dtypes int32 --timing"
```

### D. baseline、准确性与结果怎么读

- 普通分支使用该脚本既有的同格式厂商适配；不要把它统一理解成直接 ctypes C API。
  实际 baseline 身份、可用性及原因以脚本输出/CSV 为准。
- 显式 mixed 分支接入直接 cuSPARSE Generic C API（此处是 CUDA 作业），
  baseline 在隔离子进程中测试。库不可用或组合不支持时记录原因，不据此跳过
  FlagSparse 算法；运行故障单独记录。脚本接通不等于所有 CUDA 版本/格式/操作均支持。
- 只测试 FlagSparse：在 `--benchmark-args` 追加 `--no-vendor`，仍检查准确性并记录自身耗时，
  此时没有可计算的厂商加速比。检查 baseline 时不要带该开关。
- mixed 正确性 reference 使用**先量化为目标输入类型的数据**，在 CPU SciPy/FP64 上计算，
  再按目标输出类型比较，沿用共享容差；不是用 cuSPARSE 的输出充当唯一正确答案。
  普通分支沿用各脚本已有 reference，具体身份以输出为准。
- 真实矩阵转 FP16 时可能超出表示范围；mixed CSV 会记录输入、reference、输出的非有限值。
  这类 FAIL 要与有限值计算误差区分，不通过缩放原矩阵或放宽容差掩盖。
- 查看每算子的原始 `performance.csv` 及日志，核对矩阵名、算法、类型/四元组、操作、
  layout/N、索引、正确性状态、耗时、vendor 状态和原因。mixed 有效加速比为
  `vendor_ms / ms`，FlagSparse FAIL 与有效耗时分别保留。
- `summary.csv` 等交付汇总有自身筛选口径，不能替代 compare 扫描明细，尤其不能用它
  推断“所有算法、30 矩阵都通过”。分析算法最优表现时固定测试条件逐矩阵选最快，
  保留准确性状态，并报告覆盖数。
- 计时仍走原 main 入口；显式 mixed 的完整测量与 `--timing` 诊断分开，
  `ms=process_cpu_ms+gpu_ms`，分段诊断不替换加速比分母。

### 直接测试单算法及配置

```bash
sbatch tools/run_cuda_tests.sbatch --entry tests/test_spmv_csr.py -- \
  --synthetic --type-combos float16:float16:float32 \
  --alg csr_nnz_tile_reduce --ops non \
  --config '{"block_nnz":128,"reduce_block_size":256}' \
  --warmup 10 --iters 50 --timing --csv-csr spmv_nnz_smoke.csv
```

直接 benchmark 模式仍需自行指定唯一 CSV 路径，避免多个作业覆盖同名 CSV。
测试参数逐个传递，不使用 eval；含空格的 `--benchmark-args` 和 JSON 保持引号。

## 路径、日志与退出状态

环境路径可通过脚本参数覆盖，例如：

```bash
sbatch /data/path/FlagSparse-main/tools/run_cuda_tests.sbatch \
  --repo /data/path/FlagSparse-main \
  --image /data/path/flagtree.sif \
  --venv /data/path/vpy \
  --libcuda /usr/local/cuda/compat/lib.real -- \
  --ops spmm_csr --phase accuracy --mode quick
```

- 库目录选择顺序：`--repo`、`FLAGSPARSE_REPO`、当前目录（含 runner 时）、`SLURM_SUBMIT_DIR`。交互式 job 中先 cd 进库再用 bash 启动即可；旧脚本可显式传 `--repo "$PWD"`，避免申请 job 时的提交目录覆盖当前目录。
- 从其他目录提交且该目录不含库时，应设置 `--repo`（或 `FLAGSPARSE_REPO`），因为 Slurm 会复制脚本，无法从脚本自身位置推断源库。
- Slurm 输出：提交目录下 `flagsparse-JOBID.out/.err`，可用 sbatch `--output/--error` 覆盖。
- 作业日志与可复现命令：库内 `slurm_results/JOBID/launch.txt`、`test.log`。
- 默认 runner 结果：`slurm_results/JOBID/results/`；显式 `--results-dir` 优先。
- Triton 缓存：`slurm_results/JOBID/triton_cache/`，避免不同作业争用；JIT/warmup 仍由原入口处理。
- Python 的非零退出状态通过管道传给 Slurm；作业成功不代表所有精度结果 PASS，仍需看 JSON/CSV。
- 不要在该目录测试时清理对应作业缓存；作业结束后按需要归档结果并清理缓存。

本地仅进行脚本静态语法检查；容器、GPU、venv 和调度资源是否匹配需在集群验证。
