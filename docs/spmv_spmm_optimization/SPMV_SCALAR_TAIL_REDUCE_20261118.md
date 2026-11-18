# SpMV 标量前缀与并行尾部：独立候选及验收

## 1. 本轮范围与依据

目标：`FlagSparse-main/zyq_sparse`。起点commit：`011c2c4c1bc6003471f02a4a602c2cc0d40f39f9`，开始实施时工作区干净。只推进SpMV，不修改SpMM计算、厂商baseline、计时框架或默认路由。不暂存提交。

依据为11.17 candidate数据分析：

- CSR scalar-bound对短行有稳定方向的收益，但FP16→FP32在wiki-Talk比同批旧最佳慢26.2倍，在mip1慢16.7倍。
- CSC scalar-bound在roadNet/ecology1 trans分别约3.09/2.42倍，但在mip1长列慢12.39倍。
- 长行queue只有mip1获胜；普通行producer及多次launch的固定成本尚未通过消融定位，暂不改写queue。

先保留成功的短行组织，尝试消除其长行串行尾部。新名称均为独立实现，原kernel不修改：

| 算子 | 新算法 | 合法op | 主要对照 |
|---|---|---|---|
| SpMV CSR | `csr_row_scalar_tail_reduce` | non | scalar-bound、scalar-strip、static-tile、row-vector、queue |
| SpMV CSC | `csc_col_scalar_tail_reduce` | trans/conj | scalar-bound、subgroup、vector、split |

## 2. 计算方式与正确性

每个program独占R个连续行（CSR）或列（CSC）。令对象长度为l，标量步数为S，并行尾部宽度为K。

1. 一次计算本组最长长度lmax。
2. 标量阶段处理位置`0 <= j < min(l,S)`；每个对象维护一个计算精度累加器。
3. 当`lmax>S`时，尾部lane v处理`j=S+v+t*K`，其中`0<=v<K`，且`j<l`；使用[R,K]计算精度累加器。
4. 沿K归约尾部，与标量前缀相加；按out真实stride直接写回。

索引集合是互斥完备的：每个`j<S`只属于标量阶段；每个`j>=S`唯一对应`v=(j-S)%K`、`t=(j-S)//K`。空对象写零。尾行/列、尾NNZ分别mask；地址与行指针运算采用int64。每个输出只有一个program负责，因此不需要浮点atomic、清零kernel、全局队列或部分和工作空间。

CSC直接读取原CSC，不转CSR；conj只共轭A。实虚分量分别累加。CSR只声明non，不能通过compare把该名称用作转置算法。

支持FP16、BF16、FP32、FP64、complex64、complex128，A与x同类型。FP16/BF16加载后FP32计算，支持同类型或FP32输出；宽输出不经过半精度缓冲。CSR FP32仍按原新CSR约定FP64计算，CSC FP32仍FP32计算；复数采用对应分量精度。不是任意mixed组合。

## 3. 参数与理论成本

默认：R=64，S=8，K=32，2warps、1stage。内置warp经原后端能力筛选；显式非法值报错，未知能力不宣称支持。参数尚未调优。

| 字段 | CSR | CSC | 合法候选 |
|---|---|---|---|
| R | rows_per_program | columns_per_program | 32/64/128 |
| S | scalar_steps | scalar_steps | 0/4/8/16/32 |
| K | block_nnz | block_nnz | 8/16/32/64 |

`S=0`是纯并行归约的内部消融，仍走本新kernel，不能称作复用了旧static-tile。原bound不接受scalar_steps，配置不能跨算法继承。

旧bound的组循环约为lmax次。新候选约为`min(lmax,S)+ceil(max(lmax-S,0)/K)`次迭代，加一次尾部归约；这只说明依赖链长度变化，不是理论加速比。长尾阶段增加R×K累加器、布局转换、masked lane工作和一次归约。即使运行时短组未走尾部分支，编译器为整个kernel安排的寄存器/布局仍可能拖慢短路径。

因此本轮目标是：在保留原bound作对照的情况下，给短长混排及中长行增加一个低全局调度成本的候选。它没有跨program拆分单个超长行，不能代替queue/split；若长尾较少，R个对象中大部分lane仍可能空闲。最终判断依赖完整run，不以循环次数缩减宣称优化成功。

## 4. 执行与计时契约

- 原高层API、prepare/run、注册查询、compare及统一runner接入，不增加顶层算法函数或Python文件。
- 原始输入别名、strided out、op冲突、类型冲突及同算法索引回退沿用原检查。
- 每次调用一个计算kernel，无跨调用缓存，无额外数值工作空间（输出本身不计为临时workspace）。CSC原有resolve/必要输入处理仍按其原run边界计时。
- 始终`ms=process_cpu_ms+gpu_ms`。完整采样与分段诊断独立；局部前缀、尾部归约均属于compute。
- 保留cuSPARSE baseline、精度FAIL有效耗时、吞吐和逻辑带宽字段。没有修改任何baseline实现；之前SpMM诊断问题不在本轮范围内。
- 元数据包含实际alg/config、scalar_steps、tail_block_nnz、plan_strategy=none、workspace_peak_bytes=0及unverified状态；不为计数额外同步。

## 5. 准确性与静态审查

扩展原CSR/CSC accuracy文件，沿用marker、quick/normal、shared golden/容差，包含：

- 六种dtype、四种半精度组合及全部声明op；CSC非对称复数trans/conj。
- S前后、K及S+K边界、空组、组尾、单长对象、短长混排、重复项和未排序索引。
- S=0/4/8/32，R32/64/128及K8/16/32/64的代表性组合；normal加入8193长度和四种索引组合。
- NaN预填、输入/输出stride=2、返回out身份、原始输入不变、宽输出尾部精度、非法参数与安全索引回退。
- no-plan检查和纯策略测试；异常不按泛化unsupported吞掉，不放宽容差。

当前仅AST、源码和diff检查；未运行pytest、功能导入、Triton编译或GPU算子。原两个实现文件中41个既有JIT函数的AST保持不变，新增两个kernel与一个CSC局部乘积helper。测试通过及性能收益必须由计算节点单独确认。

## 6. CUDA计算节点命令

以下在已分配计算节点、仓库根目录运行；不加`--no-vendor`。先确认真实矩阵路径，避免空变量。

```bash
MATRIX_DIR="$HOME/run/matrix"
test -d "$MATRIX_DIR" || { echo "Matrix directory missing: $MATRIX_DIR"; exit 1; }
RESULT_ROOT="$PWD/11.18data/spmv_scalar_tail"
mkdir -p "$RESULT_ROOT"

# 先编译/准确性smoke，四种半精度组合与regular均由原参数化覆盖。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmv_csc --phase accuracy --mode quick \
  --pytest-args "-k scalar_tail -x" \
  --results-dir "$RESULT_ROOT/smoke"
```

确认smoke通过后再做normal准确性（沿用同命令，`--mode normal`、结果目录`accuracy_normal`）。不要仅以wrapper退出码判断全部通过，要看accuracy结果及失败日志。

以下两批使用compare，保留旧算法和新候选，同时收集旧路径其他op作为回归；CSC non不包含新候选，CSR trans/conj也不包含新候选。regular CSR转置baseline受原旧入口限制时会N/A，不伪造可用性。

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmv_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --dtypes float32,float64,complex64,complex128 --ops all --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/regular"

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr,spmv_csc --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--alg compare --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 --ops all --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics" \
  --results-dir "$RESULT_ROOT/mixed"
```

### 第一轮消融：只改变S

固定R64/K32，先FP16/BF16宽输出。专用config只配显式单算法。S0是必测控制；完成后再单独比较K、R，不做三参数全笛卡尔积。

```bash
for s in 0 4 8 16 32; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
    "$MATRIX_DIR" --alg csr_row_scalar_tail_reduce --config "{\"scalar_steps\":$s}" \
    --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --ops non --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csr "$RESULT_ROOT/csr_s${s}.csv"
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csc.py -- \
    "$MATRIX_DIR" --alg csc_col_scalar_tail_reduce --config "{\"scalar_steps\":$s}" \
    --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --ops trans,conj --index-dtypes int32 --indptr-dtypes int32 \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csc "$RESULT_ROOT/csc_s${s}.csv"
done
```

时间有限时先把MATRIX_DIR设为只含roadNet、ecology1、ASIC、NACA、engine、Stanford、TSOPF、mip1、wiki-Talk的子集目录。保留完整矩阵原值，后续再跑全部30矩阵；不得把子集GM称全体GM。正式消融至少3轮，交换配置执行顺序，单独目录保存，不拼接跨轮最小值。

DCU可使用上面原runner参数直接调用`python run_flagsparse_pytest.py`，不使用CUDA容器脚本；仅在baseline明确不可用时加`--no-vendor`，保留自身测量。CUDA与DCU结果分开统计。

## 7. 验收及后续决定

先处理真实矩阵数值FAIL，特别是此前engine宽输出和CSC FP32长归约；保留失败性能但不称其可部署通过。根据实际错误检查索引/掩码、累加顺序与消去，不预设“容差不合理”。

性能固定同类型/op/索引逐矩阵比较：旧算法族最佳、新候选单独、加入候选后的整体最佳；另报胜出>5%、全部GM/AM和回退矩阵。模型逻辑带宽不是物理显存利用率。

重点问三个问题：短行是否因尾部布局成本回退；长尾是否击败旧vector而非仅击败弱scalar；S>0是否胜过S=0。若S0始终最好，则没有证据保留前缀策略为推荐；若只改善相对scalar而没有任何新赢家，则停止扩大这个候选。新候选无论成败均不自动切换默认。

queue保留现状，另用已有L/S/worker配置消融；本轮没有修改其producer、consumer或计时。SpMM和COO不增加新算法。
