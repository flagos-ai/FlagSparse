# SpMV 定向优化设计：通用计算、局部补位与转置分片

状态：源码接入、硬件未验证，2026-11-14。目标分支 `zyq_sparse`，实现基线
`628a3c2eaaa21645f5a50d3a201db9c27623289f`。两个独立kernel路线、规格、配置、
原benchmark/pytest及支持表已接入。仅做源码/AST/diff检查；未执行算子、pytest
或Triton编译，不暂存或提交。

数据依据：`test_mixed_data/11.12data/analysis/SpMV_瓶颈与定向优化设计_20261114.md`
及该目录逐矩阵 CSV。性能结论保留精度状态；没有 profiler 的原因只列为假设。

## 1. 修正上一轮出发点

“mixed”是类型契约，不是性能机制。FP16/BF16 输入已经由通用算法加载后提升到
FP32；再增加双累加器、预取块、gather、scan，并不会自动减少索引流量、x 的
随机访问或行间不均衡。新增活跃张量可能反而增加寄存器和布局转换成本。

已有数据也不是所有新 mixed 算法均失败：CSR pair 在四组合 non 的120个最佳
记录中占16个；但 COO prefetch 的整体对照明显落后。因此保留局部胜者，停止
以 dtype 名称作为新算法动机。新候选必须改变可量化的工作分配或竞争成本。

源码重复检查结论：

- `_spmv_csr_transpose._scatter` 已有原生 CSR 转置散射；单纯扩充 dtype/注册入口
  是兼容能力补齐，不作为新数学算法成果。
- `row_split_reduce` 已有分段、独占部分和及多级归约；上次建议的固定 Q 分条
  仍属于 split 简化，不在本轮另起名字重试。
- `row_subgroup_stream_kernel` 一组 R 行全部执行完，再处理下一组。当前没有
  活跃槽位完成一行后立即领取窗口中下一行的调度。
- 不修改现有 NNZ compact/fixed、rowgroup scan、mixed prefetch 的计算体。

本轮只计划两个独立候选，算法名不带 mixed：

| 名称 | 主要操作 | 真正差异 |
|---|---|---|
| `csr_row_refill_reduce` | non；trans/conj 可沿用原重建生命周期 | program 内活跃行槽位动态补位，不等同一批最长行 |
| `csr_transpose_sharded_atomic` | trans/conj | H 个计算精度输出累加区分散跨 program 的热点竞争，最后固定归约 |

不承诺对所有矩阵获益；不扩展 CSC/COO 的同义版本，先用 CSR 验证机制。

## 2. 类型、操作及公共契约

| A/RHS 输入 | 累加与临时缓冲 | 输出 |
|---|---|---|
| FP16/BF16 | FP32 | 同输入或 FP32 |
| FP32 | FP64，保持新 CSR 通用算法既有策略 | FP32 |
| FP64 | FP64 | FP64 |
| complex64/128 | FP32/FP64 实虚分量 | 对应复数类型 |

不增加异类型输入、整数、任意 compute_dtype，不改变原算法精度。FP32 输入的
FP64 累加不能因 atomic 性能较差而静默改为 FP32。新候选对旧算法比较需同时
报告计算精度；旧 legacy 的 FP32 累加成绩不能解释为同精度实现差异。

`conj` 为 A.conj().T @ x，不共轭 x。重复项全部参与计算。沿用原合法索引组合，
地址、指针跨度及调度游标使用 int64；没有 GPU 计数读回。最终 out 支持原契约
允许的 stride，写入身份、原始输入别名检查、同算法索引回退保持完整。

新后端范围不超过已有普通 Triton CSR 路径。复用规范后端身份；refill 需要整数
scan/归约，sharded 需要相应分量的 atomic。现有 BackendCaps 没有明确 atomic
字段，需要增量补齐已知/未知能力，而不是用 fp64=True 代替 FP64 atomic 能力。

## 3. `csr_row_refill_reduce`：活跃行槽位补位

### 3.1 工作分配

每个 program 独占 W 个连续行，维护 R 个活跃槽位，每槽 V 个逻辑 lane。
初始取窗口前 R 行。每轮各槽处理该行最多 V 个 NNZ；完成的槽立即写回，
然后从本 program 窗口的剩余行中领取下一行。不同 program 的行窗口不重叠。

状态均为局部 SSA 张量：row[R]、pos[R]、end[R]、active[R]、acc[R,V]，加一个
scalar next_offset。没有全局队列、全局 owner-map、行排序或预处理计划。

领取方法：令 done[R] 表示本轮完成的活跃槽。计算

    rank = cumsum(done.int32) - 1
    next_row = window_base + next_offset + rank
    assign = done & (next_row < window_end)
    next_offset += sum(done)

仅 assign 为真的槽读取新行指针；done 槽清空累加器，其余槽保留状态。
没有被分配到新行的 done 槽失活。空行在一个空贡献轮中写零，然后继续领取。

### 3.2 Triton 实数核心草案

以下为一个完整的实数 JIT 计算体草案，用于说明类型、形状和 mask，**不是已编译
通过的实现**。ACC 是 tl.float32/tl.float64 constexpr；complex 实现需另加独立
实虚累加器。R/V 为2的幂；W>=R。首版每轮一个 V 子块，不加入预取或双链。

```python
@triton.jit
def row_refill_real(A, I, P, X, Y, M, XS, YS,
                    R: tl.constexpr, V: tl.constexpr,
                    W: tl.constexpr, ACC: tl.constexpr):
    base = tl.program_id(0).to(tl.int64) * W
    limit = tl.minimum(base + W, M)
    slot = tl.arange(0, R).to(tl.int64)
    lane = tl.arange(0, V).to(tl.int64)
    row = base + slot
    active = row < limit
    pos = tl.load(P + row, active, other=0).to(tl.int64)
    end = tl.load(P + row + 1, active, other=0).to(tl.int64)
    cursor = tl.full((), R, tl.int64)
    acc = tl.zeros((R, V), ACC)
    while tl.sum(active.to(tl.int32), 0) > 0:
        p = pos[:, None] + lane[None, :]
        valid = active[:, None] & (p < end[:, None])
        col = tl.load(I + p, valid, other=0).to(tl.int64)
        a = tl.load(A + p, valid, other=0).to(ACC)
        x = tl.load(X + col * tl.cast(XS, tl.int64), valid, other=0).to(ACC)
        acc = acc + a * x
        pos = tl.minimum(pos + V, end)
        done = active & (pos >= end)
        count = tl.sum(done.to(tl.int32), 0)
        if count > 0:
            result = tl.sum(acc, 1)
            tl.store(Y + row * tl.cast(YS, tl.int64), result, done)
            rank = tl.cumsum(done.to(tl.int32), 0).to(tl.int64) - 1
            candidate = base + cursor + rank
            assign = done & (candidate < limit)
            fresh_pos = tl.load(P + candidate, assign, other=0).to(tl.int64)
            fresh_end = tl.load(P + candidate + 1, assign, other=0).to(tl.int64)
            row = tl.where(done, candidate, row)
            pos = tl.where(done, fresh_pos, pos)
            end = tl.where(done, fresh_end, end)
            acc = tl.where(done[:, None], tl.full((R, V), 0, ACC), acc)
            active = (active & ~done) | assign
            cursor = cursor + count.to(tl.int64)
```

实现时 XS/YS 作为运行时整数传入，与内核已有约定一致。不能把循环状态隐式截断
到 int32；不能用 `where` 包住无 mask 的非法 load。将二维累加器减到一维用
tl.sum(acc,1)，完成槽计数/cumsum只作用于 R 个整数，不对 NNZ 乘积做 scan。
if/while 条件均为 program 统一 scalar，没有假设 CUDA 的独立线程控制流。

复数采用两个 [R,V] 累加器及显式复数乘法；对每一行仍是 lane 分担位置，最终
归约。不可把 strided 复数 out reshape 成新副本后写入；使用分量指针与真实 stride。

### 3.3 正确性与终止性

1. program 的窗口互斥；初始行及 cursor 分配区间互斥。
2. 一次完成的槽 rank 是0..count-1；cursor每次增加count，因此不重复领行。
3. 每槽 pos 按 V 推进，最后 mask；一行所有 NNZ 覆盖一次，不与别槽重复。
4. 只有 done 写回，写回后槽立即换行或失活；空行写零，无需全输出初始化。
5. 每行长度有限，窗口行数有限。每轮活跃非空行推进，空行完成，故最终终止。
6. M=0 不启动 kernel；N=0/nnz=0 的合法输入只执行 mask 为假的加载并写零。

### 3.4 成本模型和适用条件

令 t_i=max(1,ceil(length_i/V))。原 sequential-group 的工作轮数约为

    C_static = sum_groups max(t_i in that group)

补位相当于 R 个槽的窗口内顺序任务调度，其轮数 C_refill 至少为
max(max_i t_i, ceil(sum_i t_i/R))。该式是下界，不是收益预测。

例：R=4，8行的 t 为 [1,1,1,8,8,1,1,1]，原分组16轮，理想补位调度9轮。
但补位增加 done 判断、整数 cumsum、状态 select 和分散时刻的写回；因此实际
加速要求 (C_static-C_refill)*每轮成本 大于这些额外成本。

真正目标是**一个窗口内多组短长交错、每组被长行拖住**。全短行、长度齐整行、
一个孤立超长行均可能没有收益；后者仍受 max_i t_i 限制。这不是为所有差矩阵
找万能解法。先离线统计行长并模拟上述轮数，仅用于选择实验矩阵，不进入运行时。

起始候选 V=8、W=128、2 warps，R按合法subgroup宽度/warp数/V推导（CUDA通常8，
ROCm通常16）；不是已调优值。显式 R/V/W 均校验，W>=R且为R整数倍；W=R用于
调度开销对照，W=4R/8R/16R用于窗口规模消融。逻辑槽并不保证编译器恰好映射
为物理subwarp，需要计算节点检查 layout 和寄存器。

## 4. `csr_transpose_sharded_atomic`：输出分片散射

### 4.1 与现有 scatter 的差异

单输出 scatter 的热点 y[col] 会被许多 program 同时更新。新候选将计算精度
输出分为 H 份 Z[H,N]，每个源行组 program 使用 h=pid % H 的独占选择：

    Z[h, col] += op_value(A[p]) * x[source_row]
    y[col] = sum_h Z[h,col]

这里“独占选择”不表示 shard 只有一个 program，shard 内仍然使用 atomic。
不同 shard 的物理地址不相交。H=1是控制组，思想对应旧 scatter；H>1 的竞争
分散和固定输出归约才是新增机制。不能把“免转置重建”和“分片收益”合为一项。

### 4.2 三阶段与核心地址

1. 初始化 Z 为计算类型零。
2. 直接读取 CSR 行组 [R,K]，加载 x[row]；按本组最大行长循环，写 shard atomic。
3. 固定 H 归约并按实际 out stride 写回，最终 store 才允许输出窄化。

实数关键片段（依赖外围已建立的 mask/ptr，不是独立可运行 kernel）：

```python
pid = tl.program_id(0).to(tl.int64)
shard = pid % H
col = tl.load(I + p, mask, other=0).to(tl.int64)
value = tl.load(A + p, mask, other=0).to(ACC)
product = value * x_row[:, None]
offset = shard * tl.cast(N, tl.int64) + col
tl.atomic_add(Z + offset, product, mask, sem="relaxed", scope="gpu")
```

x_row 先提升至 ACC；Z元素类型必须为ACC。复数每分量分别atomic，conj仅对矩阵
虚部取负。kernel之间在同一实际设备流上顺序提交，不能尝试同kernel中先zero
再跨program atomic，不能采用跨CTA自旋等待或假设全局barrier。

归约kernel沿 H 维使用 tl.sum，H选1/2/4/8且为2的幂；沿N维tile以合并读取。
N尾部严格mask。空矩阵跳过scatter，初始化与归约仍产生正确零输出；N=0直接
返回正确空输出。H=1且out恰为计算精度、布局合法时可直接zero/atomic到out，
否则仍使用计算缓冲；元数据说明是否省掉了归约/转换，不能把该成本差归为竞争。

### 4.3 成本、容量与失败模式

H份缓冲占 H*N*C 字节，C=计算分量字节数*实虚分量数。初始化和最终读取带来
约2*H*N*C字节的流量，另有输出写入和atomic读改写；这只是模型，不是实际HBM计数。
默认上限沿用256MiB。显式H超过容量明确报错，不能自动改H、分输出列多次重读
所有A或静默回退。首版默认H=4、R=4、K=32、4warps、1stage，经后端合法性筛选。

收益条件：跨program同地址竞争降低，足以覆盖额外初始化/归约/空间成本。
不承诺atomic数量下降，不承诺竞争精确下降H倍。若热点集中在同一program内、
或者有规则的pid模H冲突，分片帮助有限。多数输出列几乎无竞争、N很大或NNZ很少
时通常不适合H>1。需统计目标列入度与源行组到shard的分布，再选择测试子集。

FP32输入采用FP64 atomic，这可能是A800或DCU的独立瓶颈；不能借半精度结果
宣称所有dtype有效。atomic求和顺序不确定；共享容差必须通过，不声称bitwise确定性。

## 5. 路由与改动边界

只在已有文件增量实现，不新增Python文件，不变默认算法：

- `_spmv_csr_kernels.py`：独立refill JIT，不改原subgroup/row_tile计算体。
- `_spmv_csr_transpose.py`：独立shard scatter和reduce JIT，不改旧scatter/gather。
- `_spmv_csr_config.py`：独立规格、合法参数、计算类型与atomic能力要求；禁止用
  字符串前缀决定路线。alg/config隔离，非法显式配置报错。
- `spmv_csr.py`：执行策略按spec区分。refill trans/conj沿用原重建；sharded在
  原始CSR上执行，不能先重建再scatter。原prepared字段/输出长度/op检查保留。
- 原benchmark、accuracy、策略测试、registry和支持表接入；无新顶层算法函数。

实现必须先解决规格中目前统一ops/transpose_strategy的问题，不能只在算法
名称列表增加条目。sharded不支持non，查询与compare必须明确排除该组合。
后端缺失atomic能力为不可用；编译和执行失败不能伪装为能力不足。

prepared不保存运行工作空间或新跨调用计划。配置解析遵守原生命周期；每次必需
的初始化、分片缓冲、归约及转换在运行范围内。类型/输出/索引回退不能改算法身份。

## 6. 分阶段实施与停止条件

### A. 证据和低成本筛选

在当前日期数据目录追加结构统计和CPU调度模型，不把它当算子测试：对R/V/W模拟
静态组与补位的工作轮数，记录空行、长尾NNZ份额、目标列入度。模型只筛候选，
不能证明GPU收益。若目标矩阵几乎没有可回收的等待，refill不进入实现阶段。

记录旧kernel摘要/commit、实际默认配置与工作区diff，不暂存或提交。纠正混淆的
prepared/graph比较：保留现有完整计时，图重放和直接绑定只能做单独标识的诊断。

### B. 独立实现与静态检查

先refill实数状态机，再复数；先H=1验证sharded数学/输出路径，再H>1。
独立分支不重写旧计算体。核对地址类型、mask、SSA循环类型、编译期形状，AST与
diff检查通过后交计算节点。AST通过不代表Triton编译通过。

### C. 原pytest准确性

使用原 `test_spmv_csr_accuracy.py`、marker和runner，无新入口。quick小矩阵
覆盖六dtype、四种half组合及声明ops；normal增加合法混合索引、长行和stride。

refill专用：全空窗口、W/R尾部、连续空行、[1,1,1,8,8,1,1,1]工作轮数形态、
重复列、单超长行、所有槽同时完成/只有一个槽完成、最后一次领取不足R。
验证每行唯一写入、NaN out完整覆盖、输入不变。

sharded专用：单目标列热点、均匀目标列、同program内重复坐标、跨program热点、
H=1/2/4/8、源program模H偏置、空行列、复数非对称conj、FP64 atomic能力拒绝、
容量不足、半精度宽输出没有提前窄化。保留别名、prepared/op/config冲突与安全回退。

共享容差不变；编译/执行失败FAIL。仅明确能力缺失SKIP。不用带输入溢出的FP16
矩阵证明数值成功；该类真实性能记录保留并标注，另用有限值合成样例验证精度。

### D. 性能与可否继续投入

refill对照原row_tile/subgroup/row_vector及新候选W=R。固定V/R先改W，再改V；
分别报告结构不规则子集和全部30矩阵。sharded依次比较旧重建、同精度H=1、H>1；
完整run和初始化/scatter/reduce诊断分开，禁止用compute-only冒充总收益。

沿用 `ms=process_cpu_ms+gpu_ms`，诊断不替换gpu_ms，保留prepared_run_ms、吞吐、
逻辑带宽、采样过滤和精度FAIL时间。CUDA保留同格式同操作cuSPARSE基线；若regular
trans入口无基线，诚实记录N/A并比较自身算法，不借用其他格式/类型基线。

目标子集可用 >=1.3x 作为继续投入门槛，>=1.5x作为重点推广候选；这些是项目
决策阈值，不是预计收益。必须重复测量确认超过噪声，保留每矩阵回退和精度状态。
若仅H=1胜出而H>1不胜，报告“接口/免重建收益”，不宣称分片思想成功。
若refill只在合成数据获胜，真实矩阵没有收益，则停止，不继续叠加预取来掩盖失败。

CUDA使用原 `tools/run_cuda_tests.sbatch`，已分配计算节点用bash，登录节点用sbatch。
两个名称现已注册，下面的命令已核对CLI源码，尚未在计算节点运行。DCU沿用原
runner，不要求容器脚本；平台分开统计。

## 7. 预期与后续记录模板

| 候选 | 预期改善 | 明确不解决的问题 |
|---|---|---|
| row_refill | 不规则局部窗口中的子组空闲等待 | 孤立超长行跨program并行、x随机访问、全短行启动成本 |
| transpose_sharded | 原生转置散射的跨program热点竞争；避免每次CSR重建 | program内部冲突、无竞争散射、巨大输出的初始化成本 |

后续每轮结果在所属日期目录保存：commit、配置、dtype四元组、op、索引、
矩阵、状态、ms/gpu_ms/prepared_run_ms、阶段诊断、基线状态/时间、旧最佳、新候选、
胜出比例与GM/AM。另记录寄存器/spill、layout转换、atomic/访存证据。无需profiler
也可先决定是否有完整耗时收益，但不能据此声称具体硬件瓶颈已经证实。

## 8. Triton 语法依据

- `tl.cumsum`：https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
- masked `tl.load`：https://triton-lang.org/main/python-api/generated/triton.language.load.html
- `tl.atomic_add` 类型、sem和scope：https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html

这里只使用现有Triton普通JIT的基础运算；不使用Gluon、内联汇编、协作grid barrier、
CUDA线程私有调度假设或仅Blackwell支持的warp_specialize。不把普通提前load称为
硬件异步流水。具体CUDA/ROCm编译合法性仍需计算节点验证。

## 9. 本轮源码交付与结构模型

- 新内核为 `row_refill_kernel`、`_sharded_scatter`、`_sharded_finish`；旧JIT函数
  的AST计算体与基线一致。配置为独立扁平字典；sharded在重建判断之前进入独立分支。
- 补齐高层新算法清单、strided out准入、原始输入检查、计算类型atomic能力、
  同算法fallback、操作范围过滤。静态支持表不把sharded标为支持non。
- regular CSV新增执行路径、工作空间、输出缓冲和stride字段；mixed已有metadata
  字段承载相同信息。原runner通过compare发现新名称，不改变默认交付算法。
- 原accuracy增加独立targeted用例；全类型通用用例按算法实际ops参数化，split
  配置只传给split算法。新增精确区分宽输出与half回转的样例、sharded实际入口
  的int64 fallback注入、容量异常和未知atomic能力策略用例。

离线模型已读取本地tests/data的10个真实MatrixMarket文件（非完整30矩阵），
按对称头展开结构，不coalesce重复项。结果在
`test_mixed_data/11.12data/analysis/spmv_refill_structure_model.csv`，复现入口为该目录
已有 `spmv_targeted_analysis.py --structure-root FlagSparse-main/tests/data`。
这不是算子测试，也没有导入Torch/Triton。

CUDA形状R=8,V=8,W=128的静态/补位工作轮数比：filter3D=1.692、ASIC=1.209、
auto=1.149、amazon0601=1.122、wave=1.117；roadNet-TX=1.000、NACA0015=1.003。
比值未计调度、寄存器和访存成本，不能称为加速比。优先filter3D验证补位是否兑现；
roadNet-TX/NACA0015作为负对照。转置分片优先观察amazon0601（最大目标列入度2751），
但目标列入度本身不能证明跨program竞争；需H=1/H>1同批实测。

## 10. 计算节点命令（先小样本，再全量）

以下从仓库根目录、已分配CUDA计算节点运行。登录节点将`bash`改为`sbatch`并在
脚本文件名前提供站点允许的分区/GPU参数。不要在登录节点用bash直接启动测试。
所有CUDA性能命令保留vendor；regular转置入口若不支持同格式vendor，仍如实N/A。

### 10.1 原pytest：新路径quick，然后CSR完整quick/normal

```bash
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase accuracy --mode quick \
  --pytest-args "-k 'targeted or sharded' -x" \
  --results-dir pytest_results_spmv_targeted_smoke

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase accuracy --mode quick \
  --results-dir pytest_results_spmv_targeted_quick

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase accuracy --mode normal \
  --results-dir pytest_results_spmv_targeted_normal
```

第一条包含新路径六dtype、四half组合、合法ops、配置边界和宽输出样例；-x只用于
准确性小样本遇错停止。通过后再跑下面性能，不直接提交整批长任务。

### 10.2 单矩阵基线与编译确认

```bash
MATRIX_DIR="$HOME/run/matrix"
test -s "$MATRIX_DIR/filter3D.mtx" || exit 1
COMBOS="float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32"
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase performance --mode quick \
  --benchmark-input "$MATRIX_DIR/filter3D.mtx" \
  --benchmark-warmup 3 --benchmark-iters 10 \
  --benchmark-args "--type-combos $COMBOS --alg compare --ops all --index-dtypes int32 --indptr-dtypes int32 --timing --perf-metrics --fail-fast" \
  --results-dir pytest_results_spmv_targeted_matrix_smoke
```

compare按ops筛选，non中不会调用sharded。精度FAIL继续保留；编译/接口异常在此
smoke命令中停止，先检查traceback，不继续提交完整矩阵集。

### 10.3 全矩阵 regular 和四组合 mixed

```bash
test -n "$MATRIX_DIR" && test -d "$MATRIX_DIR" || exit 1
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase performance --mode normal \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--dtypes float32,float64,complex64,complex128 --alg compare --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing" \
  --results-dir pytest_results_spmv_targeted_regular

bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmv_csr --phase performance --mode normal \
  --benchmark-input "$MATRIX_DIR" --benchmark-warmup 10 --benchmark-iters 50 \
  --benchmark-args "--type-combos $COMBOS --alg compare --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --perf-metrics" \
  --results-dir pytest_results_spmv_targeted_mixed
```

吞吐/逻辑带宽开关当前只用于type-combos路径，不为regular捏造不支持的CLI。
两组包含原算法及新候选；不要从不同批次抽最小值充当同批最佳。

### 10.4 单因素消融（仅显式单算法）

```bash
mkdir -p targeted_ablation
for W in 8 32 64 128; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
    "$MATRIX_DIR" --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --alg csr_row_refill_reduce --ops non --index-dtypes int32 --indptr-dtypes int32 \
    --config "{\"rows_per_program\":8,\"lanes_per_row\":8,\"window_rows\":$W}" \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csr "targeted_ablation/refill_w${W}.csv"
done
for H in 1 2 4 8; do
  bash tools/run_cuda_tests.sbatch --repo "$PWD" --entry tests/test_spmv_csr.py -- \
    "$MATRIX_DIR" --type-combos float16:float16:float32,bfloat16:bfloat16:float32 \
    --alg csr_transpose_sharded_atomic --ops trans,conj --index-dtypes int32 --indptr-dtypes int32 \
    --config "{\"output_shards\":$H}" \
    --warmup 10 --iters 50 --timing --perf-metrics \
    --csv-csr "targeted_ablation/sharded_h${H}.csv"
done
```

DCU已激活库环境后，把runner命令前缀替换为`python run_flagsparse_pytest.py`，
直接benchmark前缀替换为`python tests/test_spmv_csr.py`，其余参数不变。若该DTK
已确认不支持mixed vendor，可显式增加--no-vendor；自身准确性/性能仍须完整执行。


### 2026-11-14 smoke 修复状态

filter3D 的 FP16 non smoke 在 refill 首次编译时失败：单位 stride 特化为 Python int，不能调用 `XS.to(tl.int64)`。refill/sharded 的标量 stride、维度转换已改为 `tl.cast(..., tl.int64)`，保持64位寻址；同步修正 SpMM NNZ atomic 的单位行数转换。原测试补充连续/strided 输出和单行特化回归。当前仅完成源码与AST检查，未进行修复后的GPU编译及性能验收。
