# CSR SpMV：长行ticket动态领取独立算法实施设计

日期：2026-11-23。状态：**源码接入及静态检查完成；GPU编译、正确性与性能待验证**。目标分支：FlagSparse-main/zyq_sparse。修改基线commit：0636fc091b7dab360a122d64d28768416a8e295f。测试批次11.23中的commit为unknown，不将本地源码版本当作远端版本证明。

本方案只新增独立候选`csr_three_tier_ticket_reduce`，保留原算法、默认路由、默认配置、计时与精度策略。下文保留设计与验收要求，实际接入情况见第12节。本地不运行算子/pytest，不暂存提交。详细命令按实际CLI核对后直接交付，不写入本文。

## 1. 设计决策与范围

核心变化是：保留短行producer、中行bounded consumer和固定P部分和，将长行的静态跨步任务分工改为“首轮静态分配，剩余任务通过整数ticket领取”。它是长行调度候选，不是短行改造，也不是consumer融合或NNZ分段算法重命名。

**操作范围只声明non。** 当前`_spmv_csr_config.py`明确把三类任务算法放入NON_ONLY_ALGORITHMS。此前分析文档提到沿用三种op属于错误描述，本方案纠正；trans/conj显式请求报不支持，compare按规格排除。本轮不扩转置。

| 输入A/x | 计算及部分和 | 输出 |
|---|---|---|
| FP16/BF16，同类型输入 | FP32 | 同输入类型或FP32 |
| FP32 | FP64，沿用当前新CSR策略 | FP32 |
| FP64 | FP64 | FP64 |
| complex64 | FP32实虚分量 | complex64 |
| complex128 | FP64实虚分量 | complex128 |

保留合法int32/int64行指针/列索引及同算法安全回退；保留strided x/out、别名拒绝、out身份、prepare/config匹配。不增加混合A/x类型、任意输出精度或FP64变体。后端首版与原三类算法相同，仅CUDA/ROCm，能力未知明确不可用。

## 2. 与已有算法的实质区别

| 项目 | csr_three_tier_bounded_reduce | csr_longrow_queue_atomic | 新ticket候选 |
|---|---|---|---|
| 普通行处理 | 短行producer、中行队列 | producer处理非长行 | 与bounded相同 |
| 长行任务 | 每行固定P片 | 按segment_nnz生成变长任务数 | 每行固定P片 |
| 任务定位 | 二维program_id+静态跨步 | JOB_SLOT/START/END三数组 | 由t//P、t%P和CSR行指针推导 |
| 任务取得 | program固定处理相应行槽/part | 整数atomic每次取一个descriptor | 首轮静态，余下整数ticket；可小批领取 |
| 数值写入 | 每片独占partial | 浮点atomic合并到行partial | 每片独占partial |
| 归并 | 固定P归约 | 拷贝/转换已atomic累加的行结果 | 与bounded相同的固定P归约 |
| 计划构建 | 两条行号队列 | 行号＋三数组分段描述符 | 两条行号队列，另加一个int32 ticket |
| 长行小于worker数 | 静态直接执行 | 仍有领取atomic | 直接执行，不使用领取atomic |

因此新算法并不是首次在库内使用动态队列。它借用旧动态领取思想，但针对当前固定P体系去掉额外NNZ描述符和浮点atomic；与bounded的主要对照因素是长行任务分配，而非计算精度或分片方式。

不能断言它优于旧queue：旧queue的segment长度相近，单任务负载可能更均匀；新ticket仍有很长/很短的任务。必须保留两个旧路径，以及NNZ-local、S128、完整旧算法族作为对照。

## 3. 变量、配置与默认对照

符号：M行数，Z=nnz，S短行阈值，L长行阈值，K每片循环向量宽度，P协作分片数。Cm/Cl为host可证明的容量，Qm/Ql为producer写入GPU的实际计数。

\[
C_m=\min(M,\lfloor Z/(S+1)\rfloor),\quad
C_l=\min(M,\lfloor Z/(L+1)\rfloor),\quad T=Q_lP,\quad T_{cap}=C_lP.
\]

配置延续bounded的默认值，避免新默认调参混入调度收益：R128、S32、L1024、K128、P4、Gm512、Gl128、stage1及原合法warp解析。实际中行grid=min(Cm,Gm)，finish行grid=min(Cl,Gl)。

新算法专用字段：

| 字段 | 首版默认/合法值 | 用途 |
|---|---|---|
| long_schedule | ticket；ticket/static | static是新kernel内部因果控制，不调用旧算法run |
| ticket_batch | 1；1/4 | 首轮之后每次领取B个任务；static要求为1，拒绝无效消融字段 |
| long_worker_programs | 0；0或32/64/128/256/512/1024/2048/4096 | 0明确表示按原Gl×P派生；显式值只影响long consumer |

默认W=min(Cl,Gl)×P；显式worker值时W=min(Tcap,worker_limit)，不超过实际容量上界，元数据同时记录请求上限与实际W。Tcap=0时W=0，跳过长行consumer。这里的容量裁界是grid定义，不是静默修改显式配置参数。

long_grid_blocks在新算法中仍控制finish网格，并为worker=0提供默认派生基准；显式W与finish网格独立，便于只改worker数量。所有新字段只对ticket算法合法，不向旧算法注入，不改旧配置解析结果。

首批实验明确覆盖P8/Gm1024/Gl128（默认派生W1024）与P16/Gm1024/Gl128（W2048）。不把Gm扩容、finish批量化或consumer融合放进本次实现。

## 4. 完整执行与Triton实现约束

### 4.1 host运行阶段

1. 检查配置、容量、workspace及ticket溢出上界，只用shape/nnz等已有host信息。
2. 分配计数`COUNTS[3]`并清零：Qm、Ql、ticket。分配中/长行号队列和计算精度partial。partial不清零。
3. 复用原producer：只写COUNTS前两个位置，计算短行并写两条队列。
4. 原bounded中行数学kernel处理完整中行。
5. 新独立长行kernel执行静态控制或ticket调度，写独占partial。
6. 复用原固定P finish，写目标out。

同一实际设备流顺序提交。producer、consumer、finish之间靠kernel完成顺序保证数据可见；没有GPU全局自旋、跨CTA屏障或host计数读回。每次run重置ticket与队列，不缓存计划。

### 4.2 每个任务的数学内容

t唯一映射为slot=t//P、part=t%P；从LONG[slot]取得row，读取CSR begin/end。令c=ceil((end-begin)/K)：

\[
chunk_{lo}=\lfloor cp/P\rfloor,\quad chunk_{hi}=\lfloor c(p+1)/P\rfloor.
\]

依原实现连续处理[begin+K×chunk_lo, min(end,begin+K×chunk_hi))，每K项加载并累加，最终沿K归约到partial[slot×P+part]。复数分别写实虚部。空分片也写零，不能跳过导致finish读未初始化数据。

### 4.3 先分配首轮，再领取余下任务

每个program首先处理t=pid（若pid<T）。这与固定任务序列的前W项对应，避免让所有program先争抢一个ticket才能开始计算。

- 若T≤W：各有效program恰好计算一个任务；不执行任何ticket atomic。
- 若T>W：所有W个program各自完成首项后，领取剩余任务。ticket初值0，领取起点为W+atomic_add(ticket,B)，随后顺序消费最多B项；到起点≥T时退出。

算法级伪代码如下，仅描述控制流，**不是可直接编译的Triton实现**：

```python
def long_worker(pid, W, T, B, schedule):
    if schedule == "static":
        task = pid
        while task < T:
            compute_one_exclusive_part(task)
            task += W
    else:
        if pid < T:
            compute_one_exclusive_part(pid)
        if T > W:
            base = W + atomic_fetch_add_ticket(B)
            while base < T:
                for offset in range(B):
                    task = base + offset
                    if task < T:
                        compute_one_exclusive_part(task)
                base = W + atomic_fetch_add_ticket(B)
```

正式实现：pid/T/base/地址均转int64；atomic指向int32且增量为对应类型，返回值立即转int64；B/P/K为编译期常量。两个分支使用固定[K]累加器；每个任务重新初始化。新增纯任务数学helper可以供该kernel的static/ticket分支共享，但不能改变旧kernel计算体。

每个program的领取是一个逻辑标量atomic，不构造每lane领取的向量。使用`sem="relaxed", scope="gpu"`。Triton文档确认atomic_add返回更新前的值并支持这两项语义；relaxed足够用于唯一编号，不承担队列发布同步。发布同步来自前一kernel完成，而不是ticket。参见[Triton atomic_add官方文档](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html)。最终仍需在计算节点核查编译与生成代码。

### 4.4 static内部控制的意义

static模式使用同一1D网格、同一任务解码及数学helper，唯一差异是task+=W而非领取ticket。W=Gl×P时，对同一part的行槽增量为Gl，与旧二维cooperative的任务集合一致。

但1D/2D launch顺序、编译结果可能不同。因此先测旧bounded vs 新static，再测新static vs 新ticket，避免把launch表示变化造成的收益误称为动态调度收益。static控制仍报告新算法及实际long_schedule，不能伪装旧名称。

## 5. 整数atomic额外开销的可计算部分

对T>W，剩余批次数U=ceil((T-W)/B)。每个worker在完成最后一次有效领取后，还执行一次失败领取用于退出（未领取到有效剩余任务的worker同样如此）。所以逻辑领取次数为：

\[
A(T,W,B)=\begin{cases}0,&T\le W,\\
\lceil(T-W)/B\rceil+W,&T>W.
\end{cases}
\]

B=1时T>W有A=T；相较所有任务都从ticket开始的T+W方案，省去W次初始atomic。B=4降低领取次数，但每次保留一组任务，使负载均衡粒度变粗；若相邻任务同属一行，可能提高局部性，也可能让一个worker连续承担重任务。

按11.23结构计数推导（不是GPU实测atomic数）：

| 矩阵/P | Ql | T | 默认W（Gl128） | B1领取次数 | B4领取次数 |
|---|---:|---:|---:|---:|---:|
| wiki/P8 | 992 | 7936 | 1024 | 7936 | 2752 |
| wiki/P16 | 992 | 15872 | 2048 | 15872 | 5504 |
| mip1/P8 | 72 | 576 | 1024 | 0 | 0 |
| mip1/P16 | 72 | 1152 | 2048 | 0 | 0 |
| TSOPF/P8 | 69 | 552 | 1024 | 0 | 0 |
| TSOPF/P16 | 69 | 1104 | 2048 | 0 | 0 |

实际成本不能用atomic次数乘一个固定常数得到。所有worker争用同一地址，成本取决于并发、缓存一致性、目标后端、CTA驻留及内存系统。B1增加的串行领取可能吞掉所有均衡收益。它不是“整数atomic所以几乎免费”。

成本判断使用：减少的静态尾部等待是否大于领取竞争、动态控制流和局部性变化的代价。不得用总task数证明必然获益。只有长行阶段改善且完整run改善，才算有效。

## 6. 正确性与溢出证明

### 6.1 无遗漏、无重复、无写竞争

1. producer把每行唯一划入短/中/长之一，队列slot唯一。
2. 初始任务集合[0,min(T,W))由pid唯一拥有。
3. T>W时atomic每次返回互不重叠的B长度区间；加W后恰覆盖剩余[W,T)，尾部用task<T屏蔽。不存在重叠初始集合。
4. t到(slot,part)为一一映射，故每个有效partial恰写一次。
5. 该行的P个chunk区间首尾相接，覆盖全部NNZ、不重叠；空片显式零。
6. finish在所有consumer完成后读取这些partial。短/中/长输出行互斥，空行由短行写零。

### 6.2 结束性与数值策略

每次有效领取使ticket增加B，单任务的CSR区间有限。worker不等其他worker释放锁，也不自旋等待publisher；耗尽后退出。不要求全部worker同时驻留，否则可能死锁的persistent barrier在本设计中不存在。

动态调度只改变任务在哪个program、何时执行，不改变片内NNZ次序或finish的P归约顺序；没有浮点atomic竞争。新static/ticket共用数学helper时预期数值一致，仍以共享容差为验收标准，不把跨编译结果的bitwise相同当保证。P变化造成的求和顺序变化单列；既有wideout FAIL不得掩盖。

### 6.3 ticket溢出上界必须包含退出领取

使用Python整数按容量Tcap和实际W预检：

\[
V_{max}=\begin{cases}0,&T_{cap}\le W,\\
B\left(\lceil(T_{cap}-W)/B\rceil+W\right),&T_{cap}>W.
\end{cases}
\]

要求Vmax≤2^31−1，另保留原Qm/Ql计数的int32容量检查。否则明确报ticket容量不支持；不无条件转int32、不自动换算法，也不默默引入int64 atomic后端要求。

最重要的边界：B>1时只检查Tcap≤INT32_MAX并不充分，尾批padding和W个退出领取也可能越界。B=1、Tcap>W时Vmax=Tcap；B=4时可能超过Tcap约3W个单位及尾部对齐量。所有乘加/地址计算在host Python或GPU int64进行，不能先int32溢出再检查。

当Cl=0，W=0且跳过长行launch，不能在host执行除以W；GPU算法中从不除以W。T>W分支内W>0由launch定义保证。

## 7. 工作空间、接口和能力

原容量不变。新数值空间为Cl×P×分量数×分量字节数；描述符空间为12+8(Cm+Cl)字节，相对旧三类算法只增加4字节ticket。仍使用256MiB默认总workspace限制，峰值包括全部存活缓冲，校验发生在分配前。不能沿用旧函数的8字节计数结果后漏计新增ticket。

可以独立实现ticket容量wrapper复用旧容量数学，但必须按新总量再次检查workspace并返回真实字段；不改变旧capacity helper的返回值。输出存储按既有约定单独记录。

后端能力复用main：int64地址、reduction、scan及int32 atomic；FP64计算依实际dtype检查。不需要浮点atomic能力，也不新增平行后端名单。未知能力报不可用；配置/编译/运行错误不得吞成SKIP。

通过原prepare/run/high-level alg/config接入；新增规格进入查询、compare和支持表，不新增顶层算法函数。旧默认不变；prepared算法/config不匹配报错。索引回退保持ticket配置、计算/输出类型、out及non语义。

元数据记录long_schedule、ticket_batch、worker请求/实际W、task_capacity、ticket_counter_dtype、ticket_counter_max_bound、数值/描述符空间、执行版本及精度。正常run没有实际Ql读回，因此不能伪造actual_atomic_count；CPU结构统计可记predicted_ticket_acquisitions并标明来源。不能宣称T≤W旁路实际发生，除非来自独立诊断。

## 8. 计时与因果对照

保持init→producer→medium→long→finish五阶段，不与consumer融合或tiled finish合并实施。

- ticket清零及新增计数分配归入init；领取、片内计算、partial写入归入long consumer。
- 全部初始化、队列构建、数学计算及归并都处于完整run；process_cpu_ms正常为0。
- 总ms仍为process_cpu_ms+gpu_ms，分段诊断另行运行，不用分段之和覆盖总时间。
- 吞吐和logical bandwidth继续使用既有定义；逻辑带宽模型不能把ticket cacheline真实流量当作已测显存流量。
- cuSPARSE保留，baseline不可用不阻断FlagSparse；FAIL有效时间保留。

因果比较按四层组织：旧bounded(P/G相同)→新static(W相同)→新ticket B1→新ticket B4。改变W时保持P、Gm、finish Gl固定；改变P时明确部分和/求和顺序也改变。最后才与旧最佳含S128、NNZ-local和旧动态队列比较整体最佳GM。

## 9. 原pytest与诊断的接入方案

不新增Python文件或测试入口。原CSR accuracy和策略测试扩展新规格、路由与专用用例；旧用例不被替换。

必须覆盖：

- T=0、1、W−1、W、W+1；B批尾；W远小于有效任务数。结合P=1避免部分T受Ql×P限制，P4/8/16另测。
- 新static与ticket B1/B4；先领取的任务特别长、其他任务短的合成队列；长行顺序正序/逆序。随机GPU执行顺序不可用于假定确定排队。
- 六dtype及四种半精度组合，空矩阵、短/中/长单类、混合类、未排序列、重复项、复数和抵消。
- partial预填NaN，验证所有有效slot写完；out预填NaN、strided x/out、返回身份、输入不变及别名拒绝。
- 验证同算法索引fallback真正进入ticket run；当前测试的二选一run映射必须改成显式算法到run字典，不能把第三个名称误映射到capacity路径。
- 验证每次run新建并清零ticket；相同prepared多次运行不能沿用上次票号。
- 参数非法、workspace仅差4字节、Tcap/退出领取的int32边界，使用纯策略数字测试，不分配巨大GPU张量。
- non声明与trans/conj拒绝；不得为新算法误生成全部ops参数组合。

测试专用诊断可给每task设置visit计数、捕获ticket和队列，断言visit=1及最终ticket符合公式；这些计数/读回不得进入生产kernel或性能采样。若需要专用编译特化，必须有显式test-only constexpr且生产为False，同时承认诊断特化不是生产性能结果。

## 10. 实施顺序与验收门槛

1. 固定旧kernel源码摘要与默认配置，新增ticket规格、配置、容量及溢出策略；不改旧计算体。
2. 先实现新独立long kernel的static控制与复用数学helper，接通完整run和输出。
3. 增加首轮静态+余下ticket B1；证明任务唯一性和结束性，完成容量边界测试设计。
4. 增加B4同实现特化；不增加另一公开算法名称。
5. 更新原pytest、benchmark元数据/CPU结构诊断、runner算法枚举、CI及支持文档。特别核对算法参数化和fallback映射，不漏第三个算法。
6. 本地只源码/AST/diff检查。用户在计算节点先smoke/normal，再同批性能。

第一轮目标：wiki，正向收益假设；mip1/TSOPF，预期无领取atomic或收益有限的负面对照；再扩全30矩阵。P8/P16固定Gm1024/Gl128，regular与mixed分开，CUDA/ROCm分开统计。

成功标准不是“ticket比自身static快”这一项：需要准确性没有未解释回归，long consumer与完整run均改善，并提高目标条件或全体逐矩阵最佳表现。宽输出现有FAIL保留分析，不能说算法已完全验收。

如果ticket开销超过静态尾部损失，保留明确负结果并停止；如果最大单片本身主导，动态领取不能拆开它，应另行考虑粒度设计。不得通过降低精度、改阈值或改变baseline分母挽救该候选。

## 11. 源码与证据定位

- `_spmv_csr_config.py`：THREE_TIER_ALGORITHMS、NON_ONLY_ALGORITHMS、three_tier_queue_capacity、resolve_config。
- `_spmv_csr_kernels.py`：three_tier_producer_kernel、short_rowqueue_strided_vector_kernel、cooperative_row_consumer_kernel、cooperative_row_finish_kernel、run_three_tier_bounded。
- 旧动态路径：longrow_producer_kernel、longrow_consumer_kernel、run_longrow_queue。
- `tests/pytest/test_spmv_csr_accuracy.py`：three_tier_native、three_tier_mixed_wide_output、three_tier_mixed_index_fallback。
- `tests/benchmark_utils.py`：csr_tile_structure与类型组合benchmark。
- 数据：[11.23分析](../../../test_mixed_data/11.23data/analysis/REPORT.md)、[后续源码论证](../../../test_mixed_data/11.23data/analysis/NEXT_SOURCE_REVIEW.md)。

## 12. 源码交付记录（2026-11-23）

- 新增`_ticket_exclusive_part`、`three_tier_ticket_consumer_kernel`、`run_three_tier_ticket`，原49个kernel/runtime函数AST保持不变。static/ticket为constexpr特化；首轮任务后仅在T>W时领取，B1/B4均已接入。
- 新增`ticket_counter_bound`及`ticket_queue_capacity`，按终止领取余量检查int32上界，计入额外4字节计数；旧容量函数不变。
- 原注册/prepare/run、算法查询、strided out校验和两级导出通过既有公共API到达新独立分支。新增算法没有专属顶层API，不扩大缓存/直接绑定范围。
- 原regular及mixed benchmark追加ticket配置/worker/容量字段；CPU结构诊断提供预测领取次数并标明模型来源。原runner无需新分支，通过原spmv_csr脚本、compare及marker收集；GPU CI策略清单已更新。
- 原accuracy文件新增`ticket_native`（regular/mixed明确ID）、动态宽输出和regular回退测试；原mixed宽输出/回退及通用算法参数化增加新名称。normal扩展四种索引、P16、反向任务顺序、空分片和多轮消费。测试读回ticket和有效partial仅发生于accuracy，不进入生产实现。
- 原CI策略文件补配置隔离、能力、计数边界、模型字段和独立路由检查；没有在本地执行这些测试。任务唯一性采用源码数学证明、输出/有效partial完整性及最终票号检查，生产kernel未添加逐task访问计数缓冲。
- 仅源码、AST及diff检查；Triton lowering、真实调度和数值收益尚未验证。没有更改共享容差、cuSPARSE baseline、默认算法、原计时公式或采样规则。

已确认的范围仍是non；trans/conj显式不支持。Gm上限仍1024；本轮没有扩容Gm，没有实现consumer融合或tiled finish。

