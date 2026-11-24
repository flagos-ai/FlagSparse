# CSR SpMV 独立静态 consumer 融合与任务粒度诊断（2026-11-24）

## 状态、动机与边界

目标为 FlagSparse-main / zyq_sparse，实施前基线为
`6d79ed5350cc33e4476563ce515728899786af21`。本轮只增加
`csr_three_tier_fused_static_reduce`，不替换旧算法、ticket/static 对照、默认选择或缓存。
源码已接入，GPU 编译、正确性和性能尚未验证；静态检查不是硬件验收。
本轮不修改 SpMM，不扩展 trans/conj，不引入新 Python 文件。

11.23-newdata 的分析显示 ticket 动态领取没有取得家族最佳，部分矩阵根本没有首轮以外的任务。
新的实验只验证：在同样短行分类、中行整行处理、长行固定 P 部分和的基础上，将两个 consumer launch
合成一个是否改善完整调用。它不声称解决所有长尾负载不均衡，也不把少一次 launch 等同于必然加速。
事实与后续依据位于日期目录的 `analysis/REPORT.md` 和 `NEXT_SOURCE_PLAN.md`。

## 执行与正确性

每次 run 的顺序为初始化 → 原 producer → 新联合 consumer → 原 finish。
producer 写短行结果，生成互斥的中行和长行行号队列；计数不读回 CPU。
沿用上界 Cm=min(M,nnz/(S+1))、Cl=min(M,nnz/(L+1))，除法取整。
实际启动 Gm=min(Cm,medium_grid_blocks)，Gl=min(Cl,long_grid_blocks)，W=Gl×P。

联合 consumer 网格为 Gm+W：

| program 域 | 起点 | 下一任务 | 写入 |
|---|---|---|---|
| pid < Gm | 中行队列 slot=pid | slot += Gm | 整行最终输出 |
| pid >= Gm | task=pid-Gm，slot=task//P，part=task%P | task += W | 独占部分和 |

两种跨步不能使用联合网格大小。每个有效 slot/task 有唯一 residue class，因此不重不漏。
每长行 chunk 数 C=ceil(nnz_row/K)，part p 处理 [floor(Cp/P),floor(C(p+1)/P))；
空 part 也写零。独立 finish 在同流 consumer 完成后归约 P 项，不依赖跨 program 自旋或屏障。
短、中、长三类最终输出互斥，空行在短行分支写零；不清零整个输出，不使用浮点 atomic。
队列入队仍使用原 producer 的整数 atomic，融合 consumer 不领取 ticket。

源码位置：`src/flagsparse/sparse_operations/_spmv_csr_kernels.py` 中
`three_tier_fused_static_consumer_kernel`、`run_three_tier_fused_static`；长行数值计算复用未改动的
`_ticket_exclusive_part`，producer/finish 复用原实现。配置与容量在 `_spmv_csr_config.py`。
普通 API、prepare/run、compare 和原 spmv_csr pytest marker 均到达独立运行分支。

## 支持与资源

仅 non，沿用 CUDA/ROCm 能力边界、合法 int32/int64 指针/索引、同算法安全回退及正 stride out。

| 输入 | 中间计算/部分和 | 输出 |
|---|---|---|
| FP16/BF16 | FP32 | 同类型或 FP32 |
| FP32/FP64 | FP64 | 同输入类型 |
| complex64/complex128 | FP32/FP64 实虚分量 | 同输入类型 |

默认参数与 bounded 相同：S=32、L=1024、K=128、P=4、Gm上限512、Gl上限128，stage=1。
性能对照显式用 P8/P16、Gm1024、Gl128；默认不随实验变动。专用配置不能跨算法继承。
不接受 ticket_batch、long_schedule 或 long_worker_programs；这些仍属于 ticket 控制组。
保留 static 对照的三个 int32 计数槽，第三槽恒为零，以避免把省去计数初始化误算成融合收益。
数值空间为 Cl×P×分量数×分量字节，描述符为12+8(Cm+Cl)字节；默认上限256 MiB。
容量不足明确报错，不换算法、不读回实际队列数量后缩小缓冲。

融合只改变 consumer 提交与分工。代价可能包括联合 kernel 的寄存器/代码资源约束、program 排序
及两种工作域调度差异；不存在保证中长行在硬件上同时运行的语义。须用同批 static 对照验证。

## 计时与诊断口径

保持 ms=process_cpu_ms+gpu_ms，精度 FAIL 不剔除有效性能；保留 cuSPARSE 接口及采样过滤。
完整性能采样不插入分段或任务插桩；phase diagnostics 另行 run。

| 字段 | 含义 |
|---|---|
| producer_ms | 原分类与短行计算联合阶段 |
| consumer_fused_ms / consumer_ms | 同一次联合 consumer 阶段 |
| medium_consumer_ms / long_consumer_ms | 新算法留空，无法独立测量 |
| finish_ms | 长行数值归并 |
| consumer_timing_scope | joint_medium_long |

phase 事件测量包含其原有主机提交/同步影响，不能把旧中行+长行阶段和新联合阶段差额直接当作
节省的硬件 launch 成本。最终判断使用完整 ms 与独立 prepared_run_ms，并保留全部采样值。
regular benchmark 仍为预分配 out 的外层事件；mixed 保留内部 run 事件及原输出分配边界。
只在同类型、同入口内作新旧比较，不声称两入口有完全相同的包装成本。CSV额外记录
`output_reused`、`event_time_boundary`、`output_allocation_policy`，区分用户预分配与run内目标分配。

`--task-diagnostics` 仅适用于 CSR SpMV，要求 `--perf-metrics` 和 CSV；
`--task-diagnostic-repeats` 为1..16，默认1。未启用时不运行插桩 kernel，不读回队列。
其他算法在该开关下记录 not_applicable，仍正常计算性能，不从 compare 删除。
诊断文件写入 `<csv>.task_diagnostics/<identity>.json.gz`，CSV记录路径及简要状态。

诊断包括：

1. CPU CSR 结构模型：每个长行 part 的 K-chunk 分布、零工作 part、分位数、最大值；这不是耗时。
2. 独立 run 的实际中/长行队列顺序、计数、queue hash，检查 owner 无遗漏、重复或交叉。
3. 在实际队列顺序上重建静态 worker 的任务数、chunk 总数、最重任务、首轮/余下工作量。
   旧二维协作映射与 flat static 映射分开记录，不能默认原 CSR 行序就是 GPU 入队顺序。
4. ticket 的独立插桩执行记录每 worker 的任务数、chunk数、最大任务、领取次数、首轮/余下chunk；
   验证总任务、总chunk、含退出领取的计数。B4最后不足4项按实际有效任务统计。
5. 原未插桩性能采样原值、wall采样原值，供后续判断过滤及轮次波动；不拿插桩run计时替代。

ticket trace 复用原数值 helper，但多读行指针并写诊断数组，可能改变动态领取顺序。
它只能说明该次插桩运行分配，不是先前计时运行的精确调度轨迹，更不是 profiler 的设备活跃时间。
benchmark 暂时包装模块中的 producer/ticket consumer，在 finally 恢复；仅用于串行 benchmark。
复测输出要求与未插桩同算法输出逐元素一致（包括相同 NaN）；该一致性检查不能替代 CPU golden 准确性。

## 测试与验收

扩展原 `tests/pytest/test_spmv_csr_accuracy.py`：六种dtype、四种half输出组合、混合索引、
空矩阵/全空行、三类纯集合/混合集合、反向长短分布、多轮、空part、P1/P8/P16、不同Gm/Gl、
NaN预填正stride out、对象身份、输入不变、宽输出非半精度回转、op/config冲突和安全索引回退。
新case禁止调用旧run，捕获联合网格及部分和完整覆盖；编译或执行错误必须FAIL。
任务诊断覆盖 bounded、fused、ticket static/B1/B4，并检查异常后恢复。
策略测试沿用 `tests/ci/test_spmv_csr_policy.py`，runner/marker没有第二套入口。

先 regular/mixed smoke，再对应 normal，最后同批30矩阵性能。
保留 compare 的旧最佳、显式 S128、bounded P8/P16、ticket static P8/P16及新融合P8/P16；
另外 ticket B1提供长尾工作分布诊断。P/Gm/Gl以外参数一致。
建议至少3轮交替先后顺序，分别统计每轮与跨轮中位数，避免在单轮最小值上作成功结论。

重点看 mip1 的中行consumer成本，wiki-Talk 的长行工作/零part分布，TSOPF 的旧row_vector差距；
同时记录整套矩阵GM、胜出数、回退与准确性。新算法有无用须看加入后的整体最优，而不只看赢static。
没有完整调用收益则保留实验结论，不切换默认、不把诊断模型当作优化成功证据。

硬件验收尚未执行。后续若融合无收益，依据实际worker工作量区分“颗粒度失配”与“提交/初始化成本”，
再决定是否需要改变长行分片策略；本轮不预先添加另一种调度算法。
