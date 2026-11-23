# SpMV / SpMM 优化设计与审查

本目录专门存放 FlagSparse SpMV、SpMM 的优化设计、技术评审和验收规范。设计文档不等于实现或性能承诺；日期数据分析继续保存在对应的 `test_mixed_data/<日期>/analysis`，不搬移到本目录。

## 文档清单

| 文档 | 范围 | 当前状态 |
|---|---|---|
| [11.23 CSR SpMV 长行ticket动态领取方案](CSR_SPMV_TICKET_LONGROW_PLAN_20261123.md) | 首轮静态＋余下隐式分片动态领取；static对照、B1/B4及原测试接入 | 源码/静态检查完成，GPU编译与实机待验收；non-only，保留旧算法及默认路由 |
| [11.22 CSR SpMV 三类任务协作计划及11.23有界中行增量](CSR_SPMV_THREE_TIER_QUEUE_PLAN_20261122.md) | 同批S128、G及L/P验证；新增有界中行＋长行协作独立候选 | 原候选11.22批次零次最佳；修复strided out遗漏，新bounded候选源码接入、硬件待验收；保留默认路由 |
| [CSR non SpMV 定向优化候选](CSR_NON_SPMV_TARGETED_CANDIDATES.md) | scalar-strip、static-tile、长行任务队列及 Triton 实现约束 | 设计提案，待独立评审，未实现 |
| [11.16 CSR SpMV / CSC SpMM 定向方案](CSR_SPMV_CSC_SPMM_TARGETED_PLAN_20261116.md) | 短行条带、静态块、中长行固定分段、CSC non列分段/输出分片及实施门槛 | 短行A/B与CSC non源码接入；静态检查完成，硬件未验证；中长行C待门槛 |
| [11.17 独立SpMV候选与baseline排查计划](SPMV_INDEPENDENT_CANDIDATES_BASELINE_PLAN_20261117.md) | CSR短行、CSR长行GPU队列、CSC trans/conj短列；BF16厂商诊断及原测试接入 | 三个独立候选与baseline诊断已源码接入，静态检查完成；GPU正确性/性能待验收 |

## 使用规则

1. 每份方案分别标明源码实现、GPU正确性、完整调用收益三个状态。
2. 说明旧算法对照、算法实质差异、准备/运行边界和支持范围。
3. 假设、已观察事实、未决设计必须明确分开；不能将历史厂商或其他设备的收益移植为本库收益。
4. 审查意见可追加到对应方案的评审记录；结论必须指向具体问题与证据。
5. 默认保留旧算法、默认路由和数值策略。未经授权，不以设计评审为理由修改算子代码。

相关既有文档：[SpMV targeted 算法设计](../SPMV_TARGETED_ALGORITHM_DESIGN.md)。
