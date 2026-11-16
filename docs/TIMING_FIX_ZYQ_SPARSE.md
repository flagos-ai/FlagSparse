# zyq_sparse SpMV/SpMM 计时修正

2026-11-16；基于 zyq_sparse 7892d78。main 保持 a5b7382，未在 main 提交改动。
本次修正实际执行计时，不改变数值累加策略，不强拆所有算法的独立 preprocess API。

## 执行边界

- CSR SpMM 不在 prepare 缓存 A.T/A.H 数值；每次运行计入请求的矩阵变换。
- CSR SpMV auto transpose/gather 在完整运行计时内建立该次转置结构，保持原 gather 计算策略。
- 常规 COO SpMM 的 trans/conj、FP16/BF16 数值物化延迟到 run；从原始输入读取最新值。
  普通非半精度 non 的 canonical prepared 数值快照仍保留原语义，调用方更新 A 后须重建。
- legacy mixed 计时从 contiguous、index 转换、lazy conjugate/negative 解析之前开始。
  不扩展既有不支持的 dtype/op 组合，native mixed baseline 保持直接 C API。
- CSC SpMV 的 run 对 native/half 路线统一完整 Event 和元数据，输入物化及输出处理在内。
- SpMM CSR/COO/CSC 的可选 phase 插桩在另一调用采集，不改变主测量；COO 诊断不再替换主时间。

## 统计和 baseline

- CPU/GPU 按同次调用配对，按完整样本筛选；普通 CSR/COO SpMM 不再用另一调用的 CPU 值补均值。
- CSR/COO SpMM 的 bucket descriptor CPU bridge 只构造 launch 字典/视图，设 CPU process=0；
  完整 Event 已跨过该桥接，不重复加 CPU wall time。真实 GPU bucket 构造仍计入。
- 普通/q4 CSR SpMV 不再单边使用私有 kernel binding 或 Graph。
- CUDA COO SpMV 对照改为 CuPy 包装的 native COO spmv 和 CuPy Event，不用预转 CSR+wall 伪装 COO。
- CSC CuPy trans/conj 在被测 lambda 内；CSC/COO SpMM 不再免费改变 B 的实际布局。
  保留 CuPy 高层结构；它仍是 wrapper 层计时，不冒称纯 vendor kernel 时间。
- mixed type-combos 与 q4 的 speedup 仅在本方及 vendor 正确性有效时使用。
- vendor 明确可复用的 preprocess 保持已有接口配置，本轮不强制启用所有可选 vendor 优化。

## 验证与限制

CPU 检查使用 tests/ci/test_spmx_timing_contract.py 和 tests/ci/test_benchmark_policy.py。
GPU 回归在 tests/pytest/test_spmx_timing_gpu_accuracy.py，覆盖 CSR/COO/CSC 的 non/trans/conj、
FP16/BF16→FP32、复数、op_b=conj、A 值更新及阶段诊断。

```powershell
python -m pytest tests/ci/test_spmx_timing_contract.py tests/ci/test_benchmark_policy.py -q
python -m pytest tests/pytest/test_spmx_timing_gpu_accuracy.py -q
python tests/test_spmm_csr.py --synthetic --type-combos float16:float16:float32 --ops non,trans,conj --op-b all --alg auto --warmup 2 --iters 5 --timing --csv-csr timing_smoke.csv
```

当前本机无 torch/triton，不能执行真实 GPU 数值及性能验证；不能据 CPU 插桩测试宣称 GPU 性能无回归。
已知原分支 CSR policy 的测试桩导入/旧文本断言、delivery scope 的 runner 断言存在既有失败，
应与本次新增回归区分，未为通过它们更改数值策略或 runner。

本次 CPU 验证：33 passed，2 skipped；21 个修改/新增 Python 文件通过语法检查，git diff --check 通过。
另以未修改 HEAD 的临时源码快照复现 CSR policy 的同样 5 个既有失败（2 passed），不计为本次新增回归。
