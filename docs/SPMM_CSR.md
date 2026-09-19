# Registered native CSR SpMM algorithms

## Status and invocation

The four extensions are implemented but **unverified on hardware**. No operator,
pytest, interpreter or offline kernel compilation was run on the development host.
The historical high-level default and registry `auto -> csr_base` are unchanged.
Use an explicit algorithm to exercise an extension:

```python
p = flagsparse.prepare_spmm_csr_route(values, columns, pointers, shape,
                                    op="conj", alg="csr_adaptive_tile_split")
C, meta = flagsparse.flagsparse_spmm_csr_run(p, B, return_meta=True)
```

`get_spmm_csr_algorithm_spec(name)` describes declared dtype/op/layout/backend
coverage. `list_spmm_csr_algorithms(op=None, dtype=None, backend=None, layout=None)`
filters declarations, not hardware validation results. Runtime capability checks
can still reject a declared combination. The prepared route fixes `op`; algorithms
may change between calls. Configurations belong to the algorithm specified during
prepare and are not inherited by a different algorithm. A run-level configuration
replaces the prepared override dictionary; unspecified fields use resolved defaults.

The high-level API accepts keyword `alg`, `config`, `prepared`, `timing` and
`dense_layout`; legacy calls without these dispatch options retain their behavior.
With `prepared`, pass `None` for raw sparse arrays/shape. Legacy block tuning
arguments cannot be mixed with the registered route configuration.

## Algorithms and numerical policy

| Name | Execution |
| --- | --- |
| `csr_row_tile` | Simultaneous `[R,BK,BN]` multirow tile; masked local row-length loop |
| `csr_row_kparallel` | One row/output tile; lane accumulators reduced after the row loop |
| `csr_split_nnz_reduce` | Cross-program segments, exclusive partial sums, bounded multilevel reduction |
| `csr_adaptive_tile_split` | Per-call GPU classification and the three mutually exclusive paths above |

Every name returns the complete output. FP32 and complex64 use FP32 arithmetic;
FP64 and complex128 use FP64 arithmetic. Complex products use explicit real and
imaginary components. No new floating-point atomics, precision promotion or relaxed
tolerances are introduced. Existing accuracy/HP algorithms retain their policy.
FP16/BF16 are not supported by these four extensions.

For `trans`, the operation is `A.T @ B`; `conj` is `A.conj().T @ B` and never
conjugates B. Each run builds a transposed CSR with stable sorting, preserving
duplicate entries and explicit zeros. Plans use the resulting row lengths. No
transpose or algorithm plan is cached on prepared. This shared preparation change
also affects TLE callers; TLE kernels themselves are unchanged and excluded from
this work's acceptance sweep.

Input column/pointer types may independently be int32 or int64. The existing
validated int32 execution-column restriction remains; pointers and address
arithmetic retain int64 width. Transposed columns are range-checked again.

## Configuration and heterogeneous backends

The pure Python `_spmm_csr_config.py` separates algorithm declarations from device
discovery and kernels. Canonical names are cuda, rocm, metax, mthreads, ascend, xpu,
gcu and mlu. Actual target/architecture and launch capabilities are recorded
separately. A CUDA-compatible API name alone does not prove NVIDIA capabilities.

Defaults: tile R/BK/BN = 4/8/16, row and segment BK/BN = 32/32, segment length
1024, reduction fan-in 256, four warps, one stage, one column panel. complex128
halves BN. Short thresholds are 32 for row-layout N<=32, otherwise 16; long
thresholds are two segments, or four for N>=128. These are reproducible starting
points, not measured optimal configurations.

Overrides use flat integer fields: `tile_rows`, `tile_k`, `tile_n`, `block_k`,
`block_n`, `panels`, `segment_nnz`, `reduce_block_size`, `num_warps`, `num_stages`,
`workspace_bytes`, `short_row_threshold`, `split_row_threshold`. Explicit invalid
parameters raise; they are never clamped. Architecture profiles precede backend
profiles and conservative defaults. Rejected built-in profiles are reported.
Profiles can specialize an algorithm with `(backend, algorithm)` or
`(backend, architecture, algorithm)` keys; such overrides do not leak to siblings.

Other backends have candidate configuration entries and capability-gated runtime
dispatch, not a claim of verified support. Missing launch, FP64, stable-sort or
scan capability information produces an explicit unavailable reason. In
particular, Ascend's historical high-level fallback is not used under a new name.
Compiler errors are failures and must be diagnosed on the target. Verified device
profiles should be added only with device/compiler/correctness evidence.

The numeric scratch cap defaults to 256 MiB. Split computation iterates over column
waves and segment batches, including batches within a single giant row. Reduction
buffers remain in component precision; batches add to their unique output rows on
one stream without atomics. The cap excludes the required output, input tensors,
transpose storage and integer descriptors. Numeric scratch peak and a descriptor
space estimate are reported separately. Allocation errors are not algorithm fallback.

## Timing and results

Always `ms = process_cpu_ms + gpu_ms`, including with `--timing`. Full-run backend
events include transpose, GPU plans, necessary internal conversions, initialization,
products, reduction and output handling. GPU readback/waiting and Python launch
dictionaries are not CPU algorithm work. New routes report zero process_cpu_ms.

Phase diagnostics execute separately. `process_gpu_ms` covers transpose and plan
construction; `compute_ms` covers numeric work including initialization/reduction.
Their sum does not replace the complete event time. `run(timing=True)` performs
one undiagnosed full call and an additional diagnostic call. The benchmark's warmup,
JIT (including one mandatory cold call when warmup=0), input loading, correctness reference and vendor descriptor creation are outside
the measured calls. Old CUDA transpose timings with prepare-time caching are not
comparable to these results.

```bash
python tests/test_spmm_csr.py ../matrix --alg compare --exclude-tle \
  --dtypes float32,float64,complex64,complex128 --ops all --layout all \
  --index-dtypes int32,int64 --indptr-dtypes int32,int64 \
  --dense-cols 1,8,16,32,64,128,256 --warmup 10 --iters 50 --timing \
  --csv-csr ./results/spmm_csr_compare.csv
```

Use `--synthetic` in place of a matrix directory for short-row, heavy-tail and empty
cases. `compare` aliases `all`; auto is not duplicated. Legacy singular flags, `--csv`
and vendor-disable aliases remain accepted. Unsupported configurations print reasons.
Each algorithm result is flushed to CSV immediately. FAIL rows do not enter speedup
or best-algorithm selection. The vendor result is also checked against the independent
correctness reference. Unsupported same-device CSR vendor operations remain N/A.
The terminal prints one environment line and a compact shared header/result table:
matrix, dtype, index/pointer types, op, layout, N, algorithm, full-call ms, vendor ms,
speedup (`x = vendor_ms / ms`), native Check and vendor VCheck. `--timing` keeps the
same terminal columns; phase timings, errors, configuration and software versions
remain in CSV. Accuracy failures mark FAIL and continue unless `--fail-fast` is set.
Compilation, interface and execution exceptions propagate with a traceback;
previously completed CSV rows remain flushed. Known unsupported combinations print
deduplicated reasons. A completed measurement remains visible
in `vendor_ms`, its compatibility alias `cusparse_ms`, and `vendor_raw_ms` even
when vendor correctness fails. `vendor_status=FAIL` and `vendor_reason` distinguish
that case from an unavailable interface. Speedup requires both native and vendor
correctness to pass; failed vendor output is never used as a valid speedup baseline.
Historical `torch_ms`/speedup fields remain empty: other-format correctness references
are not performance baselines. All actual parameters are saved in the metadata column.

Compute-node validation:

```bash
pytest tests/ci/test_spmm_csr_policy.py
pytest tests/pytest/test_spmm_csr_accuracy.py -k 'new or split_workspace or per_run or transpose_classifies'
python tests/test_spmm_csr.py --synthetic --alg compare --exclude-tle \
  --dtypes all --ops all --layout all --dense-cols 1,32,128 --timing \
  --csv-csr ./results/spmm_csr_synthetic.csv
```

Repeat on each available backend, recording software/device versions, correctness,
full-call latency and metadata. Matrix-suite speedups remain unmeasured.

## 中文说明

四个新算法已接入注册入口，当前状态均为“尚未实机验证”。默认路由不变。
支持 float32/float64/complex64/complex128 和 non/trans/conj；FP32 与 complex64
采用 FP32 分量精度，不沿用 SpMV 的 FP32→FP64 策略。

短行采用真正多行并行，中长行采用行内并行与延迟归约，超长行跨 program 分段并
分层归并；组合算法每次调用重新分类。分段数值缓冲默认限制 256 MiB，超限时按输出列
及分段分批，整数描述符和转置矩阵不属于该数值缓冲额度。

所有注册路由按次构建转置 CSR；开启阶段诊断仍使用完整调用的 gpu_ms 计算总耗时。
旧 CUDA prepare 缓存转置的成绩需要重新测量。计数读回、设备等待及 launch 字典不计
CPU 算法处理时间。阶段计时单独执行，不替代完整调用时间。

多平台通过规范后端名、实际编译目标及能力选择配置；缺少能力信息时明确不可用，
不借用旧回退冒充新算法。所有平台均需分别完成实机验证。上面的 compare 命令是完整
比较入口，逐条写入结果，正确性失败不参与加速比及最佳算法排名。
