# SpMV/SpMM mixed precision: vendor reference and implementation checklist

Recorded: 2026-10-20. Documentation audit only; no GPU tests performed.

## Terminology and scope

FP16 inputs/output with FP32 computation IS mixed precision. Distinguish
an algorithm's built-in mixed strategy from FlagSparse's `mixed_spmx.py`
dispatch/implementation module; the filename does not define the concept.

Scope: generic unstructured SpMV/SpMM, primarily CSR/COO/CSC. Do not infer
support for these formats from dense GEMM or structured-sparsity Lt APIs.
The queried vendor tables use FP32 computation for real FP16/BF16 paths,
including same-type output. They do not expose FP16 accumulation for these
combinations. This does not establish the exact hardware instructions or
prove that the implementation materializes full FP32 input copies.

## Sources and version boundary

- [NVIDIA cuSPARSE 13.4 current documentation](https://docs.nvidia.com/cuda/cusparse/index.html),
  sections `cusparseSpMV` and `cusparseSpMM`.
- [NVIDIA CUDA 12.5 archived cuSPARSE documentation](https://docs.nvidia.com/cuda/archive/12.5.0/cusparse/index.html),
  same sections; useful comparison with this repository's 12.5 references.
- [AMD rocSPARSE current generic API documentation](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html),
  page identifies itself as rocSPARSE 5.1.0; this is NOT a ROCm 5.1 or ROCm 6.1
  support certification.

Runtime library versions, device architecture and actual statuses must be
recorded on the compute node. Do not map these latest tables directly onto
the user's HIP 6.1.25065/DCU installation. hipSPARSE and rocSPARSE APIs are
distinct entry points; direct rocSPARSE documentation alone does not certify
that a particular hipSPARSE release exposes a corresponding combination.

## Type combinations to represent explicitly

Both queried vendor generic SpMV/SpMM tables include the following:

| Sparse A | Dense x/B | Output y/C | Compute |
| --- | --- | --- | --- |
| FP16 | FP16 | FP16 | FP32 |
| FP16 | FP16 | FP32 | FP32 |
| BF16 | BF16 | BF16 | FP32 |
| BF16 | BF16 | FP32 | FP32 |
| INT8 | INT8 | INT32 | INT32 |
| INT8 | INT8 | FP32 | FP32 |

These are listed combinations, not permission to combine arbitrary types.
SpMV additionally lists real FP32 A with FP64 x/output; real FP32/FP64 A
with corresponding complex64/complex128 x/output. AMD additionally lists
complex64 A with complex128 x/output. These SpMV entries must not be
automatically declared for SpMM. Low-precision complex entries in NVIDIA's
table are deprecated and are not proposed as a first implementation target.

## Restriction ledger

| API/version | Documented constraint or qualification |
| --- | --- |
| cuSPARSE 13.4 SpMV | CSR/CSC use CSR algorithms; COO uses COO algorithms. ALG2 deterministic behavior is restricted to non-transpose; transpose uses ALG1 behavior. This is a reproducibility distinction, not proof that transpose is unsupported. |
| cuSPARSE 13.4 SpMM | Documentation states mixed COO/CSR default to COO_ALG2/CSR_ALG3; unsupported computations on those algorithms fail. Do not assume every explicit algorithm accepts every listed type combination. |
| cuSPARSE 13.4 CSR_ALG3 | CSR only; A must be non-transposed; B conjugate-transpose unsupported; batched modes excluded. |
| cuSPARSE 12.5 CSR_ALG3 | Archived table names CSR/CSC, requires non-transposed A, excludes conjugate-transposed B. Do not replace this version's rules with the newer CSR-only wording. |
| cuSPARSE 12.5 vs 13.4 | The inspected 12.5 section does not contain the newer mixed-default algorithm note. Absence is not evidence of unrestricted support; probe exact version/algorithm combinations. |
| rocSPARSE SpMM | Mixed support is listed for CSR/CSC/COO/BSR, not Blocked ELL. B/C may independently use row or column order. |
| rocSPARSE SpMM transpose | Notes restrict COO/Blocked ELL A to non-transpose. Another passage describes a COO transpose fallback: contradictory documentation; mark COO transpose unresolved until version-specific source/runtime verification. |
| rocSPARSE SpMM CSR/CSC transpose | Documentation describes reverting to the CSR algorithm when A is transposed. Requested and actual algorithms must not be conflated. |
| rocSPARSE SpMM reproducibility | Transposed/conjugate-transposed A has no deterministic-result guarantee in the documented algorithms. |
| All baseline combinations | Validate format, A/RHS/output/compute types, opA/opB, algorithm, layout, indices, batches, device and library version together. A type enum in a wrapper is not support verification. |

The ledger is a documented starting point, not an exhaustive runtime
capability matrix. Unlisted per-combination behavior remains unverified.
Layout performance recommendations are not legality restrictions.

## FlagSparse follow-up (proposed, not implemented by this record)

1. Represent `(A dtype, RHS dtype, compute dtype, output dtype)` separately.
   Keep existing fixed FP16-to-FP32-compute strategies classified as mixed.
2. Extend type resolution and per-algorithm capabilities without changing
   existing defaults. A valid type tuple can still be unsupported by an
   algorithm/backend/op combination.
3. Add FP16/BF16 wide output to each algorithm while retaining its execution
   method. Keep split partials, reduction buffers and cross-batch sums in
   compute precision; allocate final output according to its declared type.
4. Add INT8 and heterogeneous inputs separately. Define integer overflow
   behavior, integer versus float accumulation, and complex component rules.
   Integer matmul does not imply quantization scale/zero-point support.
5. Extend existing accuracy and performance parametrization with these tuples.
   Preserve q4 coverage until all of its combinations have explicit replacements.
6. Baselines should retain native input/output types and correct scalar types;
   do not pre-cast inputs to FP32 and label the result native FP16 performance.
   Record vendor requested algorithm, known actual route, status and reason.
7. Vendor restrictions constrain baseline availability, not FlagSparse's
   permitted functionality. Unsupported vendor cases remain N/A while the
   FlagSparse operator is still tested.

Suggested result fields: library/version, backend/device, format, algorithm,
A/RHS/compute/output dtype, opA/opB, layout/strides, pointer/index dtype,
batch information, preparation boundary, correctness, status/reason.

## Direct hipSPARSE adapter source audit (2026-10-22)

The explicit type-combination benchmarks now select a direct CUDA/ROCm backend
in `tests/cusparse_generic_baseline.py`. Omitting the backend preserves the old
cuSPARSE/q4 behavior; hip-python remains available in the original benchmarks.

HIP enum values were checked independently against the
[ROCm 6.1 hipSPARSE header](https://github.com/ROCm/hipSPARSE/blob/rocm-6.1.0/library/include/hipsparse.h)
and [HIP datatype declarations](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.1/doxygen/html/library__types_8h.html).
Generic descriptors, SpMV/SpMM signatures and optional SpMM preprocessing follow
the [hipSPARSE Generic API reference](https://rocm.docs.amd.com/projects/hipSPARSE/en/docs-6.1.0/reference/generic.html).

This is a source audit, not a hardware support certification. A loaded library's
NOT_SUPPORTED status is recorded as N/A; other API failures retain their status
and traceback. No dtype/format conversion or rocSPARSE fallback is used to make
an unsupported native combination appear available. Record the runtime library
version and correctness independently of its measured time.

## 2026-11-18：单行/单列 dense 描述符修复与复测

这里有两个独立问题，不能混为“cuSPARSE 都有问题”：

| 现象 | 简单解释 | 本次处理 |
|---|---|---|
| `CreateDnMat(rhs)` 报错，尚未开始乘法 | 我们填错了部分单行矩阵的行跨度参数。比如一行4个数，PyTorch可能记录stride=(1,1)，但cuSPARSE行优先描述符的ld至少应为4 | 修复Python绑定；不是修改cuSPARSE库 |
| BF16窄输出能执行，但结果不准且很慢 | 调用已经进入计算，和前一项的“创建失败”不同。诊断已复现，但尚未确认是绑定其他参数、库算法还是版本问题 | 本次不宣称解决；继续保留vendor FAIL与有效耗时 |

`ld` 可以理解为描述符中相邻行（列优先则是相邻列）的跨度。
官方约束为：row-major的ld不小于物理列数，column-major的ld不小于物理行数。
见 [cuSPARSE 12.9 Dense Matrix Format](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html#dense-matrix-format)。
只有一行时，根本不会访问“下一行”，因此修正这个未使用的跨度不会改变任何元素的地址。
普通多行矩阵的真实跨度则必须保留，不能通过放大ld掩盖重叠或不支持的布局。

源码状态：两条已有入口（历史q4的`_dnmat`、显式CUDA/HIP组合的`dense`）共用
`_dense_matrix_layout`。保留已有合法layout/ld、数据指针、dtype、op映射、公共签名及计时范围；
不复制/转型输入，不转格式，不调整容差。空矩阵仅生成合法描述参数，不承诺厂商算子支持所有空维度组合。

测试已加入原文件：

- `tests/ci/test_spmm_baseline_gate.py`：17种合法布局检查逐元素地址一致性、9种非法布局拒绝、两入口接线。
- `tests/ci/test_runtime_policies.py`：mock C API捕获shape/order/ld/data_ptr，覆盖普通与单例布局、算法/预处理参数和历史q4入口。
- `tests/pytest/test_spmm_csc_accuracy.py`：`test_spmm_csc_native_vendor_singleton_descriptor`。
  quick为四种半精度组合×九种opA/opB×两种布局，共72例；normal每例再测N=7。
  样本数值小且精确可表示，隔离长归约的BF16窄输出问题。baseline状态独立写入JSON property；
  ERROR/数值失败必须FAIL，仅明确不支持/库不可用允许SKIP。

**本地仅做AST及diff审查；下列测试未在本地执行。修复源码完成不等于硬件验收通过。**

在已分配的CUDA计算节点、仓库根目录执行，继续使用现有容器/venv脚本：

```bash
RESULT_ROOT="$PWD/11.18data/cusparse_descriptor_retest"
mkdir -p "$RESULT_ROOT"

# 1. 原runner下的72例baseline准确性回归；不依赖真实矩阵目录。
bash tools/run_cuda_tests.sbatch --repo "$PWD" -- \
  --ops spmm_csc --phase accuracy --mode quick \
  --pytest-args "-k native_vendor_singleton_descriptor -x" \
  --results-dir "$RESULT_ROOT/accuracy"

# 2. 原CSC benchmark的144条件诊断；与前次矩形数据对照。
bash tools/run_cuda_tests.sbatch --repo "$PWD" \
  --entry tests/test_spmm_csc.py -- \
  --vendor-diagnostic rectangular \
  --type-combos float16:float16:float16,float16:float16:float32,bfloat16:bfloat16:bfloat16,bfloat16:bfloat16:float32 \
  --alg spmm_csc_base --ops all --op-b all --layout all \
  --dense-cols 1,32 --index-dtypes int32 --indptr-dtypes int32 \
  --vendor-alg default --vendor-preprocess off \
  --warmup 3 --iters 10 --timing --perf-metrics \
  --csv-csc "$RESULT_ROOT/baseline_rectangular.csv"
```

没有`--no-vendor`，所以保留直接cuSPARSE。第二条故意不加`--synthetic`，只运行命名诊断样本；
也不扫描FlagSparse全部算法，避免重复工作。它验证baseline和小样本计时链路，不代表真实矩阵性能改善。
`--timing`仍独立诊断，不改变完整时间`ms=process_cpu_ms+gpu_ms`或采样过滤规则。

验收先看`accuracy/spmm_csc/accuracy_result.json`和stdout/stderr，再看CSV的
`vendor_status/vendor_reason/vendor_error_stage/vendor_ms`。原来48条`CreateDnMat(rhs)`错误应该消失；
若后续阶段产生新错误，仍需按该阶段排查，不能转成PASS。若只有vendor数值FAIL，须和描述符修复分别归因。

可选：在同一已激活的容器/venv中运行CPU策略与mock回归（这些命令不在登录节点执行）：

```bash
python -m pytest tests/ci/test_spmm_baseline_gate.py -k native_dnmat -q
FLAGSPARSE_TRITON_SMOKE=1 python -m pytest tests/ci/test_runtime_policies.py \
  -k cuda_vendor_algorithm_and_preprocess_reach_c_api -q
```

回传时保留整个结果目录（含`.vendor_logs`、JSON与CSV），不要只给summary.csv。
