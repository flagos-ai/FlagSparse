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
