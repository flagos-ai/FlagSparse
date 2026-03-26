# tests/pytest：参数化用例说明 / Parametrized test notes

---

## 中文

### 用例名里的方括号 `[…]` 是什么

形如 `test_xxx[float32-1024-256]` 的片段是 **pytest 自动生成的用例实例 ID**：把这一次 **`@pytest.mark.parametrize` 选用的实参** 用 `-` 拼成一段，便于区分笛卡尔积里的某一组。

- **dtype**：`FLOAT_DTYPE_IDS` 用于 gather/scatter 与 CSR SpMV；COO SpMV 使用 `SPMV_COO_DTYPE_IDS`（仅 `float32` / `float64`）。
- **数字段**：对应当前测试里形状或阶数参数，含义见下文。

参数列表定义在 `param_shapes.py`；`--mode quick|normal` 在 `conftest.py` 中切换 **QUICK_MODE**，会缩短形状列表。

---

### 合成数据生成规则（CUDA）

全部参数化用例在 **GPU** 上构造张量（`torch.device("cuda")`），**不**读 `.mtx`、ODPS 或外部矩阵文件。规则与对应 `test_*_accuracy.py` 一致，摘要如下：

- **公共**：掩码采样概率（CSR / COO SpMV 共用）对形状 `(rows, cols)` 取 `p = min(0.25, max(0.06, 32 / max(rows*cols, 1)))`；`mask = (torch.rand(rows, cols, device=cuda) < p)`；若 `mask` 全假则强制 `mask[0, 0] = True`；非零处值为 `randn(rows, cols, dtype, device=cuda) * mask`。
- **CSR SpMV**：由上式得到稠密阵后 `to_sparse_csr()` 得 **A**；`x = randn(N, dtype, device=cuda)`；参照 `torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)`；被测 `flagsparse_spmv_csr(...)`。
- **COO SpMV**：**同一掩码与 `randn*mask` 的稠密阵**再 `to_sparse_coo()`；`x` 与参照同上（`torch.sparse.mm` 作用于该 COO 张量）；仅 **float32 / float64**；被测 `flagsparse_spmv_coo(...)`。
- **Gather**：`dense = randn(dense_size, dtype, device=cuda)`；`nnz_eff = min(nnz, dense_size)`；`indices = randperm(dense_size, device=cuda)[:nnz_eff].to(int32)`；参照为 `dense[indices.to(int64)]`。
- **Scatter**：`vals = randn(nnz_eff, ...)`，索引规则与 gather 相同；输出初值为全零；参照为对同一下标的 `index_copy_`。
- **SpSV（CSR）**：`base = tril(randn(n, n, dtype, device=cuda))`，`A = base + eye(n) * (n/2 + 2)`（加强对角），`b = randn(n, ...)`；`A_csr = A.to_sparse_csr()`；参照 `torch.linalg.solve_triangular(A, b.unsqueeze(-1), upper=False).squeeze(-1)`；被测 `flagsparse_spsv_csr(...)`（仅下三角、非单位对角）。

容差：`gather`/`scatter` 用 `torch.equal`；SpMV / SpSV 用 `torch.allclose`，阈值见各测试文件中的 `_tol` 或固定 `rtol`/`atol`。

---

### `test_gather_matches_indexing`

**ID 格式**：`[dtype-dense_size-nnz]`

| 段 | 含义 |
|----|------|
| dtype | 稠密向量与输出流的数据类型 |
| dense_size | 被索引的稠密向量长度 |
| nnz | 索引条数（gather 输出长度）；代码里会做 `min(nnz, dense_size)` |

**形状来源**：`GATHER_SCATTER_SHAPES` × `FLOAT_DTYPES`。

---

### `test_scatter_matches_index_copy`

**ID 格式**：与 gather 相同 — `[dtype-dense_size-nnz]`。

测的是 **scatter**（覆盖写）与 `index_copy_` 的一致性；索引为随机 **不重复** 子集。

**形状来源**：同上。

---

### `test_spmv_csr_matches_torch`

**ID 格式**：`[dtype-M-N]`

| 段 | 含义 |
|----|------|
| dtype | 稀疏矩阵与向量 **x** 的数据类型 |
| M | **A** 行数，**y** 的长度 |
| N | **A** 列数，**x** 的长度（`y = A @ x`） |

**形状来源**：`SPMV_MN_SHAPES` × `FLOAT_DTYPES`。

---

### `test_spmv_coo_matches_torch`

**ID 格式**：`[dtype-M-N]`（与 CSR SpMV 相同）

| 段 | 含义 |
|----|------|
| dtype | 仅 **float32 / float64**（与 Triton COO SpMV 核一致；见 `SPMV_COO_DTYPES`） |
| M | **A** 行数，**y** 长度 |
| N | **A** 列数，**x** 长度 |

**形状来源**：`SPMV_MN_SHAPES` × `SPMV_COO_DTYPES`。数据由与 CSR 相同的掩码规则得到稠密矩阵后 `to_sparse_coo()`。

---

### `test_spsv_csr_lower_matches_dense`

**ID 格式**：`[dtype-n]`（dtype 在测试里显式写为 `float32` / `float64`）

| 段 | 含义 |
|----|------|
| dtype | 仅 **float32 / float64**（`flagsparse_spsv_csr` / CSR） |
| n | 方阵阶数：**A** 为 `n×n` 下三角 CSR，**b** 长度 `n`，解 **A x = b** |

**形状来源**：`SPSV_N` × `{float32, float64}`。**仅 CSR**：无 COO SpSV 参数化用例。

---

## English

### What the `[…]` suffix means

Strings like `test_xxx[float32-1024-256]` are **pytest case instance IDs**: the **concrete arguments** chosen by `@pytest.mark.parametrize` for that run, joined with `-`, so you can tell one combination in the Cartesian product from another.

- **dtype**: `FLOAT_DTYPE_IDS` for gather/scatter and CSR SpMV; COO SpMV uses `SPMV_COO_DTYPE_IDS` (`float32` / `float64` only).
- **Numeric segments**: Shape or order parameters for that test (see below).

Grids live in `param_shapes.py`. `--mode quick|normal` in `conftest.py` toggles **QUICK_MODE** (fewer shapes).

---

### Synthetic data construction (CUDA)

All parametrized tests build tensors on **`torch.device("cuda")`** and do **not** read `.mtx`, ODPS, or external matrix files. This matches the `test_*_accuracy.py` sources; summary:

- **Shared SpMV mask** (CSR & COO): for shape `(rows, cols)`, `p = min(0.25, max(0.06, 32 / max(rows*cols, 1)))`; `mask = (torch.rand(rows, cols, device=cuda) < p)`; if the mask is all false, set `mask[0, 0] = True`; values are `randn(rows, cols, dtype, device=cuda) * mask`.
- **CSR SpMV**: dense field → `to_sparse_csr()` → **A**; `x = randn(N, dtype, device=cuda)`; reference `torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)`; DUT `flagsparse_spmv_csr(...)`.
- **COO SpMV**: **same** masked dense field → `to_sparse_coo()`; same `x` / reference path with `torch.sparse.mm` on that COO tensor; dtypes **float32 / float64** only; DUT `flagsparse_spmv_coo(...)`.
- **Gather**: `dense = randn(dense_size, dtype, device=cuda)`; `nnz_eff = min(nnz, dense_size)`; `indices = randperm(dense_size, device=cuda)[:nnz_eff].to(int32)`; reference fancy indexing on `dense`.
- **Scatter**: `vals = randn(nnz_eff, ...)` with the same index rule; output starts at zeros; reference `index_copy_` on the same indices.
- **SpSV (CSR)**: `base = tril(randn(n, n, dtype, device=cuda))`, `A = base + eye(n) * (n/2 + 2)`, `b = randn(n, ...)`; `A_csr = A.to_sparse_csr()`; reference `torch.linalg.solve_triangular(A, b.unsqueeze(-1), upper=False).squeeze(-1)`; DUT `flagsparse_spsv_csr(...)` (lower-triangular, non-unit diagonal).

Checks: `gather` / `scatter` use `torch.equal`; SpMV / SpSV use `torch.allclose` (see per-file `_tol` or fixed rtol/atol).

---

### `test_gather_matches_indexing`

**ID pattern**: `[dtype-dense_size-nnz]`

| Segment | Meaning |
|---------|---------|
| dtype | Dtype of the dense vector and gathered stream |
| dense_size | Length of the indexed dense vector |
| nnz | Number of indices (output length); code uses `min(nnz, dense_size)` |

**Source**: `GATHER_SCATTER_SHAPES` × `FLOAT_DTYPES`.

---

### `test_scatter_matches_index_copy`

**ID pattern**: same as gather — `[dtype-dense_size-nnz]`.

Checks **scatter** (overwrite) vs `index_copy_` with a random **unique** index set.

**Source**: same as gather.

---

### `test_spmv_csr_matches_torch`

**ID pattern**: `[dtype-M-N]`

| Segment | Meaning |
|---------|---------|
| dtype | Dtype of **A** and **x** |
| M | Rows of **A**, length of **y** |
| N | Columns of **A**, length of **x** (`y = A @ x`) |

**Source**: `SPMV_MN_SHAPES` × `FLOAT_DTYPES`.

---

### `test_spmv_coo_matches_torch`

**ID pattern**: `[dtype-M-N]` (same as CSR SpMV)

| Segment | Meaning |
|---------|---------|
| dtype | **float32 / float64** only (Triton COO SpMV kernels; see `SPMV_COO_DTYPES`) |
| M | Rows of **A**, length of **y** |
| N | Columns of **A**, length of **x** |

**Source**: `SPMV_MN_SHAPES` × `SPMV_COO_DTYPES`. Matrix is the same masked dense field as CSR SpMV, then `to_sparse_coo()`.

---

### `test_spsv_csr_lower_matches_dense`

**ID pattern**: `[dtype-n]` (dtype ids are explicitly `float32` / `float64`)

| Segment | Meaning |
|---------|---------|
| dtype | **float32 / float64** only (`flagsparse_spsv_csr`, CSR storage) |
| n | Square order: lower-triangular **CSR** `n×n`, **b** length `n`, solve **A x = b** |

**Source**: `SPSV_N` × `{float32, float64}`. **CSR SpSV only**; no COO SpSV parametrized test.

---
