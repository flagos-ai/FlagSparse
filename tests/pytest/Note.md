# tests/pytest 参数化用例说明 / Parametrized Test Notes

## 中文

### 运行方式

`tests/pytest` 是面向算子正确性的 CUDA pytest 集合。形状和 dtype 网格主要来自 `param_shapes.py`，`--mode quick|normal` 由 `conftest.py` 切换。

常用命令：

```bash
pytest tests/pytest --mode quick -m "spmv_csr or spmm_csr"
python run_flagsparse_pytest.py --mode quick --ops gather,scatter,spmv_csr --gpus 0
```

`run_flagsparse_pytest.py` 按算子 marker 单独启动 pytest，设置每个子进程的 `CUDA_VISIBLE_DEVICES`，保存每个算子的 `accuracy.log`，并输出 `summary.json`、`summary.csv` 和可选的 `summary.xlsx`。如果子进程非正常退出且没有 pytest summary，runner 会标记为 `CRASH`。

### 数据构造

所有参数化正确性用例都在 `torch.device("cuda")` 上构造合成数据，不依赖 `.mtx`、ODPS 或外部矩阵文件。

- Gather/Scatter：使用 `GATHER_SCATTER_SHAPES` 和 `FLOAT_DTYPES`，参考分别为 PyTorch indexing 与 `index_copy_`。
- CSR/COO SpMV：使用合成稀疏矩阵，参考为 `torch.sparse.mm`。
- CSR/COO SpMM、SpSM、SpGEMM、SDDMM：使用小规模合成矩阵，参考为 PyTorch dense/sparse 运算或对应的采样 dense reference。
- SpSV：使用加强对角的三角矩阵，参考为 `torch.linalg.solve_triangular`；可选 CuPy/cuSPARSE reference 仅在依赖可用时运行。

### SPSV 覆盖范围

`test_spsv_csr_accuracy.py` 当前覆盖：

- CSR baseline：`float32` / `float64` lower triangular、non-unit diagonal。
- CSR non-trans 必过矩阵：`float32` / `float64` / `complex64` / `complex128`，索引为 `int32` 与 `int64`。
- CSR transpose：`float32` / `float64` / `complex64` / `complex128`，索引为 `int32` 与 `int64`。
- CSR CuPy/cuSPARSE 对照：non-trans 不包含 `complex64`；transpose 包含 `complex64`；依赖不可用时 skip。
- COO：`float32` / `float64`，覆盖 `auto` / `direct` / `csr` mode；另测 unsorted+duplicate COO 在 `auto` 下回退、在 `direct` 下拒绝。
- 当前 pytest 不修改或倒逼 SPSV 算子能力；`complex64 non-trans`、bf16、`unit_diagonal=True` 和 2D RHS 不作为本轮必过项。

## English

### Running

`tests/pytest` is the CUDA accuracy suite for FlagSparse operators. Shape and dtype grids live mostly in `param_shapes.py`; `--mode quick|normal` is handled by `conftest.py`.

Common commands:

```bash
pytest tests/pytest --mode quick -m "spmv_csr or spmm_csr"
python run_flagsparse_pytest.py --mode quick --ops gather,scatter,spmv_csr --gpus 0
```

`run_flagsparse_pytest.py` launches one pytest subprocess per operator marker, sets per-process `CUDA_VISIBLE_DEVICES`, writes per-op `accuracy.log`, and summarizes to `summary.json`, `summary.csv`, and optional `summary.xlsx`. If a subprocess exits abnormally without a pytest summary, the runner reports `CRASH`.

### Data Construction

All parametrized accuracy tests build synthetic tensors on `torch.device("cuda")`; they do not read `.mtx`, ODPS, or external matrix files.

- Gather/Scatter use `GATHER_SCATTER_SHAPES` and `FLOAT_DTYPES`; references are PyTorch indexing and `index_copy_`.
- CSR/COO SpMV use synthetic sparse matrices; references use `torch.sparse.mm`.
- CSR/COO SpMM, SpSM, SpGEMM, and SDDMM use small synthetic matrices and PyTorch dense/sparse or sampled dense references.
- SpSV uses diagonally strengthened triangular matrices; references use `torch.linalg.solve_triangular`; optional CuPy/cuSPARSE references run only when available.

### SPSV Coverage

`test_spsv_csr_accuracy.py` currently covers:

- CSR baseline: `float32` / `float64` lower triangular, non-unit diagonal.
- CSR non-trans required matrix: `float32` / `float64` / `complex64` / `complex128` with `int32` and `int64` indices.
- CSR transpose: `float32` / `float64` / `complex64` / `complex128` with `int32` and `int64` indices.
- CSR CuPy/cuSPARSE reference: non-trans excludes `complex64`; transpose includes `complex64`; skipped when dependencies are unavailable.
- COO: `float32` / `float64`, covering `auto` / `direct` / `csr` modes; unsorted+duplicate COO is checked for auto fallback and direct-mode rejection.
- This pytest suite does not modify or force new SPSV operator capability; `complex64` non-trans, bf16, `unit_diagonal=True`, and 2D RHS are not required pass cases in this iteration.
