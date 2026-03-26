# FlagSparse

GPU sparse operations package (SpMV, gather, scatter, sparse formats).

## Install

```bash
pip install . --no-deps --no-build-isolation
```

Use `--no-build-isolation` to avoid downloading build deps when offline.

Runtime dependencies (install when needed):

```bash
pip install torch triton cupy-cuda12x
```

## Layout

- `src/flagsparse/` — core package (`sparse_operations/` is emitted as several `.py` modules from string literals in `flagsparse.py`)
- `tests/` — pytest tests
- `benchmark/` — performance benchmarks

## Tests

Run from project root, or `cd tests` then run scripts (paths like `../matrix` for .mtx dir).

**test_spmv.py** — CSR SpMV (SuiteSparse .mtx, synthetic, or CSV export):

```bash
python tests/test_spmv.py <dir_or_file.mtx>              # batch run, default float32
python tests/test_spmv.py <dir/> --dtype float64         # optional: --index-dtype int32, --warmup 10, --iters 50, --no-cusparse
python tests/test_spmv.py --synthetic                    # synthetic benchmark
python tests/test_spmv.py <dir/> --csv-csr results.csv   # all dtypes, export CSV
```

**test_spmv_coo.py** — COO SpMV:

```bash
python tests/test_spmv_coo.py --synthetic                # synthetic
python tests/test_spmv_coo.py <dir/> --csv-coo out.csv   # .mtx batch, export CSV
```

**test_spsv.py** — SpSV CSR (triangular solve, square matrices only):

```bash
python tests/test_spsv.py --synthetic                     # synthetic vs PyTorch / CuPy
python tests/test_spsv.py <dir/> --csv-csr spsv_csr.csv  # .mtx batch, export CSV (f32/f64, int32)
```

**test_spsv_coo.py** — SpSV COO (same workflow as CSR test):

```bash
python tests/test_spsv_coo.py --synthetic
python tests/test_spsv_coo.py <dir/> --csv-coo spsv_coo.csv   # optional: --coo-mode auto|direct|csr
```

**test_spmm.py** — CSR SpMM (SuiteSparse .mtx, synthetic, CSV; dense RHS width via `--dense-cols`):

```bash
python tests/test_spmm.py <file.mtx> [<more.mtx> ...]   # or a directory of *.mtx
python tests/test_spmm.py <dir/> --dtype float32        # optional: --index-dtype, --dense-cols, --warmup, --iters
python tests/test_spmm.py --synthetic                   # full synthetic (API checks + ALG1 tile coverage)
python tests/test_spmm.py <dir/> --csv results.csv      # float32/float64 + int32 on all matrices → one CSV
# tuning: --block-n, --block-nnz, --max-segments  |  skip baselines: --no-cusparse
# synthetic only: --skip-api-checks, --skip-alg1-coverage
```

**test_spmm_coo.py** — COO SpMM (same CLI shape as CSR; native row-run / atomic / compare):

```bash
python tests/test_spmm_coo.py <file.mtx> ...            # or directory
python tests/test_spmm_coo.py --synthetic               # optional: --route rowrun|atomic|compare
python tests/test_spmm_coo.py <dir/> --csv out.csv      # --route must be rowrun or atomic (not compare)
# --block-nnz (COO tile), --skip-coo-coverage for synthetic; other flags align with test_spmm.py
```

**test_gather.py** / **test_scatter.py** — gather/scatter benchmarks (run with pytest or `python tests/test_gather.py`).
