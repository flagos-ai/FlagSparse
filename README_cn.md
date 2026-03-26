# FlagSparse

GPU 稀疏运算库（SpMV、gather、scatter、多种稀疏格式）。

## 安装

```bash
pip install . --no-deps --no-build-isolation
```

离线时可加 `--no-build-isolation` 避免拉取构建依赖。

运行时依赖（按需安装）：

```bash
pip install torch triton cupy-cuda12x
```

## 目录说明

- `src/flagsparse/` — 核心包（`sparse_operations/` 由 `flagsparse.py` 内嵌字符串生成多个 `.py`）
- `tests/` — pytest 测试
- `benchmark/` — 性能基准

## 测试用法

在项目根目录执行，或先 `cd tests` 再执行（.mtx 目录可用 `../matrix` 等相对路径）。

**test_spmv.py** — CSR SpMV（SuiteSparse .mtx、合成数据或导出 CSV）：

```bash
python tests/test_spmv.py <目录或文件.mtx>               # 批量跑，默认 float32
python tests/test_spmv.py <目录/> --dtype float64        # 可选：--index-dtype int32、--warmup 10、--iters 50、--no-cusparse
python tests/test_spmv.py --synthetic                    # 合成数据基准
python tests/test_spmv.py <目录/> --csv-csr results.csv  # 全 dtype，导出 CSV
```

**test_spmv_coo.py** — COO SpMV：

```bash
python tests/test_spmv_coo.py --synthetic                # 合成数据
python tests/test_spmv_coo.py <目录/> --csv-coo out.csv  # .mtx 批量，导出 CSV
```

**test_spsv.py** — SpSV CSR（三角求解，仅方阵）：

```bash
python tests/test_spsv.py --synthetic                     # 合成数据，对比 PyTorch / CuPy
python tests/test_spsv.py <目录/> --csv-csr spsv_csr.csv # .mtx 批量，导出 CSV（f32/f64、int32）
```

**test_spsv_coo.py** — SpSV COO（流程与 CSR 测试一致）：

```bash
python tests/test_spsv_coo.py --synthetic
python tests/test_spsv_coo.py <目录/> --csv-coo spsv_coo.csv   # 可选 --coo-mode auto|direct|csr
```

**test_spmm.py** — CSR SpMM（SuiteSparse .mtx、合成基准、CSV；右端稠密列数用 `--dense-cols`）：

```bash
python tests/test_spmm.py <文件.mtx> [更多.mtx ...]     # 或含 *.mtx 的目录
python tests/test_spmm.py <目录/> --dtype float32       # 可选：--index-dtype、--dense-cols、--warmup、--iters
python tests/test_spmm.py --synthetic                    # 合成全覆盖（含 API 校验与 ALG1 分块覆盖）
python tests/test_spmm.py <目录/> --csv results.csv      # 对所有矩阵跑 f32/f64 + int32，导出单个 CSV
# 调参：--block-n、--block-nnz、--max-segments  |  关闭基线：--no-cusparse
# 仅合成：--skip-api-checks、--skip-alg1-coverage
```

**test_spmm_coo.py** — COO SpMM（参数形态与 CSR 版类似；原生 row-run / atomic / 对比模式）：

```bash
python tests/test_spmm_coo.py <文件.mtx> ...             # 或目录
python tests/test_spmm_coo.py --synthetic                # 可选：--route rowrun|atomic|compare
python tests/test_spmm_coo.py <目录/> --csv out.csv      # CSV 时 --route 仅支持 rowrun 或 atomic（不可用 compare）
# --block-nnz 为 COO 分块；合成可加 --skip-coo-coverage；其余与 test_spmm.py 类似
```

**test_gather.py** / **test_scatter.py** — gather/scatter 基准（pytest 或 `python tests/test_gather.py`）。
