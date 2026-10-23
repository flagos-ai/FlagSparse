# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native Generic API baselines for the benchmark scripts.

Explicit CSR/COO/CSC type-combination benchmarks select ``backend="cuda"``
or ``backend="rocm"`` for direct cuSPARSE or hipSPARSE calls. Legacy callers
that omit backend retain their original cuSPARSE-only behavior.

SpVV, Axpby and SELL SpMV for the q4 operators, plus CSR/COO/CSC SpMV, CSR/COO SpMM
and CSR SDDMM for the q4 *variants* of existing operators: the mixed-precision and
integer type combinations (f16 -> f32, i8 -> i32, real A x complex x, ...) that
the CuPy-based baselines of the existing scripts cannot express.

ctypes over the generic API, so no CuPy build is needed (CuPy wraps none of these).
Descriptors are created once in ``prepare``; ``run`` issues only the library call,
so the timed window holds the same work as FlagSparse's run step.

Only CUDA: on other backends ``skip_reason()`` names why, and the benchmark writes
an empty ``cusparse_ms`` -- the row then has no vendor baseline, which is what
``tools/baseline_bound.py`` is for.
"""

import ctypes
import ctypes.util

import torch

_CUDA_TYPES = {
    torch.float16: 2,
    torch.bfloat16: 14,
    torch.float32: 0,
    torch.float64: 1,
    torch.complex64: 4,
    torch.complex128: 5,
    torch.int8: 3,
    torch.int32: 10,
}
_INDEX_32I = 2
_INDEX_64I = 3
_BASE_ZERO = 0
_OP = {"non": 0, "trans": 1, "conj": 2}
_SPMV_SELL_ALG1 = 5
_ALG_DEFAULT = 0
_ORDER_COL, _ORDER_ROW = 1, 2

_LIB = None
_LIB_ERROR = None
_HANDLE = None

_P = ctypes.c_void_p
_PP = ctypes.POINTER(ctypes.c_void_p)
_I = ctypes.c_int
_I64 = ctypes.c_int64
_SIGNATURES = {
    "cusparseCreate": [_PP],
    "cusparseSetStream": [_P, _P],
    "cusparseCreateSpVec": [_PP, _I64, _I64, _P, _P, _I, _I, _I],
    "cusparseDestroySpVec": [_P],
    "cusparseCreateDnVec": [_PP, _I64, _P, _I],
    "cusparseDestroyDnVec": [_P],
    "cusparseCreateSlicedEll": [_PP, _I64, _I64, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseDestroySpMat": [_P],
    "cusparseSpVV_bufferSize": [_P, _I, _P, _P, _P, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpVV": [_P, _I, _P, _P, _P, _I, _P],
    "cusparseAxpby": [_P, _P, _P, _P, _P],
    "cusparseSpMV_bufferSize": [_P, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpMV": [_P, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseCreateCsr": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseCreateCsc": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseCreateCoo": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I],
    "cusparseCreateDnMat": [_PP, _I64, _I64, _I64, _P, _I, _I],
    "cusparseCreateBlockedEll": [_PP, _I64, _I64, _I64, _I64, _P, _P, _I, _I, _I],
    "cusparseDestroyDnMat": [_P],
    "cusparseSpMM_bufferSize": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpMM": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseSDDMM_bufferSize": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSDDMM_preprocess": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseSDDMM": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
}


class _GenericLibrary:
    """Bind each generic API entry point only when its operation needs it.

    For example, an installation without SELL must still be usable for CSR
    SpMV. Missing operation symbols are distinct from a missing shared library.
    """

    def __init__(self, library):
        self.library = library

    def __getattr__(self, name):
        if name not in _SIGNATURES:
            raise AttributeError(name)
        try:
            function = getattr(self.library, name)
        except AttributeError as exc:
            raise NotImplementedError(f"cuSPARSE library does not provide {name}") from exc
        function.argtypes = _SIGNATURES[name]
        function.restype = _I
        setattr(self, name, function)
        return function


def _lib():
    global _LIB, _LIB_ERROR
    if _LIB is not None:
        return _LIB
    if _LIB_ERROR is not None:
        raise RuntimeError(_LIB_ERROR)
    names = [ctypes.util.find_library("cusparse"), "libcusparse.so.12", "libcusparse.so"]
    for name in filter(None, names):
        try:
            lib = _GenericLibrary(ctypes.CDLL(name))
            # Validate the common handle/stream contract without requiring
            # unrelated sparse formats or operations to be installed.
            lib.cusparseCreate
            lib.cusparseSetStream
            _LIB = lib
            return lib
        except (OSError, NotImplementedError) as exc:
            _LIB_ERROR = f"cannot load cuSPARSE generic API from {name}: {exc}"
    raise RuntimeError(_LIB_ERROR or "libcusparse not found")


def skip_reason():
    """None when the cuSPARSE baseline can run here, otherwise why not."""
    if getattr(torch.version, "hip", None) is not None:
        return "no hipSPARSE baseline wired for this operator (ROCm)"
    try:
        from flagsparse.sparse_operations._common import _backend_name

        backend = _backend_name()
    except Exception:
        backend = "cuda"
    if backend != "cuda":
        return f"no vendor sparse baseline wired for this operator on {backend}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        _lib()
    except RuntimeError as exc:
        return str(exc)
    return None


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with cuSPARSE status {status}")


def _handle():
    global _HANDLE
    if _HANDLE is None:
        h = _P()
        _check(_lib().cusparseCreate(ctypes.byref(h)), "cusparseCreate")
        _HANDLE = h
    stream = torch.cuda.current_stream().cuda_stream
    _check(_lib().cusparseSetStream(_HANDLE, _P(stream)), "cusparseSetStream")
    return _HANDLE


def _index_type(t):
    return _INDEX_64I if t.dtype == torch.int64 else _INDEX_32I


def _scalar(value, dtype):
    """Host scalar in the compute type (pointer mode is HOST by default)."""
    if dtype in (torch.complex64,):
        return (ctypes.c_float * 2)(complex(value).real, complex(value).imag)
    if dtype == torch.complex128:
        return (ctypes.c_double * 2)(complex(value).real, complex(value).imag)
    if dtype == torch.float64:
        return ctypes.c_double(float(value))
    if dtype == torch.int32:
        return ctypes.c_int32(int(value))
    return ctypes.c_float(float(value))


class _Plan:
    """Descriptors, workspace and the call; ``close`` releases the descriptors."""

    def __init__(self, run, owners, destroy):
        self._run = run
        self._owners = owners  # keeps tensors / scalars alive while descriptors point at them
        self._destroy = destroy

    def run(self):
        return self._run()

    def close(self):
        release, self._destroy = self._destroy, []
        for fn, handle in release:
            fn(handle)


def _spvec(values, indices, size):
    h = _P()
    _check(
        _lib().cusparseCreateSpVec(
            ctypes.byref(h), size, values.numel(), _P(indices.data_ptr()), _P(values.data_ptr()),
            _index_type(indices), _BASE_ZERO, _CUDA_TYPES[values.dtype],
        ),
        "cusparseCreateSpVec",
    )
    return h


def _dnvec(t):
    h = _P()
    _check(
        _lib().cusparseCreateDnVec(ctypes.byref(h), t.numel(), _P(t.data_ptr()), _CUDA_TYPES[t.dtype]),
        "cusparseCreateDnVec",
    )
    return h


def prepare_spvv(values, indices, y, op, compute_dtype):
    lib, handle = _lib(), _handle()
    x_d, y_d = _spvec(values, indices, y.numel()), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    result = _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpVV_bufferSize(handle, _OP[op], x_d, y_d, ctypes.byref(result), ctype, ctypes.byref(size)),
        "cusparseSpVV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=y.device)

    def run():
        _check(
            lib.cusparseSpVV(handle, _OP[op], x_d, y_d, ctypes.byref(result), ctype, _P(work.data_ptr())),
            "cusparseSpVV",
        )
        if compute_dtype.is_complex:
            return complex(result[0], result[1])
        return result.value

    destroy = [(lib.cusparseDestroySpVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, indices, y, work, result), destroy)


def prepare_axpby(values, indices, y, alpha, beta, compute_dtype):
    lib, handle = _lib(), _handle()
    x_d, y_d = _spvec(values, indices, y.numel()), _dnvec(y)
    a, b = _scalar(alpha, compute_dtype), _scalar(beta, compute_dtype)

    def run():
        _check(lib.cusparseAxpby(handle, ctypes.byref(a), x_d, ctypes.byref(b), y_d), "cusparseAxpby")
        return y

    destroy = [(lib.cusparseDestroySpVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, indices, y, a, b), destroy)


def prepare_spmv_sell(values, cols, offsets, x, y, shape, slice_size, compute_dtype):
    lib, handle = _lib(), _handle()
    m = _P()
    nnz = int((cols >= 0).sum().item())
    _check(
        lib.cusparseCreateSlicedEll(
            ctypes.byref(m), int(shape[0]), int(shape[1]), nnz, values.numel(), int(slice_size),
            _P(offsets.data_ptr()), _P(cols.data_ptr()), _P(values.data_ptr()),
            _index_type(offsets), _index_type(cols), _BASE_ZERO, _CUDA_TYPES[values.dtype],
        ),
        "cusparseCreateSlicedEll",
    )
    x_d, y_d = _dnvec(x), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMV_bufferSize(
            handle, _OP["non"], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
            _SPMV_SELL_ALG1, ctypes.byref(size),
        ),
        "cusparseSpMV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=x.device)

    def run():
        _check(
            lib.cusparseSpMV(
                handle, _OP["non"], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
                _SPMV_SELL_ALG1, _P(work.data_ptr()),
            ),
            "cusparseSpMV",
        )
        return y

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, cols, offsets, x, y, work, one, zero), destroy)


# ---------------------------------------------------------------------------
# Sparse-matrix operators for the q4 variants of existing operators
# ---------------------------------------------------------------------------


def _spmat(fmt, arrays, shape, value_dtype):
    """``arrays``: csr (values, col_indices, row_offsets), csc (values, row_indices,
    col_offsets), coo (values, row_indices, col_indices), bell (values as a
    rows x ellCols matrix, block column indices, block size)."""
    lib = _lib()
    h = _P()
    values, a, b = arrays
    rows, cols = int(shape[0]), int(shape[1])
    ctype = _CUDA_TYPES[value_dtype]
    if fmt == "csr":
        status = lib.cusparseCreateCsr(
            ctypes.byref(h), rows, cols, values.numel(), _P(b.data_ptr()), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(b), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "csc":
        status = lib.cusparseCreateCsc(
            ctypes.byref(h), rows, cols, values.numel(), _P(b.data_ptr()), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(b), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "bell":
        # values: rows x ellCols dense block storage; a: ellColInd (block columns, -1 pad)
        block = int(b)
        status = lib.cusparseCreateBlockedEll(
            ctypes.byref(h), rows, cols, block, int(values.shape[1]), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "coo":
        status = lib.cusparseCreateCoo(
            ctypes.byref(h), rows, cols, values.numel(), _P(a.data_ptr()), _P(b.data_ptr()),
            _P(values.data_ptr()), _index_type(a), _BASE_ZERO, ctype,
        )
    else:
        raise ValueError(f"unsupported format {fmt}")
    _check(status, f"cusparseCreate{fmt.capitalize()}")
    return h


def _dnmat(t):
    """Descriptor for a 2D tensor stored row- or column-major (any leading dimension)."""
    rows, cols = int(t.shape[0]), int(t.shape[1])
    if t.stride(1) == 1:
        order, ld = _ORDER_ROW, max(1, t.stride(0))
    elif t.stride(0) == 1:
        order, ld = _ORDER_COL, max(1, t.stride(1))
    else:
        raise ValueError("dense operand must be row- or column-major")
    h = _P()
    _check(
        _lib().cusparseCreateDnMat(ctypes.byref(h), rows, cols, ld, _P(t.data_ptr()), _CUDA_TYPES[t.dtype], order),
        "cusparseCreateDnMat",
    )
    return h


# Independently checked against ROCm 6.1 hipsparse.h and HIP library_types.h.
# Keep HIP's enum table explicit even where its values agree with CUDA.
_HIP_TYPES = {
    torch.float16: 2, torch.bfloat16: 14, torch.float32: 0,
    torch.float64: 1, torch.complex64: 4, torch.complex128: 5,
}
_HIP_OP = {"non": 0, "trans": 1, "conj": 2}
_HIP_INDEX = {torch.int32: 2, torch.int64: 3}
_HIP_ORDER = {"col": 1, "row": 2}


def _prepare_registered_generic(fmt, arrays, shape, rhs, out, op_a, op_b,
                                compute_dtype, backend, matrix):
    """Explicit native baseline, with a device-local handle owned by this plan.

    Unlike the historical q4 entry points, this path supports either vendor and
    cleans up partially constructed plans. No input conversion is performed.
    """
    if backend not in ("cuda", "rocm"):
        raise NotImplementedError(f"direct sparse C API is unavailable on {backend}")
    hip = backend == "rocm"
    if hip != bool(getattr(torch.version, "hip", None)) or rhs.device.type != "cuda":
        raise NotImplementedError(f"{backend} baseline requires a matching CUDA/HIP device")
    if fmt not in ("csr", "coo", "csc"):
        raise ValueError(f"unsupported native baseline format: {fmt}")
    prefix = "hipsparse" if hip else "cusparse"
    names = ([ctypes.util.find_library("hipsparse"), "libhipsparse.so", "hipsparse.dll"]
             if hip else [ctypes.util.find_library("cusparse"), "libcusparse.so.12", "libcusparse.so"])
    library = None
    errors = []
    for name in filter(None, names):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError as exc:
            errors.append(str(exc))
    if library is None:
        raise NotImplementedError(f"{prefix} library unavailable: {'; '.join(errors)}")

    def function(suffix, signature=None):
        try:
            fn = getattr(library, prefix + suffix)
        except AttributeError as exc:
            raise NotImplementedError(f"{prefix} lacks {prefix + suffix}") from exc
        fn.argtypes = signature if signature is not None else _SIGNATURES["cusparse" + suffix]
        fn.restype = _I
        return fn

    def check(status, operation):
        if status == 10:  # HIPSPARSE/CUSPARSE_STATUS_NOT_SUPPORTED
            raise NotImplementedError(f"{prefix}{operation}: native combination not supported (status 10)")
        if status:
            raise RuntimeError(f"{prefix}{operation} failed (status {status}, {fmt}, "
                               f"{arrays[0].dtype}/{rhs.dtype}/{out.dtype}, {op_a}/{op_b})")

    types = _HIP_TYPES if hip else _CUDA_TYPES
    operations = _HIP_OP if hip else _OP
    index_types = _HIP_INDEX if hip else {torch.int32: _INDEX_32I, torch.int64: _INDEX_64I}
    orders = _HIP_ORDER if hip else {"col": _ORDER_COL, "row": _ORDER_ROW}
    if any(t.device != rhs.device for t in (*arrays, out)):
        raise ValueError("native baseline arrays and output must be on the RHS device")
    if any(t.ndim != 1 or not t.is_contiguous() for t in arrays):
        raise ValueError("native sparse descriptors require contiguous one-dimensional arrays")
    if any(t.dtype not in types for t in (arrays[0], rhs, out)) or compute_dtype not in types:
        raise NotImplementedError(f"{prefix}: unsupported descriptor dtype")
    destroy = []

    def create(suffix, destroy_suffix, *args):
        handle = _P()
        # Resolve destruction first so missing symbols cannot leak a descriptor.
        release = function(destroy_suffix, [_P])
        check(function(suffix)(ctypes.byref(handle), *args), suffix)
        destroy.append((release, handle))
        return handle

    def dense(t):
        if not matrix:
            if t.ndim != 1 or t.stride(0) != 1:
                raise ValueError("native DnVec requires a contiguous vector")
            return create("CreateDnVec", "DestroyDnVec", t.numel(), _P(t.data_ptr()), types[t.dtype])
        if t.stride(1) == 1:
            order, ld = orders["row"], max(1, t.stride(0))
        elif t.stride(0) == 1:
            order, ld = orders["col"], max(1, t.stride(1))
        else:
            raise ValueError("native DnMat requires row/column-major storage")
        return create("CreateDnMat", "DestroyDnMat", *t.shape, ld, _P(t.data_ptr()), types[t.dtype], order)

    try:
        with torch.cuda.device(rhs.device):
            handle = create("Create", "Destroy")
            set_stream = function("SetStream")
            check(set_stream(handle, _P(torch.cuda.current_stream(rhs.device).cuda_stream)), "SetStream")
            version = _I()
            check(function("GetVersion", [_P, ctypes.POINTER(_I)])(handle, ctypes.byref(version)), "GetVersion")
            values, first, second = arrays
            if first.dtype not in index_types or second.dtype not in index_types:
                raise ValueError("native sparse baseline requires int32/int64 indices")
            if fmt == "coo":
                if first.dtype != second.dtype:
                    raise NotImplementedError("native COO descriptor requires equal index dtypes")
                sparse = create("CreateCoo", "DestroySpMat", *shape, values.numel(),
                                _P(first.data_ptr()), _P(second.data_ptr()), _P(values.data_ptr()),
                                index_types[first.dtype], 0, types[values.dtype])
            else:
                sparse = create("Create" + fmt.capitalize(), "DestroySpMat", *shape, values.numel(),
                                _P(second.data_ptr()), _P(first.data_ptr()), _P(values.data_ptr()),
                                index_types[second.dtype], index_types[first.dtype], 0, types[values.dtype])
            rhs_d, out_d = dense(rhs), dense(out)
            one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
            operation = "SpMM" if matrix else "SpMV"
            arguments = (handle, operations[op_a])
            if matrix:
                arguments += (operations[op_b],)
            arguments += (ctypes.byref(one), sparse, rhs_d, ctypes.byref(zero), out_d, types[compute_dtype], 0)
            size = ctypes.c_size_t()
            execute = function(operation)
            check(function(operation + "_bufferSize")(*arguments, ctypes.byref(size)), operation + "_bufferSize")
            workspace = torch.empty(max(1, size.value), dtype=torch.uint8, device=rhs.device)
            preprocessed = False
            if hip and matrix and hasattr(library, "hipsparseSpMM_preprocess"):
                preprocess = function("SpMM_preprocess", _SIGNATURES["cusparseSpMM"])
                check(preprocess(*arguments, _P(workspace.data_ptr())), "SpMM_preprocess")
                preprocessed = True

        def run():
            with torch.cuda.device(rhs.device):
                check(set_stream(handle, _P(torch.cuda.current_stream(rhs.device).cuda_stream)), "SetStream")
                check(execute(*arguments, _P(workspace.data_ptr())), operation)
            return out

        plan = _Plan(run, (library, arrays, rhs, out, one, zero, workspace), list(reversed(destroy)))
        plan.metadata = dict(vendor_backend=prefix, vendor_version=version.value,
                             vendor_alg="default", vendor_preprocessed=preprocessed,
                             compute_dtype=str(compute_dtype))
        return plan
    except BaseException:
        for release, descriptor in reversed(destroy):
            release(descriptor)
        raise


def prepare_spmv(fmt, arrays, shape, x, y, op, compute_dtype, *, backend=None):
    """y = op(A) @ x; x / y dtypes may differ from A's (mixed precision)."""
    if backend is not None:
        return _prepare_registered_generic(fmt, arrays, shape, x, y, op, "non", compute_dtype, backend, False)
    lib, handle = _lib(), _handle()
    values = arrays[0]
    m = _spmat(fmt, arrays, shape, values.dtype)
    x_d, y_d = _dnvec(x), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMV_bufferSize(
            handle, _OP[op], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
            _ALG_DEFAULT, ctypes.byref(size),
        ),
        "cusparseSpMV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=x.device)

    def run():
        _check(
            lib.cusparseSpMV(
                handle, _OP[op], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
                _ALG_DEFAULT, _P(work.data_ptr()),
            ),
            "cusparseSpMV",
        )
        return y

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (arrays, x, y, work, one, zero), destroy)


def prepare_spmm(fmt, arrays, shape, B, C, op_a, op_b, compute_dtype, *, backend=None):
    """C = op(A) @ op(B); B / C may be row- or column-major views."""
    if backend is not None:
        return _prepare_registered_generic(fmt, arrays, shape, B, C, op_a, op_b, compute_dtype, backend, True)
    lib, handle = _lib(), _handle()
    values = arrays[0]
    m = _spmat(fmt, arrays, shape, values.dtype)
    b_d, c_d = _dnmat(B), _dnmat(C)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMM_bufferSize(
            handle, _OP[op_a], _OP[op_b], ctypes.byref(one), m, b_d, ctypes.byref(zero), c_d,
            ctype, _ALG_DEFAULT, ctypes.byref(size),
        ),
        "cusparseSpMM_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=B.device)

    def run():
        _check(
            lib.cusparseSpMM(
                handle, _OP[op_a], _OP[op_b], ctypes.byref(one), m, b_d, ctypes.byref(zero), c_d,
                ctype, _ALG_DEFAULT, _P(work.data_ptr()),
            ),
            "cusparseSpMM",
        )
        return C

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnMat, b_d), (lib.cusparseDestroyDnMat, c_d)]
    return _Plan(run, (arrays, B, C, work, one, zero), destroy)


def prepare_sddmm(arrays, shape, A, B, out_values, op_a, op_b, compute_dtype):
    """out_values = op(A) @ op(B) sampled on the CSR pattern ``arrays`` (values unused)."""
    lib, handle = _lib(), _handle()
    _, cols, ptr = arrays
    pattern = (out_values, cols, ptr)
    m = _spmat("csr", pattern, shape, out_values.dtype)
    a_d, b_d = _dnmat(A), _dnmat(B)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    args = (handle, _OP[op_a], _OP[op_b], ctypes.byref(one), a_d, b_d, ctypes.byref(zero), m, ctype, _ALG_DEFAULT)
    size = ctypes.c_size_t(0)
    _check(lib.cusparseSDDMM_bufferSize(*args, ctypes.byref(size)), "cusparseSDDMM_bufferSize")
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=A.device)
    _check(lib.cusparseSDDMM_preprocess(*args, _P(work.data_ptr())), "cusparseSDDMM_preprocess")

    def run():
        _check(lib.cusparseSDDMM(*args, _P(work.data_ptr())), "cusparseSDDMM")
        return out_values

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnMat, a_d), (lib.cusparseDestroyDnMat, b_d)]
    return _Plan(run, (pattern, A, B, work, one, zero), destroy)
