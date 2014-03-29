"""
A drop in replacement for numpy.dot

Avoid temporary copies of non C-contiguous arrays

The code is available here: http://pastebin.com/raw.php?i=M8TfbURi 
In future this will be available in numpy directly:
https://github.com/numpy/numpy/pull/2730
"""

from . import _np
from . import _linalg

_assert_equal = _np.testing.assert_equal

def dot(A, B, out=None):
    """ A drop in replaement for numpy.dot
    Computes A.B optimized using fblas call """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("only 2D numpy arrays are supported")

    gemm = _linalg.get_blas_funcs('gemm', arrays=(A, B))

    if out is None:
        lda, x, y, ldb = A.shape + B.shape
        if x != y:
            raise ValueError("matrices are not aligned")
        dtype = _np.max([x.dtype for x in (A, B)])
        out = _np.empty((lda, ldb), dtype, order='C')

    if A.flags.c_contiguous and B.flags.c_contiguous:
        gemm(alpha=1., a=A.T, b=B.T,
                c=out.T, overwrite_c=True)
    if A.flags.c_contiguous and B.flags.f_contiguous:
        gemm(alpha=1., a=A.T, b=B, trans_a=True,
                c=out.T, overwrite_c=True)
    if A.flags.f_contiguous and B.flags.c_contiguous:
        gemm(alpha=1., a=A, b=B.T, trans_b=True,
                c=out.T, overwrite_c=True)
    if A.flags.f_contiguous and B.flags.f_contiguous:
        gemm(alpha=1., a=A, b=B, trans_a=True, trans_b=True,
                c=out.T, overwrite_c=True)
    return out


def test_dot():
    A = _np.random.randn(1000, 1000)
    _assert_equal(A.dot(A), dot(A, A))
    _assert_equal(A.dot(A.T), dot(A, A.T))
    _assert_equal(A.T.dot(A), dot(A.T, A))
    _assert_equal(A.T.dot(A.T), dot(A.T, A.T))
    assert(dot(A, A).flags.c_contiguous)


def test_to_fix():
    """ 1d array, complex and 3d """
    v = _np.random.randn(1000)
    dot(v, v)
    c = _np.asarray(_np.random.randn(100, 100), _np.complex)
    dot(c, c)
    t = _np.random.randn(2, 2, 3)
    dot(t, t)
