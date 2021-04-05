"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
    @Author: Aaron Guan (zhongg@andrew.cmu.edu), 2021
"""

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    """
    Return x s.t. Ax = b using pseudo inverse.
    x = (A^T A)^-1 A^T b for least square |Ax - b|^2
    """
    x = inv(A.T @ A) @ A.T @ b
    return x, None


def solve_lu(A, b):
    """
    Return x, U s.t. Ax = b, and A = LU with LU decomposition.
    x = (A^T A)^-1 A^T b for least square |Ax - b|^2.
    A^T @ A is a square matrix and can do LU decomposition.
    x = (A^T A)^-1 A^T b  = (LU)^-1 A^T b
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    """
    LU = splu(A.T @ A, permc_spec='NATURAL')
    x = LU.solve(A.T @ b)
    U = LU.U
    return x, U


def solve_lu_colamd(A, b):
    """
    Return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    """
    LU = splu(A.T @ A, permc_spec='COLAMD')
    x = LU.solve(A.T @ b)
    U = LU.U
    return x, U


def solve_qr(A, b):
    """
    Return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - z|^2 + |e|^2
    R would be a sparse UPPER triangular matrix. We can calculate x from |Rx - z|^2
    https://github.com/theNded/PySPQR
    """
    z, R, E, rank = rz(A, b, permc_spec='NATURAL')
    x = spsolve_triangular(R, z, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    """
    Return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - z|^2 + |e|^2, with reordered QR decomposition
    E is the permutation matrix. Used reordered QR decomposition to get R and z, then solve
    |Ry - z|^2. After we get y, we use the permutation matrix to permute it to get correct solution
    based on E^T x = y and E^T E=I when E is a permutation matrix.
    https://github.com/theNded/PySPQR
    """
    z, R, E, rank = rz(A, b, permc_spec='COLAMD')
    x = spsolve_triangular(R, z, lower=False)
    x = inv(permutation_vector_to_matrix(E).T) @ x
    return x, R


def solve(A, b, method='default'):
    """
    :param A: (M, N) Jacobian matirx
    :param b: (M, 1) residual vector
    :return x: (N, 1) state vector obtained by solving Ax = b.
    """
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)
