"""
    A Python implementations of a genetic ACA like algorithm which probes all possible pivots.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: May 2024 - Feb 2025
    License: BSD 3-Clause
"""

import numpy as np
from numba import jit, prange, float64, int32
from numba.types import Tuple
import os

os.environ["NUMBA_THREADING_LAYER"] = "omp"
NCPU = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NCPU)

def frobenius_norm(x):
    return np.linalg.norm(x)

def line_kernel(t_coord, s_coord, E, kpower):
    factor = 1/(np.pi*E)
    if t_coord.ndim == 1 and s_coord.ndim > 1:
        result = np.zeros(s_coord.shape[0])
        for i in prange(s_coord.shape[0]):
            dist = 0.0
            for j in prange(t_coord.shape[0]):
                dist += (t_coord[j] - s_coord[i, j])**2
            result[i] = factor / np.sqrt(dist)**kpower
        return result
    elif s_coord.ndim == 1 and t_coord.ndim > 1:
        result = np.zeros(t_coord.shape[0])
        for i in prange(t_coord.shape[0]):
            dist = 0.0
            for j in prange(s_coord.shape[0]):
                dist += (t_coord[i, j] - s_coord[j])**2
            result[i] = factor / np.sqrt(dist)**kpower
        return result
    elif s_coord.ndim == 1 and t_coord.ndim == 1:
        dist = 0.0
        for i in prange(t_coord.shape[0]):
            dist += (t_coord[i] - s_coord[i])**2
        return factor / np.sqrt(dist)**kpower
    else:    
        raise ValueError("Unknown dimension of t_coord or s_coord. One of two should be 1D array of size dim.")


###########################################
##        ACA test all: genetic          ##
###########################################

def aca_iterative(t_coord, s_coord, Estar, tol, max_rank, min_pivot, kpower, I, J, U_=None, V_=None):
    """
    Iterative skeletonization of a matrix for two clouds of points.

    Args:
        t_coord (numpy.ndarray):    The target cloud of points.
        s_coord (numpy.ndarray):    The source cloud of points.
        Estar (float):              The kernel/Green function parameter.
        tol (float):                The tolerance for the ACA algorithm.
        max_rank (int):             The maximum rank of the ACA algorithm.
        min_pivot (float):          The minimum tolerable value of the pivot.
        kpower (int):               The power of the kernel.
        iter_i (int):               Iteration for the pivot selection for the row
        iter_j (int):               Iteration for the pivot selection for the column

    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA algorithm which decompose the matrix A as U @ V.
    """

    n = t_coord.shape[0]
    m = s_coord.shape[0]
    if U_ is not None and V_ is not None:
        max_rank = U_.shape[1]+1
        U = np.zeros((n, max_rank), dtype=np.float64)
        V = np.zeros((max_rank, m), dtype=np.float64)
        U[:, :max_rank-1] = U_
        V[:max_rank-1, :] = V_
    else:
        max_rank = 1
        U = np.zeros((n, max_rank), dtype=np.float64)
        V = np.zeros((max_rank, m), dtype=np.float64)

    if max_rank == 1:
        i1 = I
        j1 = J
        u1 = line_kernel(t_coord, s_coord[j1], Estar, kpower)
        v1 = line_kernel(t_coord[i1], s_coord, Estar, kpower)
        pivot = v1[j1]
        if abs(pivot) < min_pivot:
            print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", max_rank," i1 = ", I, " j1 = ", J)
            return U, V
            # raise ValueError("Pivot is too small: " +str(pivot))
        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))
        # 4. Evaluate the associated row and column of the matrix
        U[:, 0] = sign_pivot * u1 / sqrt_pivot
        V[0, :] = v1 / sqrt_pivot
    else:
        i_k = I
        j_k = J
        u_k = line_kernel(t_coord, s_coord[j_k], Estar, kpower)
        u_k -= np.dot(U[:,:max_rank-1], V[:max_rank-1,j_k])

        v_k = line_kernel(t_coord[i_k], s_coord, Estar, kpower)
        v_k -= np.dot(U[i_k,:max_rank-1], V[:max_rank-1,:])
    
        pivot = u_k[i_k]
        if abs(pivot) < min_pivot:
            print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", max_rank," i1 = ", I, " j1 = ", J)
            return U, V
            # raise ValueError("Pivot is too small: " +str(pivot))
        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        U[:, max_rank-1] = u_k * sign_pivot / sqrt_pivot
        V[max_rank-1, :] = v_k / sqrt_pivot

    U_contiguous = np.ascontiguousarray(U[:, :max_rank])
    V_contiguous = np.ascontiguousarray(V[:max_rank, :])

    return U_contiguous, V_contiguous

