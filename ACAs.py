"""
    A Python (not yet Numba) implementations of 
    1. Adaptive Cross Approximation (ACA) algorithm with a random pivot selection.
    2. Adaptive Cross Approximation with Geometrical Pivot selection (ACA-GP) algorithm.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: May-Oct 2024
    License: BSD 3-Clause
"""

import numpy as np
from numba import jit, prange, float64, int32
from numba.types import Tuple
import os
import math

os.environ["NUMBA_THREADING_LAYER"] = "omp"
NCPU = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NCPU)

# Extra functions

def find_circle(x1, y1, x2, y2, x3, y3):
    # Calculate the coefficients of the equations
    A1 = x2 - x1
    B1 = y2 - y1
    C1 = x2**2 - x1**2 + y2**2 - y1**2

    A2 = x3 - x1
    B2 = y3 - y1
    C2 = x3**2 - x1**2 + y3**2 - y1**2

    # Compute the determinant
    D = A1 * B2 - A2 * B1

    if D == 0:
        return 0,0,np.inf
        # raise ValueError("The three points are colinear and do not define a circle.")

    # Calculate the center of the circle (x0, y0)
    x0 = (C1 * B2 - C2 * B1) / (2 * D)
    y0 = (A1 * C2 - A2 * C1) / (2 * D)

    # Calculate the radius of the circle
    r = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    return x0, y0, r

def distance_to_the_circle(x,y,x0,y0,r):
    return abs(np.sqrt((x-x0)**2 + (y-y0)**2) - r)

#FIXME: shall be adjusted for numba usage
def frobenius_norm(x):
    return np.linalg.norm(x)

# @numba.jit(nopython=True, parallel=True)
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

def find_max_vector_element(v, Ik):
    mask = np.ones(len(v), dtype=bool)
    mask[Ik] = False
    max_index = np.argmax(np.abs(v[mask]))
    return np.abs(v[mask][max_index]), np.arange(len(v))[mask][max_index]


###########################################
##        ACA  classic algorithm         ##
###########################################

def aca(t_coord, s_coord, Estar, tol, max_rank, min_pivot, kpower):
    """
    Classical ACA with a random choice of pivot.

    Args:
        t_coord (numpy.ndarray):    The target cloud of points.
        s_coord (numpy.ndarray):    The source cloud of points.
        Estar (float):              The kernel/Green function parameter.
        tol (float):                The tolerance for the ACA algorithm.
        max_rank (int):             The maximum rank of the ACA algorithm.
        min_pivot (float):          The minimum tolerable value of the pivot.
        kpower (int):               The power of the kernel.

    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA algorithm which decompose the matrix A as U @ V.
        R_norm/M_norm (float): The ratio of the residual norm to the approximate matrix norm.
        ranks (int): The rank of the approximate decomposition.
        Jk (numpy.ndarray): The indices of the columns of the source cloud of points.
        Ik (numpy.ndarray): The indices of the rows of the target cloud of points.
    """
    n = t_coord.shape[0]
    m = s_coord.shape[0]
    U = np.zeros((n, max_rank), dtype=np.float64)
    V = np.zeros((max_rank, m), dtype=np.float64)
    ranks = 0
    R_norm = float64(1.)
    M_norm = float64(1.)
    Jk = []
    Ik = [] 
    history = np.zeros((max_rank, 4), dtype=np.float64)

    # Start algorithm
    # 1. Find randomly a column
    j1 = np.random.randint(0, m)
    # 2. Find the maximal elemnt 
    u1 = line_kernel(t_coord, s_coord[j1], Estar, kpower)
    _, i1 = find_max_vector_element(u1, Ik)
    v1 = line_kernel(t_coord[i1], s_coord, Estar, kpower)
    Ik.append(i1)
    Jk.append(j1)
    pivot = v1[j1]
    if abs(pivot) < min_pivot:
        print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
        return U, V, 0., ranks, Jk, Ik, history
        # raise ValueError("Pivot is too small: " +str(pivot))
    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))
    # 4. Evaluate the associated row and column of the matrix
    U[:, 0] = sign_pivot * u1 / sqrt_pivot
    V[0, :] = v1 / sqrt_pivot

    # Compute matrix norm
    R_norm = frobenius_norm(u1) * frobenius_norm(v1) / np.abs(pivot)
    M_norm = R_norm 
    history[0] = np.array([R_norm, M_norm, R_norm/M_norm, pivot])

    ranks = 1
    # Main loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Version 1: Get a random column which is not in Jk
        j_k = Jk[0]
        while j_k in Jk:
            j_k = np.random.randint(0, m)
        # Version 2: Search for maximal entry in v_k
        # _, j_k = find_max_vector_element(v_k, Jk)

        Jk.append(j_k)
        # Extract the column
        u_k = line_kernel(t_coord, s_coord[j_k], Estar, kpower)
        # Remove the contribution of the previous ranks
        u_k -= np.dot(U[:,:ranks], V[:ranks,j_k])

        # Find the maximal element
        _, i_k = find_max_vector_element(u_k, Ik)
        # Store the column
        Ik.append(i_k)
        # Compute the new row and column
        v_k = line_kernel(t_coord[i_k], s_coord, Estar, kpower)
        # Remove the contribution of the previous ranks
        v_k -= np.dot(U[i_k,:ranks], V[:ranks,:])
        
        pivot = u_k[i_k]
        if abs(pivot) < min_pivot:
            print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
            return U, V, 0., ranks, Jk, Ik, history
            # raise ValueError("Pivot is too small: " +str(pivot))
        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        U[:, ranks] = u_k * sign_pivot / sqrt_pivot
        V[ranks, :] = v_k / sqrt_pivot

        # Compute residual norm
        u_k_norm = frobenius_norm(u_k)
        v_k_norm = frobenius_norm(v_k)
        R_norm = float64(u_k_norm * v_k_norm / np.abs(pivot))

        # Approximate matrix norm
        cross_term = 2 * np.dot(np.dot(U[:,:ranks].T, u_k), np.dot(v_k, V[:ranks,:].T)) / pivot
        M_norm = float64(np.sqrt(M_norm**2 + R_norm**2 + cross_term))

        # Increment the rank
        ranks += 1
        # Compute real norm of the approximate matrix
        # M_real_norm = np.linalg.norm(U[:, :ranks] @ V[:ranks, :], "fro")
        history[ranks-1] = np.array([R_norm, M_norm, R_norm/M_norm, pivot]) #M_real_norm])

    # plt.show()
    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])

    return U_contiguous, V_contiguous, R_norm/M_norm, ranks, Jk, Ik, history

###########################################
##           ACA-GP algorithm            ##
###########################################

def aca_gp(t_coord, s_coord, Estar, tol, max_rank, min_pivot, central_fraction, kpower):
    """
    Adaptive Cross Approximation with Geometrical Pivots (ACA-GP) for $m \le n$.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        tol (float): The global tolerance for the ACA-GP algorithm.
        max_rank (int): The maximum rank for the ACA-GP algorithm.
        min_pivot (float): The pivot tolerance.
        central_fraction (float): A relative distance to define the central subsets.
        kpower (int): The power of the kernel.

    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA-GP algorithm which decompose the matrix A as U @ V.
        R_norm/M_norm (float): The ratio of the residual norm to the approximate matrix norm.
        ranks (int): The rank of the approximate decomposition.
        Jk (numpy.ndarray): The indices of the columns of the source cloud of points.
        Ik (numpy.ndarray): The indices of the rows of the target cloud of points.
    """

    # # Defind the minimal size of the rectangular cloud
    # if min_cloud_section_t is None:
    #     min_x = np.min(s_coord[:,0])
    #     max_x = np.max(s_coord[:,0])
    #     min_y = np.min(s_coord[:,1])
    #     max_y = np.max(s_coord[:,1])
    #     min_cloud_section_t = min(max_x - min_x, max_y - min_y)
    #     min_cloud_section_s = min_cloud_section_t
    n = t_coord.shape[0]
    m = s_coord.shape[0]
    U = np.zeros((n, max_rank), dtype=np.float64)
    V = np.zeros((max_rank, m), dtype=np.float64)
    ranks = 0
    R_norm = 1.0
    M_norm = 1.0
    Jk = []
    Ik = []
    Ic = []
    Jc = []
    history = np.zeros((max_rank, 4), dtype=np.float64)

    # Find centers of both point clouds
    center_t = np.mean(t_coord, axis=0)
    center_s = np.mean(s_coord, axis=0)

    # Find first pivot (i1, j1) as the closest to the center and on the side of the opposite cloud
    # select only those t_coord which verify the condition np.dot(t_coord-center_t, center_s-center_t) > 0
    t_coord_filtered = t_coord[np.dot(t_coord-center_t, center_s-center_t) > 0]
    t_coord_filtered_index = np.arange(n)[np.dot(t_coord-center_t, center_s-center_t) > 0]
    distances_t = np.linalg.norm(t_coord_filtered - center_t, axis=1)
    i1 = t_coord_filtered_index[np.argmin(distances_t)] #[np.dot(t_coord-center_t, center_s-center_t) > 0])
    s_coord_filtered = s_coord[np.dot(s_coord-center_s, center_t-center_s) > 0]
    s_coord_filtered_index = np.arange(m)[np.dot(s_coord-center_s, center_t-center_s) > 0]
    distances_s = np.linalg.norm(s_coord_filtered - center_s, axis=1)
    j1 = s_coord_filtered_index[np.argmin(distances_s)] #[np.dot(s_coord-center_s, center_t-center_s) > 0])

    Ik.append(i1)
    Jk.append(j1)

    # Construct central subsets.
    # If number of elements in the central subset is less than the maximal rank, increase the central subset
    central_fraction_t = central_fraction
    central_fraction_s = central_fraction
    while len(Ic) < max_rank:
        for i in range(n):
            if np.linalg.norm(t_coord[i] - center_t) < central_fraction_t  and i != i1:
                Ic.append(i)
        central_fraction_t *= 1.1
    while len(Jc) < max_rank:
        for j in range(m):
            if np.linalg.norm(s_coord[j] - center_s) < central_fraction_s  and j != j1:
                Jc.append(j)
        central_fraction_s *= 1.1


    # Initialize first row and column for U and V matrices
    pivot = line_kernel(t_coord[i1], s_coord[j1], Estar, kpower)
    if abs(pivot) < min_pivot:
        print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
        return U, V, 0., ranks, Jk, Ik, history

    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))

    u1 = line_kernel(t_coord, s_coord[j1], Estar, kpower)
    U[:, 0] = sign_pivot * u1 / sqrt_pivot

    v1 = line_kernel(t_coord[i1], s_coord, Estar, kpower)
    V[0, :] = v1 / sqrt_pivot

    # Compute initial residual and approximate matrix norms
    R_norm = np.linalg.norm(U[:, 0]) * np.linalg.norm(V[0, :])
    M_norm = R_norm
    history[0] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    # return U, V, R_norm / M_norm, ranks, Jk, Ik, history
    ranks = 1

    # Main ACA-GP loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Select next pivot points (ir, jr)
        if ranks == 1:
            ir, jr, pivot = rank2_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, Estar, kpower)
        elif ranks == 2:
            ir, jr, pivot = rank3_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, Estar, kpower)
        else:
            # ir, jr, pivot = higher_ranks_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, ranks, Estar, kpower)
            # ir, jr, pivot = higher_ranks_selection_first_circle(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, ranks, Estar, kpower)
            ir, jr, pivot = higher_ranks_selection_random(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, ranks, Estar, kpower)

        # else:
        #     return U, V, 0., ranks-1, Jk, Ik, history

        # Add pivots to the selected sets
        Ik.append(ir)
        Jk.append(jr)
        # remove pivots from central subsets
        # if ranks <= 2:
        Ic.remove(ir)
        Jc.remove(jr)


        # Compute new row and column
        if abs(pivot) < min_pivot:
            print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
            return U, V, 0., ranks, Jk, Ik, history

        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        ur = line_kernel(t_coord, s_coord[jr], Estar, kpower) - np.dot(U[:, :ranks], V[:ranks, jr])
        vr = line_kernel(t_coord[ir], s_coord, Estar, kpower) - np.dot(U[ir, :ranks], V[:ranks, :])
        assert np.abs(pivot - ur[ir]) < 1e-10, "Pivot is not correct"
        assert np.abs(pivot - vr[jr]) < 1e-10, "Pivot is not correct"
        # Update U and V
        U[:, ranks] = ur * sign_pivot / sqrt_pivot
        V[ranks, :] = vr / sqrt_pivot

        # Update residual norm
        ur_norm = np.linalg.norm(ur)
        vr_norm = np.linalg.norm(vr)
        R_norm = float(ur_norm * vr_norm / np.abs(pivot))

        # Update matrix norm
        cross_term = 2 * np.dot(np.dot(U[:, :ranks].T, ur), np.dot(vr, V[:ranks, :].T)) / pivot
        M_norm = float(np.sqrt(M_norm**2 + R_norm**2 + cross_term))
        history[ranks] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

        # Increment rank
        ranks += 1

    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])

    return U_contiguous, V_contiguous, R_norm / M_norm, ranks, Jk, Ik, history


def rank2_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, Estar, kpower):
    """
    Rank 2 selection of optimal points based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
    Returns:
        i2, j2 (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a random point from the target set excluding the first pivot
    i2 = np.random.choice(Ic)

    # Step 2: construct the circle passing through the first pivots (x_i1,y_j1) and the selected point x_i2
    x1 = t_coord[Ik[0]][0]
    y1 = t_coord[Ik[0]][1]
    x2 = s_coord[Jk[0]][0]
    y2 = s_coord[Jk[0]][1]
    x3 = t_coord[i2][0]
    y3 = t_coord[i2][1]

    x0, y0, r = find_circle(x1, y1, x2, y2, x3, y3)

    # Step 3: Initialization
    l = 0
    Jcl = Jc.copy()
    prev_pivot = 0
    while True:
        # Step 4: Select a point y_{j^l_2} that minimizes distance to the circle
        distances = [distance_to_the_circle(s_coord[j][0], s_coord[j][1], x0, y0, r) for j in Jcl]
        j2_l = Jcl[np.argmin(distances)]

        # Step 5: Evaluate the residual component
        pivot = line_kernel(t_coord[i2], s_coord[j2_l], Estar, kpower) - U[i2,0] * V[0,j2_l]

        # Step 6: If the pivot's absolute value is smaller than the previous pivot, return the previous pivot
        if abs(pivot) <= abs(prev_pivot):
            return i2, j2_l_, prev_pivot

        # Step 7: Update l and the central subset
        prev_pivot = pivot
        l += 1
        j2_l_ = j2_l
        Jcl.remove(j2_l)

def rank3_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, Estar, kpower):
    """
    Rank 3 selection of optimal points based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
    Returns:
        i3, j3 (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a point i_3 that minimizes distance to the conjugate circle
    distances = [np.linalg.norm(t_coord[i] - t_coord[Ik[0]]) for i in Ic]
    i3 = Ic[np.argmin(distances)]

    # Step 2: Initialize l and adjust the central subset of the source cloud
    l = 1
    Jcl = Jc.copy()

    x1 = t_coord[Ik[0]][0]
    y1 = t_coord[Ik[0]][1]
    x2 = s_coord[Jk[0]][0]
    y2 = s_coord[Jk[0]][1]
    x3 = t_coord[Ik[1]][0]
    y3 = t_coord[Ik[1]][1]
    x0,y0,r = find_circle(x1, y1, x2, y2, x3, y3)
    # conjugate circle at x1,y1
    rad = np.array([x1-x0,y1-y0])
    tan = np.zeros(2)
    tan[0] = -rad[1]
    tan[1] = rad[0]
    tan /= np.linalg.norm(tan)
    if np.dot(tan, np.array([x2-x1,y2-y1])) < 0:
        tan *= -1
    x1ort,y1ort = x1+r*tan[0], y1+r*tan[1]
    # conjugate circle at x2,y2
    rad = np.array([x2-x0,y2-y0])
    tan[0] = -rad[1]
    tan[1] = rad[0]
    tan /= np.linalg.norm(tan)
    if np.dot(tan, np.array([x1-x2,y1-y2])) < 0:
        tan *= -1
    x2ort,y2ort = x2+r*tan[0], y2+r*tan[1]

    # Select a point in Ic close to the conjugate circle at x1,y1
    distances = [distance_to_the_circle(t_coord[i][0], t_coord[i][1], x1ort, y1ort, r) for i in Ic]
    i3 = Ic[np.argmin(distances)]
    j3_l_ = None
    prev_pivot = 0
    
    while True:
        # Step 3: Select a point y_{j^l_3} that minimizes distance to the conjugate circle at x2,y2
        distances = [distance_to_the_circle(s_coord[j][0], s_coord[j][1], x2ort, y2ort, r) for j in Jcl]
        j3_l = Jcl[np.argmin(distances)]

        # Step 4: Evaluate the residual component
        pivot = line_kernel(t_coord[i3], s_coord[j3_l], Estar, kpower) - np.dot(U[i3,:2], V[:2,j3_l])

        # Step 5: Check if the residual condition is met
        if abs(pivot) <= abs(prev_pivot):
            return i3, j3_l_, prev_pivot

        # Step 6: Update l and the central subset
        prev_pivot = pivot
        l += 1
        j3_l_ = j3_l
        Jcl.remove(j3_l)

def higher_ranks_selection_random(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, k, Estar, kpower):
    # Step 1: select randomly in all t_coord points except the ones in Ik
    i_k = np.random.choice(Ic)
    # Step 2: evaluate the residual row
    pivots = [line_kernel(t_coord[i_k], s_coord[j], Estar, kpower) - np.dot(U[i_k,:k+1], V[:k+1,j]) for j in Jc]
    # Step 3: select the column j_k with the maximal absolute value
    j_k_dx = np.argmax(np.abs(pivots))
    j_k = Jc[j_k_dx]
    pivot = pivots[j_k_dx]
    return i_k, j_k, pivot


def higher_ranks_selection(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, k, Estar, kpower):
    """
    Pivot Choice for Higher Ranks $k \ge 4$ based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
        k (int): Current rank.
    Returns:
        i_k, j_k (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a trial point i^t randomly from the target set excluding previously selected points
    Ick = Ic.copy()
    Jck = Jc.copy()
    i_t = np.random.choice(Ick)

    # Step 2: Compute R_{i^t j} for all j in J^c_k
    # FIXME: check whether we need to substract up to k or k+1
    pivots = [line_kernel(t_coord[i_t], s_coord[j], Estar, kpower) - np.dot(U[i_t,:k+1], V[:k+1,j]) for j in Jck]

    # Step 3: Select the column j_k with the maximal absolute value
    j_k_idx = np.argmax(np.abs(pivots))
    j_k = Jck[j_k_idx]
    pivot = pivots[j_k_idx]

    # Step 4: Compute R_{i j_k} for all i in I^c_k
    # FIXME: check whether we need to substract up to k or k+1
    pivots = [line_kernel(t_coord[i], s_coord[j_k], Estar, kpower) - np.dot(U[i,:k+1], V[:k+1,j_k]) for i in Ick]
    
    # Step 5: Select the row i_k with the maximal absolute value
    i_k_idx = np.argmax(np.abs(pivots))  # Get index in Ick
    i_k = Ick[i_k_idx]  # Get actual i_k value
    pivot = pivots[i_k_idx]  # Get corresponding pivot value

    return i_k, j_k, pivot

def higher_ranks_selection_first_circle(t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, k, Estar, kpower):
    """
    Pivot Choice for Higher Ranks $k \ge 4$ based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
        k (int): Current rank.
    Returns:
        i_k, j_k (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a trial point i^t randomly from the target set excluding previously selected points
    Ick = Ic.copy()
    Jck = Jc.copy()
    i_k = np.random.choice(Ick)

    x1 = t_coord[Ik[0]][0]
    y1 = t_coord[Ik[0]][1]
    x2 = s_coord[Jk[0]][0]
    y2 = s_coord[Jk[0]][1]
    x3 = t_coord[i_k][0]
    y3 = t_coord[i_k][1]

    x0, y0, r = find_circle(x1, y1, x2, y2, x3, y3)

    # Step 3: Initialization
    l = 0
    prev_pivot = 0
    jk_l_ = None
    while True:
        # Step 4: Select a point y_{j^l_2} that minimizes distance to the circle
        distances = [distance_to_the_circle(s_coord[j][0], s_coord[j][1], x0, y0, r) for j in Jck]
        jk_l = Jck[np.argmin(distances)]

        # Step 5: Evaluate the residual component
        pivot = line_kernel(t_coord[i_k], s_coord[jk_l], Estar, kpower) - np.dot(U[i_k,:k+1], V[:k+1,jk_l])

        # Step 6: If the pivot's absolute value is smaller than the previous pivot, return the previous pivot
        if abs(pivot) <= abs(prev_pivot):
            return i_k, jk_l_, prev_pivot

        # Step 7: Update l and the central subset
        prev_pivot = pivot
        l += 1
        jk_l_ = jk_l
        Jck.remove(jk_l)


