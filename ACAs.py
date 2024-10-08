"""
    A Python-Numba implementations of 
    1. Adaptive Cross Approximation (ACA) algorithm with a random pivot selection.
    2. Adaptive Cross Approximation with Geometrical Pivot selection (ACA-GP) algorithm.
    3. Adaptive Cross Approximation with Principal Component Analysis (PCA) based pivot selection (PCA-CA) algorithm.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: May-Aug 2024
    License: BSD 3-Clause
"""

import numpy as np
from numba import jit, prange, float64, int32
from numba.types import Tuple
import os

os.environ["NUMBA_THREADING_LAYER"] = "omp"
NCPU = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(NCPU)

# Classes for cluster tree
class Cluster:
    def __init__(self, cl_id, ids, coords, parent=None):
        self.cl_id = cl_id
        self.ids = ids
        self.coords = coords
        self.parent = parent
        self.children = []

    def add_children(self, children):
        self.children.extend(children)

class PCACluster:
    def __init__(self, coords):
        self.coords = coords
        self.clusters = {}
        self.depth = 0

    def pca_split(self, cluster, target_level, current_level=0,dir_vector=None):
        if current_level == target_level:
            if self.clusters.get(current_level) is None:
                self.clusters[current_level] = []
            self.clusters[current_level].append(cluster)
            self.depth = max(self.depth, current_level)
            return [cluster]

        node_coords = cluster.coords
        node_ids = cluster.ids

        mean = np.mean(node_coords, axis=0)
        if dir_vector is None:
            cov = np.cov(node_coords, rowvar=False)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[indices]
            eigenvectors = eigenvectors[:, indices]
            principal_eigenvector = eigenvectors[:, 0]
        else:
            principal_eigenvector = dir_vector            

        coords_pca = np.dot(node_coords - mean, principal_eigenvector)

        cluster1_id = np.where(coords_pca > 0)[0]
        cluster2_id = np.where(coords_pca <= 0)[0]

        cluster1_ids = node_ids[cluster1_id]
        cluster1_coords = node_coords[cluster1_id]
        c1id = cluster.cl_id + "0"
        cl1 = Cluster(c1id, cluster1_ids, cluster1_coords, cluster)

        cluster2_ids = node_ids[cluster2_id]
        cluster2_coords = node_coords[cluster2_id]
        c2id = cluster.cl_id + "1"
        cl2 = Cluster(c2id, cluster2_ids, cluster2_coords, cluster)

        if self.clusters.get(current_level) is None:
            self.clusters[current_level] = []
        self.clusters[current_level].append(cluster)
        
        # if dir_vector is not None:
        #     return self.pca_split(cl1, target_level, current_level + 1, dir_vector) + self.pca_split(cl2, target_level, current_level + 1, dir_vector)
        # else:
        return self.pca_split(cl1, target_level, current_level + 1) + self.pca_split(cl2, target_level, current_level + 1)

    def split_to_level(self, points, target_level, dir_vector=None):
        initial_cluster = Cluster("root", np.arange(len(points)), points)
        if dir_vector is not None:
            final_clusters = self.pca_split(initial_cluster, target_level, 0, dir_vector)
        else:
            final_clusters = self.pca_split(initial_cluster, target_level)
        return final_clusters

# Extra functions for ACA-GP
# @jit(float64(float64[:]),nopython=True,parallel=True)
# def frobenius_norm(x):
#     return np.sqrt(np.sum(x**2))

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

# @jit(signature=Tuple([float64, int32])(float64[:], int32[:]), nopython=True, parallel=True)
# def find_max_vector_element(v, Ik):
#     max_val = 0.
#     index = int32(-1)
#     for i in prange(v.shape[0]):
#         if i in Ik:
#             continue
#         if np.abs(v[i]) > max_val:
#             max_val = np.abs(v[i])
#             index = i
#     assert isinstance(max_val, float) and isinstance(index, int), "Type mismatch"
#     return max_val, int32(index)

def find_max_vector_element(v, Ik):
    mask = np.ones(len(v), dtype=bool)
    mask[Ik] = False
    max_index = np.argmax(np.abs(v[mask]))
    return np.abs(v[mask][max_index]), np.arange(len(v))[mask][max_index]

def find_closest_point(s_coord, t_center, Jk):
    mask = np.ones(len(s_coord), dtype=bool)
    mask[Jk] = False
    distances = np.linalg.norm(s_coord[mask] - t_center, axis=1)
    j_k = np.argmin(distances)
    return np.arange(len(s_coord))[mask][j_k]

def find_farhest_point(s_coord, t_center, Jk):
    mask = np.ones(len(s_coord), dtype=bool)
    mask[Jk] = False
    distances = np.linalg.norm(s_coord[mask] - t_center, axis=1)
    j_k = np.argmax(distances)
    return np.arange(len(s_coord))[mask][j_k]

def find_average_point(s_coord, t_center, Jk):
    mask = np.ones(len(s_coord), dtype=bool)
    mask[Jk] = False
    distances = np.linalg.norm(s_coord[mask] - t_center, axis=1)
    average_distance = np.mean(distances)
    j_k = np.argmin(np.abs(distances - average_distance))
    return np.arange(len(s_coord))[mask][j_k]

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
        # Get a random column which is not in Jk
        j_k = Jk[0]
        while j_k in Jk:
            j_k = np.random.randint(0, m)
        Jk.append(j_k)
        # Extract the column
        u_k = line_kernel(t_coord, s_coord[j_k], Estar, kpower)
        # Remove all what has been already stored in previous iterations
        # for i in range(ranks):
        #     for j in range(n):
        #         u_k[j] -= U[j, i] * V[i, j_k]
        # Faster version
        u_k -= np.dot(U[:,:ranks], V[:ranks,j_k])

        # Find the maximal element
        _, i_k = find_max_vector_element(u_k, Ik)
        # Store the column
        Ik.append(i_k)
        # Compute the new row and column
        v_k = line_kernel(t_coord[i_k], s_coord, Estar, kpower)
        # for i in range(m):
        #     for j in range(ranks):
        #         v_k[i] -= U[i_k, j] * V[j, i]
        # Faster version
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

# A fast version of the geometric pivot selection
def optimized_geometric_pivot_selection(s_coord, Jk, x0, x1, y0, y1, ranks, convex_hull_distance):
    m = s_coord.shape[0]
    inv_distances = np.zeros(m)
    
    # Create a mask for points in Jk
    mask = np.ones(m, dtype=bool)
    mask[Jk] = False
    
    # Compute distances to all points in Jk
    diff = s_coord[:, np.newaxis, :] - s_coord[Jk]
    distances = np.linalg.norm(diff, axis=2)
    inv_distances[mask] = np.sum(ranks / distances[mask], axis=1)
    
    # Set large distance for points in Jk
    inv_distances[~mask] = 1e30
    
    # Compute minimum distance to convex hull
    min_dist_x = np.minimum(np.abs(x1 - s_coord[:, 0]), np.abs(x0 - s_coord[:, 0]))
    min_dist_y = np.minimum(np.abs(y1 - s_coord[:, 1]), np.abs(y0 - s_coord[:, 1]))
    min_distance_to_convex_hull = np.minimum(min_dist_x, min_dist_y)
    
    # Handle convex hull distance calculations
    non_zero_hull_dist = min_distance_to_convex_hull > 0
    if convex_hull_distance[0] == "const":
        inv_distances[non_zero_hull_dist] += convex_hull_distance[1] / min_distance_to_convex_hull[non_zero_hull_dist]
    elif convex_hull_distance[0] == "linear":
        inv_distances[non_zero_hull_dist] += convex_hull_distance[1] * ranks / min_distance_to_convex_hull[non_zero_hull_dist]
    
    # Handle points on the convex hull
    inv_distances[~non_zero_hull_dist] += 1e30
    
    # Find the point with minimum inv_distance
    j_k = np.argmin(inv_distances)
    
    return j_k

        

# FIXME numba should be integrated
# @jit(nopython=True,parallel=True)
def aca_gp(t_coord, s_coord, Estar, tol, max_rank, min_pivot, kpower, Rank3SpecialTreatment, convex_hull_distance):
    """
    ACA-GP: ACA with a geometrical pivot selection.

    Args:
        t_coord (numpy.ndarray):        The target cloud of points.
        s_coord (numpy.ndarray):        The source cloud of points.
        Estar (float):                  The kernel/Green function parameter.
        tol (float):                    The tolerance for the ACA algorithm.
        max_rank (int):                 The maximum rank of the ACA algorithm.
        min_pivot (float):              The minimum tolerable value of the pivot.
        kpower (int):                   The power of the kernel.
        Rank3SpecialTreatment (bool):   A flag to enable special treatment for the third rank.
        convex_hull_distance (tuple):   A tuple to specify the factor and the type of distance to the convex hull.

    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA algorithm which decompose the matrix A as U @ V.
        R_norm/M_norm (float): The ratio of the residual norm to the approximate matrix norm.
        ranks (int): The rank of the approximate decomposition.
        Jk (numpy.ndarray): The indices of the columns of the source cloud of points.
        Ik (numpy.ndarray): The indices of the rows of the target cloud of points.
        history (numpy.ndarray): includes the residual norm, the approximate matrix norm, the ratio of the residual norm to the approximate matrix norm, and the pivot value.
    """
    # Initialization
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

    # Alternative start algorithm // That is much better!!!
    # 1. Find center of t_coord, O(n)
    t_center = np.mean(t_coord, axis=0)
    # 2. Find the closes point from t_coord to the center of t_coord, O(n)
    distances = np.linalg.norm(t_coord - t_center, axis=1)
    # 2.1 Find the index of the minimum distance
    i1 = np.argmin(distances)
    # 3. Find the center of s_coord, O(m)
    s_center = np.mean(s_coord, axis=0)
    # 4. Find the closest point from s_coord to the center of s_coord, O(m)
    distances = np.linalg.norm(s_coord - s_center, axis=1)
    # 4.1 Find the index of the minimum distance
    j1 = np.argmin(distances)

    # So i1,j1 form the first pivot.
    Ik.append(i1)
    Jk.append(j1)

    pivot = line_kernel(t_coord[i1], s_coord[j1], Estar, kpower)
    if abs(pivot) < min_pivot:
        print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
        return U, V, 0., ranks, Jk, Ik, history
        # raise ValueError("Pivot is too small: " +str(pivot))
    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))
    # 4. Evaluate the associated row and column of the matrix
    u1 = line_kernel(t_coord, s_coord[j1], Estar, kpower)
    v1 = line_kernel(t_coord[i1], s_coord, Estar, kpower)
    U[:, 0] = sign_pivot * u1 / sqrt_pivot
    V[0, :] = v1 / sqrt_pivot

    # Compute matrix norm
    R_norm = frobenius_norm(u1) * frobenius_norm(v1) / np.abs(pivot)
    M_norm = R_norm 
    history[0] = np.array([R_norm, M_norm, R_norm/M_norm, pivot])

    # Approximation of a convex hull
    x0 = np.min(s_coord[:,0])
    x1 = np.max(s_coord[:,0])
    y0 = np.min(s_coord[:,1])
    y1 = np.max(s_coord[:,1])

    ranks = 1
    # Main loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Find the next pivot using the geometric pivot selection
        j_k = optimized_geometric_pivot_selection(s_coord, Jk, x0, x1, y0, y1, ranks, convex_hull_distance)
        # Special treatment fot the third rank
        if Rank3SpecialTreatment and ranks == 2:
            j_k = find_average_point(s_coord, t_center, Jk)
            
        Jk.append(j_k)
        u_k = line_kernel(t_coord, s_coord[j_k], Estar, kpower)
        # for i in range(ranks):
        #     for j in range(n):
        #         u_k[j] -= U[j, i] * V[i, j_k]      
        # Faster version
        u_k -= np.dot(U[:,:ranks], V[:ranks,j_k])  

        _, i_k = find_max_vector_element(u_k, Ik)
        Ik.append(i_k)

        v_k = line_kernel(t_coord[i_k], s_coord, Estar, kpower)
        # for i in range(m):
        #     for j in range(ranks):
        #         v_k[i] -= U[i_k, j] * V[j, i]
        # Faster version
        v_k -= np.dot(U[i_k,:ranks], V[:ranks,:])

        pivot = u_k[i_k]
        if abs(pivot) < min_pivot:
            print("ACA-GP: Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
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
        history[ranks-1] = np.array([R_norm, M_norm, R_norm/M_norm, pivot]) 

    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])

    return U_contiguous, V_contiguous, R_norm/M_norm, ranks, Jk, Ik, history


###########################################
##           PCA-CA algorithm            ##
###########################################

# def pca_ca(t_coord, s_coord, Estar, tol, rank, kpower):
#     # Check that the rank is a power of two using a bitwise AND operation
#     assert rank & (rank - 1) == 0, "Rank should be a power of two"

#     pivot_tol = 1e-12
#     n = t_coord.shape[0]
#     m = s_coord.shape[0]
#     U = np.zeros((n, rank), dtype=np.float64)
#     V = np.zeros((rank, m), dtype=np.float64)
#     R_norm = float64(1.)
#     M_norm = float64(1.)
#     Jk = [] 
#     Ik = [] 

#     if rank == 0:
#         U = np.zeros((n, 1))
#         V = np.zeros((1, m))
#         # Alternative start algorithm // That is much better!!!
#         # 1. Find center of t_coord, O(n)
#         t_center = np.mean(t_coord, axis=0)
#         # 2. Find the closes point from t_coord to the center of t_coord, O(n)
#         distances = np.linalg.norm(t_coord - t_center, axis=1)
#         # 2.1 Find the index of the minimum distance
#         i1 = np.argmin(distances)
#         # 3. Find the center of s_coord, O(m)
#         s_center = np.mean(s_coord, axis=0)
#         # 4. Find the closest point from s_coord to the center of s_coord, O(m)
#         distances = np.linalg.norm(s_coord - s_center, axis=1)
#         # 4.1 Find the index of the minimum distance
#         j1 = np.argmin(distances)
#         min_dist0 = np.linalg.norm(t_center - s_center)
#         # # 3.b Find the closest point from s_coord to the point found in step 3, O(m)
#         # distances = np.linalg.norm(s_coord - t_coord[i1], axis=1)
#         # j1 = np.argmin(distances)
#         # min_dist = np.min([min_dist, distances[j1]])

#         # plt.plot([t_coord[i1,0], s_coord[j1,0]], [t_coord[i1,1], s_coord[j1,1]], "k--")
#         # plt.plot([s_coord[j1,0]],[s_coord[j1,1]],"x",c="k")
#         # plt.plot([t_coord[i1,0]],[t_coord[i1,1]],"x",c="k")

#         # So i1,j1 is the first pivot.
#         Ik[0] = i1
#         Jk[0] = j1
#         pivot = line_kernel(t_coord[i1], s_coord[j1], Estar, kpower)
#         if abs(pivot) < pivot_tol:
#             print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
#             return U, V, 0., ranks, Jk, Ik, history
#             # raise ValueError("Pivot is too small: " +str(pivot))
#         sign_pivot = np.sign(pivot)
#         sqrt_pivot = np.sqrt(np.abs(pivot))
#         # 4. Evaluate the associated row and column of the matrix
#         u1 = line_kernel(t_coord, s_coord[j1], Estar, kpower)
#         v1 = line_kernel(t_coord[i1], s_coord, Estar, kpower)
#         U[:, 0] = sign_pivot * u1 / sqrt_pivot
#         V[0, :] = v1 / sqrt_pivot
#         return U,V





#     else:
#         cl = PCACluster(s_coord)
#         level = int(np.log2(rank))
#         t_center = np.mean(t_coord, axis=0)
#         s_center = np.mean(s_coord, axis=0)
#         dir = t_center - s_center
#         dir_vector = dir
#         dir_vector[0] = -dir[1]
#         dir_vector[1] = dir[0]
#         # clusters = cl.split_to_level(s_coord, level) # arbitrary can provide a direction vector
#         clusters = cl.split_to_level(s_coord, level, dir_vector) # arbitrary can provide a direction vector

#         for ic,cluster in enumerate(clusters):
#             # 1. Find center of the cluster
#             cl_center = np.mean(cluster.coords, axis=0)
#             # 2. Find the closes point from t_coord to the center of the cluster
#             distances = np.linalg.norm(s_coord - cl_center, axis=1)
#             # 2.1 Find the index of the minimum distance
#             Jk[ic] = np.argmin(distances)

#             u_v = line_kernel(t_coord, s_coord[Jk[ic]], Estar, kpower)
#             for i in range(ic):
#                 for j in range(n):
#                     u_v[j] -= U[j, i] * V[i, Jk[ic]]
#             _, i_c = find_max_vector_element(u_v, Ik)
#             Ik[ic] = i_c
#             v_u = line_kernel(t_coord[i_c], s_coord, Estar, kpower)
#             for i in range(m):
#                 for j in range(ic):
#                     v_u[i] -= U[i_c, j] * V[j, i]
#             pivot = u_v[i_c]
#             if abs(pivot) < pivot_tol:
#                 print("Warning: Pivot is too small: ", pivot, ", stop at rank = ", ic)
#                 return U, V
#                 # raise ValueError("Pivot is too small: " +str(pivot))
#             sign_pivot = np.sign(pivot)
#             sqrt_pivot = np.sqrt(np.abs(pivot))

#             U[:, ic] = u_v * sign_pivot / sqrt_pivot
#             V[ic, :] = v_u / sqrt_pivot

#         return U, V

