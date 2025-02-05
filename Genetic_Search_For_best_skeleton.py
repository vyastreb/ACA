"""
    Python script for the search of the best skeleton for the ACA-GP algorithm using a kind of genetic algorithm, which probes all possible skeletons for every rank, selects the best one and proceeds to the next rank.

    Warning: the script takes a lot of time because the complexity of such algorithm is O(kn^2m^2).

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: May 2024 - Feb 2025
    License: BSD 3-Clause
    AI-LLM assistance: Claude (Claude 3.5 Sonnet, Anthropic) and GPT4o (OpenAI)
"""

import ACA_genetic as aca

import numpy as np
import scipy.linalg as la
from numba import jit
import os, sys
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 


from dataclasses import dataclass, field
from typing import Any, Callable, Tuple
from scipy.spatial import cKDTree
import uuid
import json
import time

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "lmodern",
    "pgf.rcfonts": False,  # Use the LaTeX preamble settings from the document
    "pgf.preamble": (
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage{lmodern}"  # Ensure Latin Modern is used
    )
})


def validate_field(name: str, default_value: Any, validator: Callable[[Any], bool]):
    def field_validator(value: Any):
        if not validator(value):
            raise ValueError(f"Invalid value for {name}: {value}")
        return value
    return field(default=default_value, metadata={'validator': field_validator})

@dataclass(frozen=True)
class MainConfig:
    """
    Parameters:
        N,M (int):                              Number of points in the target and source clouds
        xi (float):                             Aspect ratio of the target cloud (rectangular area)
        Ntry (int):                             Number of trials to get statistics on the error
        target_distance (float):                Target (true) distance between the clouds
        target_distance_tol (float):            Tolerance for the distance between the clouds
        max_rank (int):                         Maximum rank for the ACA-GP/ACA algorithm
        kernel_decay (int):                     The exponent (k>0) of the power-law decay of the kernel 1/x^k
        rank3treatment (bool):                  Ad hoc procedure to handle rank 3 approximation in a special way
        convex_hull_dist (Tuple[str,float]):    Type and value of the factor to multiply the distance to the convex hull in pivot selection : ("const",1) or ("linear",1)
        E (float):                              The kernel factor
        tol (float):                            Tolerance for the ACA-GP/ACA algorithm
        min_pivot (float):                      Minimum value for the pivot in the ACA-GP/ACA algorithm
        sigma (float):                          Standard deviation for the normal distribution of points if "normal" distribution is chosen
        distribution_type (str):                A "Uniform" or "normal" distribution of points
        ifSVD (bool):                           If True, compute the SVD approximation
        Plot_cloud (bool):                      If True, plot the clouds of points (for testing needs)
        Plot (bool):                            If True, plot results including clouds of points (for testing needs)
        filename_prefix (str):                  Prefix of the filename to save the results (complemented with parameters)

    Comment:
        Using field ensures that these variables cannot be changed (similar to const in C++ or frozen in Python)
    """
    N: int                              = validate_field('N',                       20,            lambda x: x > 0)
    M: int                              = validate_field('M',                       30,            lambda x: x > 0)
    xi: float                           = validate_field('xi',                      0.5,            lambda x: 0 < x <= 1.)
    Ntry: int                           = validate_field('Ntry',                    1,              lambda x: x > 0)
    target_distance: float              = validate_field('target_distance',         1.5,            lambda x: x >= 0.5)
    target_distance_tol: float          = validate_field('target_distance_tol',     0.1,            lambda x: x > 0)
    max_rank: int                       = validate_field('max_rank',                15,             lambda x: x > 0)
    kernel_decay: int                   = validate_field('kernel_decay',            1,              lambda x: x > 0)
    rank3treatment: bool                = validate_field('rank3treatment',          False,          lambda x: isinstance(x, bool))
    convex_hull_dist: Tuple[str,float]  = validate_field('convex_hull_dist',       ("const",1),     lambda x: isinstance(x, tuple) \
                                                         and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (int, float)))
    E: float                            = validate_field('E',                       1e3,            lambda x: x > 0)
    tol: float                          = validate_field('tol',                     1e-20,          lambda x: 0 < x < 0.1)
    min_pivot: float                    = validate_field('min_pivot',               1e-20,          lambda x: 0 < x < 0.1)
    sigma: float                        = validate_field('sigma',                   1.,             lambda x: x > 0)
    distribution_type: str              = validate_field('distribution_type',       "uniform",      lambda x: x in ["uniform", "normal"])
    ifSVD: bool                         = validate_field('ifSVD',                   False,          lambda x: isinstance(x, bool))
    Plot_cloud:bool                     = validate_field('Plot_cloud',              False,          lambda x: isinstance(x, bool))
    Plot: bool                          = validate_field('Plot',                    False,          lambda x: isinstance(x, bool))
    filename_prefix: str                = validate_field('filename_prefix',         "ACA_GP_data",  lambda x: isinstance(x, str))

    def __post_init__(self):
        for field_name, field_value in self.__dataclass_fields__.items():
            if 'validator' in field_value.metadata:
                validator = field_value.metadata['validator']
                value = getattr(self, field_name)
                object.__setattr__(self, field_name, validator(value))

        # Additional validation for relative values
        # if self.target_distance - self.target_distance_tol < 0.5:
        #     raise ValueError(f"target_distance_tol ({self.target_distance_tol}) must be smaller than 0.5 - target_distance ({0.5-self.target_distance})")
        if self.convex_hull_dist[0] not in ["const", "linear"]:
            raise ValueError(f"Invalid type of convex_hull_dist: {self.convex_hull_dist[0]}")
        if self.convex_hull_dist[1] <= 0:
            raise ValueError(f"Invalid value for convex_hull_dist: {self.convex_hull_dist[1]}")

    def get_json(self):
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__.keys()}

# @jit(nopython=True)
# def true_distance_between_clouds(cloud1, cloud2):
#     """
#     Description:
#         Measures the exact distance between two clouds of points.
#         Warning 1: O(N*M) complexity.
#         Warning 2: there's no check whether cloud penetrate or not.

#     Arguments:
#     cloud1, cloud2 (numpy.ndarray): Separate clouds of points

#     Returns:
#         min_distance (float): The distance between the two clouds of points.
#     """

#     min_distance = np.inf
#     for point1 in cloud1:
#         for point2 in cloud2:
#             distance = np.linalg.norm(point1 - point2)
#             if distance < min_distance:
#                 p1 = point1
#                 p2 = point2
#                 min_distance = distance
#     return min_distance

def detect_cloud_penetration(cloud1, cloud2, tolerance=1e-6):
    """
    Detect if two point clouds penetrate each other.
    
    Args:
    cloud1, cloud2: np.array of shape (n_points, n_dimensions)
    tolerance: float, minimum separation distance to consider clouds as non-penetrating
    
    Returns:
    bool: True if clouds penetrate, False otherwise
    """
    # Combine both clouds
    combined_cloud = np.vstack((cloud1, cloud2))
    
    # Build a KD-tree for efficient nearest neighbor search
    tree = cKDTree(combined_cloud)
    
    # For each point in cloud1, find the nearest neighbor in the combined cloud
    distances, _ = tree.query(cloud1, k=2)  # k=2 to get the nearest non-self neighbor
    
    # Check if any point in cloud1 has its nearest non-self neighbor closer than the tolerance
    # and that neighbor is from cloud2
    for i, (d1, d2) in enumerate(distances):
        if d2 < tolerance and i != _[i, 1] - len(cloud1):
            return True
    
    # Repeat the process for cloud2
    distances, _ = tree.query(cloud2, k=2)
    for i, (d1, d2) in enumerate(distances):
        if d2 < tolerance and i + len(cloud1) != _[i, 1]:
            return True
    
    return False

def true_distance_between_clouds(cloud1, cloud2):
    if False and detect_cloud_penetration(cloud1, cloud2):
        return 0 # penetration
    
    # If no penetration, proceed with finding the minimum distance using k-d tree
    tree = cKDTree(cloud2)
    distances, _ = tree.query(cloud1)
    return np.min(distances)

def move_and_rotate_cloud(t_coord, dx, dy, angle):
    """
    Move and rotate the target cloud of points.
    
    Args:
    t_coord: np.array of shape (N, 2), coordinates of the target cloud
    DX: float, translation in x-direction
    DY: float, translation in y-direction
    DAngle: float, rotation angle in radians

    Returns:
    np.array: Transformed coordinates of the target cloud
    """

    # Translate to origin, rotate, then translate back and apply displacement
    center = np.mean(t_coord, axis=0)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    t_coord = t_coord - center  # Translate to origin
    t_coord = (rotation_matrix @ t_coord.T).T  # Rotate
    t_coord = t_coord + center  # Translate back
    t_coord = t_coord + np.array([dx, dy])  # Apply displacement

    return t_coord


def construct_clouds(N,M,dist,dist_tolerance,random=True, DX=0, DY=0, DAngle=0):
    # Create two clouds of points
    t_coord = np.zeros((N,2))
    s_coord = np.zeros((M,2))
    if random:
        for i in range(config.N):
            t_coord[i,0] = np.random.rand()
            t_coord[i,1] = config.xi*np.random.rand()
        for i in range(config.M):
            s_coord[i,0] = np.random.rand()
            s_coord[i,1] = config.xi*np.random.rand()
    else:
        Nx = int(np.sqrt(N/config.xi))
        Ny = int(np.sqrt(N*config.xi))
        for i in range(Nx):
            for j in range(Ny):
                t_coord[i*Ny+j,0] = i/Nx
                t_coord[i*Ny+j,1] = config.xi*j/Ny
        Mx = int(np.sqrt(M/config.xi))
        My = int(np.sqrt(M*config.xi))
        for i in range(Mx):
            for j in range(My):
                s_coord[i*My+j,0] = i/Mx
                s_coord[i*My+j,1] = config.xi*j/My
    # Rotate cloud1 by a random angle
    real_dist = 0
    t_coord_trial = np.copy(t_coord)
    angle = 0
    dist_factor = dist * 4

    # More efficient approach is to distance them along the line between centers.
    # Trial
    while real_dist == 0:
        angle = DAngle #
        dx = np.random.rand() * dist_factor
        dy = 0 # np.random.rand() * dist_factor
        # Calculate the center of the original rectangle
        center = np.array([0.5, config.xi/2])
        
        # Translate to origin, rotate, then translate back and apply displacement
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        t_coord_trial = t_coord - center  # Translate to origin
        t_coord_trial = (rotation_matrix @ t_coord_trial.T).T  # Rotate
        t_coord_trial = t_coord_trial + center  # Translate back
        t_coord_trial = t_coord_trial + np.array([dx, dy])  # Apply displacement
        
        real_dist = true_distance_between_clouds(s_coord, t_coord_trial)

    eps = 1e-3
    t_coord_center = np.mean(t_coord,axis=0)
    while abs(real_dist - dist) > eps * dist:
        t_coord_trial_center = np.mean(t_coord_trial,axis=0)
        s_coord_center = np.mean(s_coord,axis=0)
        dist_vector = t_coord_trial_center - s_coord_center
        t_coord_trial += (dist - real_dist) * dist_vector / np.linalg.norm(dist_vector)
        dx = (t_coord_trial_center - t_coord_center)[0] 
        dy = (t_coord_trial_center - t_coord_center)[1]
        real_dist = true_distance_between_clouds(s_coord, t_coord_trial)

    print(f"Real distance: {real_dist}")

    # print(f"Generated clouds in {time.time()-start_clouds:.2f} seconds")
    t_coord = t_coord_trial + np.array([DX,DY])
    print("Constructed clouds....")
    return t_coord, s_coord

def performance_test(config: MainConfig, t_coord, s_coord, iter_i=-1, iter_j=-1, U=None, V=None):

    # Compute the ACA-GP approximation
    U, V  = aca.aca_iterative(t_coord, s_coord, config.E, config.tol, config.max_rank, \
                                                     config.min_pivot, config.kernel_decay, iter_i, iter_j, U, V)
    return U,V

class SVDmatrix:
    def __init__(self, config: MainConfig, t_coord, s_coord):
        self.full_matrix = np.zeros((config.N,config.M))
        for i in range(config.N):
            self.full_matrix[i] = aca.line_kernel(t_coord[i], s_coord, config.E, config.kernel_decay)
        self.norm_full_matrix = np.linalg.norm(self.full_matrix,"fro")

        self.svd_error = np.zeros(config.max_rank)
        self.U_full, self.s_full, self.V_full = np.linalg.svd(self.full_matrix)
    def get_full_matrix(self):
        return self.full_matrix
    def get_svd_approximation(self, rank):
        if rank > self.svd_error.shape[0]:
            raise ValueError("Rank is too large")
        approx_matrix = np.dot(self.U_full[:,:rank],np.dot(np.diag(self.s_full[:rank]),self.V_full[:rank,:]))
        return approx_matrix

def main(config: MainConfig, t_coord, s_coord, matrix, uid):
    """
    Description:
        The main function to run `Ntry` performance tests for the ACA-GP algorithm.
        The function saves the results in an npz and plots relevant data.
    """
    N = config.N
    M = config.M

    full_matrix = matrix.get_full_matrix()
    norm_full_matrix = np.linalg.norm(full_matrix,"fro")
    svd_approx = [matrix.get_svd_approximation(i) for i in range(1,config.max_rank+1)]
    svd_error = np.array([np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix for approx_matrix in svd_approx])

    Iopt = np.zeros(config.max_rank, dtype=int)
    Jopt = np.zeros(config.max_rank, dtype=int)
    Uopt = None
    Vopt = None

    start = time.time()
    for ranks in range(config.max_rank):
        ACA_ITER = np.zeros((N,M)) 
        Jo = []
        for i in range(N):
            if i not in Iopt[:ranks]:
                for j in range(M):
                    if j not in Jopt[:ranks]:
                        if ranks == 0:
                            U,V = performance_test(config, t_coord, s_coord, i, j)
                        else:
                            U,V = performance_test(config, t_coord, s_coord, i, j, Uopt, Vopt)
                        approx_matrix = np.dot(U[:,:ranks+1],V[:ranks+1,:])
                        aca_iter_error = np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix
                        ACA_ITER[i,j] = aca_iter_error            

        if ranks > 0:
            ACA_ITER[Iopt[:ranks],:] = np.nan
            ACA_ITER[:,Jopt[:ranks]] = np.nan

        if ranks > 0:
            RESIDUAL_MATRIX = full_matrix - np.dot(Uopt[:,:ranks],Vopt[:ranks,:])
        else:
            RESIDUAL_MATRIX = full_matrix

        ij_opt = np.nanargmin(ACA_ITER)
        min_i, min_j = divmod(ij_opt,M)
        Iopt[ranks] = int(min_i)
        Jopt[ranks] = int(min_j)
        if ranks == 0:
            Uopt,Vopt = performance_test(config, t_coord, s_coord, min_i, min_j)
        else:
            Uopt,Vopt = performance_test(config, t_coord, s_coord, min_i, min_j, Uopt, Vopt)


        # normalized_error = ACA_ITER/svd_error[ranks]

        print(f"Rank = {ranks+1} --> Error = {ACA_ITER[min_i,min_j]:.3e} -- (i,j) = ({min_i},{min_j}), SVD error = {svd_error[ranks]:.3e}")

        np.savez(f"Genetic_approximation_with_Residual_error_rank_{ranks}_ID_{uid}.npz", ACA_error=ACA_ITER, SVD_error=svd_error[ranks], Iopt=Iopt[:ranks+1], Jopt=Jopt[:ranks+1],RESIDUAL_MATRIX=RESIDUAL_MATRIX)
    print(f"-- Time = {time.time()-start:.2f} seconds.")

if __name__ == "__main__":
    """
    Description:
        The script generates a plot of the relative error of the ACA-GP algorithm with respect to the ACA and SVD algorithms.
        The plot is saved as a PDF and PGF file.
        An `npz` file with the results is saved as "ACA_GP_error_wrt_SVD.npz"

    Arguments:
        seed (int): Seed for the random number generator

    Results:
        The script saves the results in an npz file with the following keys:
        - ACA_error: The error of the ACA-GP algorithm
        - SVD_error: The error of the SVD algorithm
        - Iopt: The optimal indices of the target cloud
        - Jopt: The optimal indices of the source cloud
        - RESIDUAL_MATRIX: The residual matrix
       It also saves the configuration of the two clouds in npz.
    """

    if len(sys.argv) != 2:
        print("Usage warning: requires a single argument <seed (int)>\nExiting...")
        exit(1)
    seed = int(sys.argv[1])
    np.random.seed(seed)


    XI = 0.5
    try:
        config = MainConfig(
            N=50, M=50, xi=XI, Ntry=1,
            target_distance = .5, # 1.5
            target_distance_tol = 0.1, max_rank=6,
            kernel_decay=1, rank3treatment=True,
            convex_hull_dist=("const",1.),
            E=1e3, tol = 1e-20, min_pivot=1e-20,
            sigma=1., distribution_type="uniform",
            ifSVD=True, Plot_cloud=False, Plot=True,
            filename_prefix="Manual_Skeleton"
        )
    except ValueError as e:
        print(f"Input validation failed: {e}")
        exit(1)

    DX = 0.
    DY = 1.
    Angle = -np.pi/3.

    t_coord_0, s_coord = construct_clouds(config.N, config.M, config.target_distance, config.target_distance_tol, random=False, DX=0, DY=0, DAngle=0)
    t_coord = move_and_rotate_cloud(t_coord_0, DX, DY, Angle)
    uid = str(uuid.uuid4())[:8]
    np.savez(f"clouds_xi_{XI}_ID_{uid}.npz", t_coord=t_coord, s_coord=s_coord)

    matrix = SVDmatrix(config, t_coord, s_coord)
    main(config, t_coord, s_coord, matrix, uid)

