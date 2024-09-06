"""
    Testing script for 
    1. Adaptive Cross Approximation with Geometrical Pivot selection (ACA-GP) algorithm
    2. Adaptive Cross Approximation with Principal Component Analysis (PCA) based pivot selection (PCA-CA) algorithm.
    compared to 
    a. Classical ACA with a random choice of pivot
    b. Singular Value Decomposition (SVD) which is expensive but the most accurate low-rank approximation method.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, MINES Paris, PSL University, Evry/Paris, France
    Date: May-Aug 2024
    License: BSD 3-Clause
    AI-LLM assistance: Claude (Claude 3.5 Sonnet, Anthropic) and GPT4o (OpenAI)
"""

import ACAs as aca

import numpy as np
import scipy.linalg as la
from numba import jit
import os, sys
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple
from scipy.spatial import cKDTree
import uuid
import json
import time

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
    N: int                              = validate_field('N',                       200,            lambda x: x > 0)
    M: int                              = validate_field('M',                       300,            lambda x: x > 0)
    xi: float                           = validate_field('xi',                      0.5,            lambda x: 0 < x <= 1.)
    Ntry: int                           = validate_field('Ntry',                    1,              lambda x: x > 0)
    target_distance: float              = validate_field('target_distance',         1.5,            lambda x: x > 0.5)
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
        if self.target_distance - self.target_distance_tol < 0.5:
            raise ValueError(f"target_distance_tol ({self.dist_factor}) must be smaller than 0.5 - target_distance ({0.5-self.target_distance})")
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
    if detect_cloud_penetration(cloud1, cloud2):
        return 0 # penetration
    
    # If no penetration, proceed with finding the minimum distance using k-d tree
    tree = cKDTree(cloud2)
    distances, _ = tree.query(cloud1)
    return np.min(distances)


def performance_test(config: MainConfig):
    """
    Description:
        It generate two random rectangular clouds of points with aspect ratio xi in 2D either with a uniform or normal distributions.
        The clouds are generated in such a way that the distance between the clouds is in the interval [dist_min, dist_max].
        The ACA-GP algorithm is applied to the clouds of points and the relative error of the approximation is computed.
        ACA and, if SVD == True, SVD algorithms are also applied to the clouds of points and the relative error of the approximation is computed.

    Output:
        real_dist (float): Real distance between the clouds
        aca_error (np.array): Relative error of the ACA algorithm
        aca_gp_error (np.array): Relative error of the ACA-GP algorithm
        svd_error (np.array): Relative error of the SVD algorithm
        history (np.array): History of the ACA-GP algorithm.
    """

    start_clouds = time.time()
    # Create two clouds of points
    t_coord = np.zeros((config.N,2))
    s_coord = np.zeros((config.M,2))
    if config.distribution_type == "uniform":
        for i in range(config.N):
            t_coord[i,0] = np.random.rand()
            t_coord[i,1] = config.xi*np.random.rand()
        for i in range(config.M):
            s_coord[i,0] = np.random.rand()
            s_coord[i,1] = config.xi*np.random.rand()
    elif config.distribution_type == "normal":
        t_center = np.zeros(2)
        s_center = np.zeros(2)
        t_center[0] = np.random.rand()
        t_center[1] = config.xi * np.random.rand()
        s_center[0] = np.random.rand()
        s_center[1] = config.xi * np.random.rand()
        for i in range(config.N):
            t_coord[i] = np.array([-1,1])
            while t_coord[i,0] < 0 or t_coord[i,0] > 1:
                t_coord[i,0] = np.random.normal(t_center[0],config.sigma)
            while t_coord[i,1] < 0 or t_coord[i,1] > config.xi:
                t_coord[i,1] = np.random.normal(t_center[1],config.sigma)
        for i in range(config.M):
            s_coord[i] = np.array([-1,1])
            while s_coord[i,0] < 0 or s_coord[i,0] > 1:
                s_coord[i,0] = np.random.normal(s_center[0],config.sigma)
            while s_coord[i,1] < 0 or s_coord[i,1] > config.xi:
                s_coord[i,1] = np.random.normal(s_center[1],config.sigma)

    # Rotate cloud1 by a random angle
    real_dist = 0
    t_coord_trial = np.copy(t_coord)
    angle = 0
    dist_factor = config.target_distance * 3
    # Iterate before you get the distance in the interval [dist_min, dist_max]
    while real_dist < config.target_distance - config.target_distance_tol or \
        real_dist > config.target_distance + config.target_distance_tol:

        angle = 2 * np.pi * np.random.rand()
        dx = np.random.rand() * dist_factor
        dy = np.random.rand() * dist_factor
        
        # Calculate the center of the original rectangle
        center = np.array([0.5, config.xi/2])
        
        # Translate to origin, rotate, then translate back and apply displacement
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        t_coord_trial = t_coord - center  # Translate to origin
        t_coord_trial = (rotation_matrix @ t_coord_trial.T).T  # Rotate
        t_coord_trial = t_coord_trial + center  # Translate back
        t_coord_trial = t_coord_trial + np.array([dx, dy])  # Apply displacement
        
        real_dist = true_distance_between_clouds(t_coord, t_coord_trial)

    t_coord = t_coord_trial
    # print(f"Generated clouds in {time.time()-start_clouds:.2f} seconds")

    if config.Plot_cloud:
        fig,ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        plt.title("Distance = {0:.3f}, Prime distance = {1:.3f}".format(real_dist,prime_dist))
        plt.scatter(t_coord[:,0],t_coord[:,1],s=2.5,c="r",label="t_coord",alpha=0.3)
        plt.scatter(s_coord[:,0],s_coord[:,1],s=2.5,c="b",label="s_coord",alpha=0.3)
        for j in range(len(Jk)):
            plt.plot(s_coord[Jk[j],0],s_coord[Jk[j],1],"o",markersize=2.5,c="k",zorder=10)
            plt.text(s_coord[Jk[j],0],s_coord[Jk[j],1],f"{j+1}",fontsize=12)
        for i in range(len(Jk)):
            plt.plot(t_coord[Ik[i],0],t_coord[Ik[i],1],"o",markersize=2.5,c="k",zorder=10)
            plt.text(t_coord[Ik[i],0],t_coord[Ik[i],1],f"{i+1}",fontsize=12)
            
        rect2 = patches.Rectangle((0,0), 1, config.xi,linewidth=1,edgecolor='#990000',linestyle="dashed",facecolor='none', zorder=3) 
        ax.add_patch(rect2)

        # Create a rotated and translated rectangle
        rect3 = patches.Rectangle((0, 0), 1, config.xi, linewidth=1, edgecolor='navy', linestyle="dashed", facecolor='none', zorder=3)
        t = (mpl.transforms.Affine2D()
            .translate(-0.5, -config.xi/2)  # Translate to origin
            .rotate(angle)  # Rotate (angle is already in radians)
            .translate(0.5, config.xi/2)  # Translate back
            .translate(dx, dy))  # Apply displacement
        t = t + ax.transData
        rect3.set_transform(t)
        ax.add_patch(rect3)

        plt.legend()
        fig.savefig("clouds.pdf")
        # plt.show()
    # Potentially not needed
    # mean_t_coord = np.mean(t_coord,axis=0)
    # mean_s_coord = np.mean(s_coord,axis=0)
    # prime_dist = la.norm(mean_t_coord - mean_s_coord) - np.sqrt(1+config.xi**2)

    start_full_matrix = time.time()
    # Compute the full matrix for the given clouds for comparison purposes
    full_matrix = np.zeros((config.N,config.M))
    for i in range(config.N):
        full_matrix[i] = aca.line_kernel(t_coord[i], s_coord, config.E, config.kernel_decay)
    norm_full_matrix = np.linalg.norm(full_matrix,"fro")
    # print(f"Computed full matrix in {time.time()-start_full_matrix:.2f} seconds")

    # Compute the ACA-GP approximation
    start_aca_gp_computation = time.time()
    U, V, error, rank, Jk, Ik, history  = aca.aca_gp(t_coord, s_coord, config.E, config.tol, config.max_rank, \
                                                     config.min_pivot, config.kernel_decay, config.rank3treatment, config.convex_hull_dist)
    acagp_time = time.time()-start_aca_gp_computation
    # print(f"Computed ACA-GP approximation in {acagp_time:.2f} seconds")

    # Compute the ACA approximation for comparison purposes
    start_aca_computation = time.time()
    Uc,Vc,_,rankc,_,_,_ = aca.aca(   t_coord, s_coord, config.E, config.tol, config.max_rank, \
                                  config.min_pivot, config.kernel_decay)
    aca_time = time.time()-start_aca_computation
    # print(f"Computed ACA approximation in {aca_time:.2f} seconds")

    rank = min(rank,rankc)


    # To store relative errors: 
    # Frobebius norm of the difference between the full matrix and its low-rank approximation normalized by the Frobenius norm of the full matrix
    aca_gp_error = np.zeros(rank)  
    aca_error = np.zeros(rank)  
    svd_error = np.zeros(rank)

    # Construct SVD if needed and compute the error
    svd_time = 0
    if config.ifSVD:
        start_svd_computation = time.time()
        U_full, s_full, V_full = np.linalg.svd(full_matrix)
        svd_time = time.time()-start_svd_computation
        # print(f"Computed SVD approximation in {time.time()-start_svd_computation:.2f} seconds")
        for i in range(1,rank+1):
            approx_matrix = np.dot(U_full[:,:i],np.dot(np.diag(s_full[:i]),V_full[:i,:]))
            svd_error[i-1] = np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix

    # Compute relative errors for ACA and ACA-GP
    for i in range(1,rank+1):
        aca_approx_matrix = np.dot(Uc[:,:i],Vc[:i,:])
        aca_error[i-1] = np.linalg.norm(aca_approx_matrix - full_matrix,"fro")/norm_full_matrix
        aca_gp_approx_matrix = np.dot(U[:,:i],V[:i,:])
        aca_gp_error[i-1] = np.linalg.norm(aca_gp_approx_matrix - full_matrix,"fro")/norm_full_matrix

    return real_dist, aca_error, aca_gp_error, svd_error, history[:,2], history[:,3], acagp_time, aca_time, svd_time

def main(config: MainConfig):
    """
    Description:
        The main function to run `Ntry` performance tests for the ACA-GP algorithm.
        The function saves the results in an npz and plots relevant data.
    """

    DIST = []
    ACA = []
    ACA_GP = []
    SVD = []
    ACA_GP_time = []
    ACA_time = []
    SVD_time = []
    for i in range(config.Ntry):
        dist, aca_error, aca_gp_error, svd_error, history, pivot, acagp_time, aca_time, svd_time = performance_test(config)
        DIST.append(dist)
        ACA.append(aca_error)
        ACA_GP.append(aca_gp_error)
        SVD.append(svd_error)
        ACA_GP_time.append(acagp_time)
        ACA_time.append(aca_time)
        SVD_time.append(svd_time)
        if (i+1) % 10 == 0:
            print("/ Completed {0:3d} out of {1:3d} trials".format(i+1,config.Ntry))

    # uniformize list lengths by padding
    max_length = max(len(arr) for arr in ACA)
    for i in range(config.Ntry):
        if len(ACA[i]) < max_length:
            ACA[i] = np.pad(ACA[i], (0, max_length - len(ACA[i])), 'constant')
            ACA_GP[i] = np.pad(ACA_GP[i], (0, max_length - len(ACA_GP[i])), 'constant')
            if config.ifSVD:
                SVD[i] = np.pad(SVD[i], (0, max_length - len(SVD[i])), 'constant')

    DIST = np.array(DIST)
    ACA = np.array(ACA)
    ACA_GP = np.array(ACA_GP)

    unique_id = str(uuid.uuid4())[:8]
    filename = config.filename_prefix + f"_Dist_{config.target_distance}_distribution_{config.distribution_type}_xi_{config.xi}_ID_{unique_id}.npz"

    if config.ifSVD:
        SVD = np.array(SVD)
        np.savez(filename, DIST=DIST, ACA=ACA, ACA_GP=ACA_GP, SVD=SVD, ACA_GP_time=ACA_GP_time, ACA_time=ACA_time, SVD_time=SVD_time)
    else:
        np.savez(filename, DIST=DIST, ACA=ACA, ACA_GP=ACA_GP, ACA_GP_time=ACA_GP_time, ACA_time=ACA_time)
    json_filename = filename.replace(".npz", ".json")
    with open(json_filename, 'w') as f:
        json.dump(config.get_json(), f, indent=4)

    if config.Plot:
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

        ACA_log = np.log10(ACA)
        ACA_GP_log = np.log10(ACA_GP)

        mean_ACA_log = np.mean(ACA_log, axis=0)
        mean_ACA_GP_log = np.mean(ACA_GP_log, axis=0)

        std_ACA_log = np.std(ACA_log, axis=0)
        std_ACA_GP_log = np.std(ACA_GP_log, axis=0)

        if config.ifSVD:
            SVD_log = np.log10(SVD)
            mean_SVD_log = np.mean(SVD_log, axis=0)
            std_SVD_log = np.std(SVD_log, axis=0)        

        # Plot relative errors
        fig,ax = plt.subplots(1,1,figsize=(5,3.5))
        ranks = np.arange(1,max_length+1)
        tics = ranks
        ax.set_xticks(tics)
        plt.grid()
        plt.title("Distance = {0:.2f}".format(config.target_distance))
        plt.xlim(1,max_length)
        # plt.ylim(0.5*np.min(ACA),np.max(ACA)*1.5)
        plt.ylim(1e-9,0.1)
        plt.ylabel(r"Relative error, $\|A_k - A\|_F \; / \; \|A\|_F$")
        plt.xlabel("Approximation rank, $k$")
        plt.yscale("log")

        plt.fill_between(ranks,10**(mean_ACA_log+std_ACA_log),10**(mean_ACA_log-std_ACA_log),color="r",alpha=0.2)
        plt.plot(ranks,10**mean_ACA_log,"v-",markersize=5, color="r", markeredgewidth=0.5, markeredgecolor='k',label="ACA+")
        
        plt.fill_between(ranks,10**(mean_ACA_GP_log+std_ACA_GP_log),10**(mean_ACA_GP_log-std_ACA_GP_log),color="g",alpha=0.2)
        plt.plot(ranks,10**mean_ACA_GP_log,"*-",markersize=8,markeredgewidth=0.5, color="g", markeredgecolor='k',label="ACA-GP, current method", zorder=3)

        if config.ifSVD:
            plt.fill_between(ranks,10**(mean_SVD_log+std_SVD_log),10**(mean_SVD_log-std_SVD_log),color="b",alpha=0.2)
            plt.plot(ranks,10**mean_SVD_log,"o-",markersize=5, markeredgewidth=0.5, color="b", markeredgecolor='k',label="SVD")

        plt.legend()
        # plt.show()
        figure_name = f"ACA_GP_error_Dist_{config.target_distance}_distribution_{config.distribution_type}_xi_{config.xi}_convexhull_{config.convex_hull_dist[0]}_factor_{config.convex_hull_dist[1]}_ID_{unique_id}"
        fig.savefig(figure_name+".pdf")
        fig.savefig(figure_name+".pgf")
        plt.close()

        # Plot accuracy gain of ACA-GP over ACA
        fig,ax = plt.subplots(1,1,figsize=(5,3.5))
        ranks = np.arange(1,max_length+1)
        tics = ranks
        ax.set_xticks(tics)
        plt.grid()
        plt.title("Distance = {0:.2f}, convex hull ({1}) factor = {2:.0f}".format(\
            config.target_distance, config.convex_hull_dist[0], config.convex_hull_dist[1]))
        plt.xlim(1,max_length)
        plt.ylabel(r"Accuracy factor, $E_{\mathrm{ACA}}/E_{\mathrm{ACA-GP}}$")
        plt.xlabel("Approximation rank, $k$")
        # plt.yscale("log")

        ratio = 10**(mean_ACA_log-mean_ACA_GP_log)
        # plt.ylim(0,1.1*np.max(ratio))
        plt.ylim(0,4)

        # Plotting
        # plt.fill_between(ranks, lower_bound/10**mean_ACA_GP_log, upper_bound/10**mean_ACA_GP_log, color="r", alpha=0.2)
        plt.plot(ranks, ratio, "o-", markersize=5, color="r", markeredgewidth=0.5, markeredgecolor='k', label="ACA-GP accuracy gain wrt ACA")
        average = np.mean(ratio)
        plt.axhline(y=average, color='#660000', linestyle='--', label=f"Average accuracy gain: {average:.2f}")
        plt.axhline(y=1, color='k', linestyle='-', label="Equivalent accuracy")

        plt.legend()
        # plt.show()
        figure_name = f"ACA_GP_accuracy_gain_Dist_{config.target_distance}_distribution_{config.distribution_type}_xi_{config.xi}_convexhull_{config.convex_hull_dist[0]}_factor_{config.convex_hull_dist[1]}_ID_{unique_id}"
        fig.savefig(figure_name+".pdf")
        fig.savefig(figure_name+".pgf")
        plt.close()

if __name__ == "__main__":
    """
    Description:
        The script generates a plot of the relative error of the ACA-GP algorithm with respect to the ACA and SVD algorithms.
        The plot is saved as a PDF and PGF file.
        An `npz` file with the results is saved as "ACA_GP_error_wrt_SVD.npz"

    Arguments:
        seed (int): Seed for the random number generator
    """

    if len(sys.argv) != 2:
        print("Usage warning: requires a single argument <seed (int)>\nExiting...")
        exit(1)
    seed = int(sys.argv[1])
    np.random.seed(seed)

    # Target_distances = [1, 1.5, 2, 2.5, 5]
    Target_distances = [1.5, 2, 5]
    Xi = [0.25, 0.5, 1]
    Convex_hull_distance_factors = [1] #, 2, 3, 4] #, 5, 6, 7, 8]
    start_tests = time.time()
    convex_hull_dist_type = "linear" # "const" or "linear"
    Rank3 = [True] #[True, False]

    for tdist in Target_distances:
        for xi_ in Xi:
            for rank3treatment_ in Rank3:
                for ch_dist in Convex_hull_distance_factors:
                    try:
                        print(f"> Running with parameters: dist={tdist:.2f}, xi={xi_:.2f}, rank3treatment={rank3treatment_}, convex_hull_dist type={convex_hull_dist_type}, factor={ch_dist:.2f}")
                        config = MainConfig(
                            N=200, M=300, xi=xi_, Ntry=50,
                            target_distance = tdist,
                            target_distance_tol = 0.1, max_rank=5, 
                            kernel_decay=1, rank3treatment=rank3treatment_,
                            convex_hull_dist=(convex_hull_dist_type,ch_dist),
                            E=1e3, tol = 1e-20, min_pivot=1e-20,
                            sigma=1., distribution_type="uniform",
                            # FIXME: ifSVD should be True if you have not data, but it is the most time consuming
                            ifSVD=False, Plot_cloud=False, Plot=False,
                            filename_prefix="ACA_GP_data"
                        )
                    except ValueError as e:
                        print(f"Input validation failed: {e}")
                        exit(1)

                    main(config)
    print(f"Completed all tests in {time.time()-start_tests:.2f} seconds")

