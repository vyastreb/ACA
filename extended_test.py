"""
    Test script for the ACA and ACA-GP algorithms.

    The script creates two separate clouds of points and computes the low-rank approximation of the matrix A' with a requested tolerance using the ACA and ACA-GP algorithms.

    Author: Vladislav A. Yastrebov
    Affiliation: CNRS, Mines Paris, PSL University, Evry/Paris, France
    Date: May 2024 - Feb 2025
    License: BSD 3-Clause
"""

import numpy as np
import time
import ACAs as aca


# Parameters
algorithm = "ACA-GP" # "ACA" or "ACA-GP"
seed = 128
np.random.seed(seed)

# ACA-GP specific parameters
square_shape = True # If True, the shape of the target and source clouds should be square-like, if False, the clouds can have arbitrary shape.
central_fraction = 0.1

# Number of points in clouds X and Y
N = 1000
M = 1500
plot_clouds = True

# Set low-rank approximation parameters
tolerance = 1e-3
min_pivot = 1e-12
max_rank = min(min(N, M),15)
Green_kernel_power = 1
Green_kernel_factor = 1

# Create two separate clouds of points
x_coord = np.random.rand(N, 2) # Source cloud in a 1x1 square
y_coord = np.random.rand(M, 2) # Target cloud in a 1x1 square
Delta_X = 1.5
Delta_Y = .5
x_coord += np.array([Delta_X,Delta_Y])

# Run the ACA algorithm
start_time = time.time()
if algorithm == "ACA":
    U,V,error,rank,Jk,Ik,history = \
        aca.aca(x_coord, y_coord, tolerance, \
                    max_rank, min_pivot, Green_kernel_factor, Green_kernel_power)
elif algorithm == "ACA-GP":
    U,V,error,rank,Jk,Ik,history,central_fraction_s,central_fraction_t = \
        aca.aca_gp(x_coord, y_coord, tolerance, \
                    max_rank, min_pivot, Green_kernel_factor, Green_kernel_power, \
                    central_fraction, square_shape)
else:
    raise ValueError("Invalid algorithm")
end_time = time.time()

# Compute the approximated matrix A'
approx_matrix = np.dot(U,V)

# Compute the error
full_matrix = np.zeros((N, M))
for i in range(N):
    full_matrix[i] = aca.line_kernel(x_coord[i], y_coord, Green_kernel_factor, Green_kernel_power)
norm_full_matrix = np.linalg.norm(full_matrix,"fro")
aca_error = np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix

# Print the results
print("/ Algorithm: {0} for {1} x {2} clouds".format(algorithm,N,M))
print(" Time (s):              {0:<10.4f}".format(end_time - start_time))
print(" Approximation rank:    {0:<10d}".format(rank+1))
print(" Storage fraction (%):  {0:<10.2f}".format((rank+1)*(N+M)/(N*M)*100))
print(" Requested tolerance:   {0:<10.2e}".format(tolerance))
print(" Approximate error:     {0:<10.2e}".format(error))
print(" True relative error:   {0:<10.2e}".format(aca_error))


if plot_clouds:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Estimate diam of the point clouds
    center_x = np.mean(x_coord, axis=0)
    center_y = np.mean(y_coord, axis=0)
    diam_x = 2 * np.max(np.linalg.norm(x_coord - center_x, axis=1))
    diam_y = 2 * np.max(np.linalg.norm(y_coord - center_y, axis=1))

    # Keep only positive arguments of Jk
    dtext = 0.01
    fig,ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    plt.scatter(x_coord[:,0],x_coord[:,1],s=2.5,c="r",label="X cloud",alpha=0.1)
    plt.scatter(y_coord[:,0],y_coord[:,1],s=2.5,c="b",label="Y cloud",alpha=0.1)
    for j in range(len(Jk)):
        plt.plot(y_coord[Jk[j],0],y_coord[Jk[j],1],"o",markersize=2.5,c="k",zorder=10)
        plt.text(y_coord[Jk[j],0]+dtext,y_coord[Jk[j],1]+dtext,f"{j+1}",fontsize=12)
    for i in range(len(Jk)):
        plt.plot(x_coord[Ik[i],0],x_coord[Ik[i],1],"o",markersize=2.5,c="k",zorder=10)
        plt.text(x_coord[Ik[i],0]+dtext,x_coord[Ik[i],1]+dtext,f"{i+1}",fontsize=12)
        
    rect2 = patches.Rectangle((0,0), 1, 1,linewidth=1,edgecolor='#999999',linestyle="dashed",facecolor='none', zorder=3) 
    ax.add_patch(rect2)

    rect3 = patches.Rectangle((Delta_X,Delta_Y), 1, 1,linewidth=1,edgecolor='#999999',linestyle="dashed",facecolor='none', zorder=3)
    ax.add_patch(rect3)

    # Draw circles around centers of clouds with radius central_fraction
    center_x = np.mean(x_coord, axis=0)
    center_y = np.mean(y_coord, axis=0)
    
    if algorithm == "ACA-GP":
        circle_x = plt.Circle((x_coord[Ik[0],0], x_coord[Ik[0],1]), central_fraction * diam_x, 
                             fill=False, linestyle='--', color='r', alpha=0.5,label="Central subset X cloud")
        circle_y = plt.Circle((y_coord[Jk[0],0], y_coord[Jk[0],1]), central_fraction * diam_y,
                             fill=False, linestyle='--', color='b', alpha=0.5,label="Central subset Y cloud")
    
    ax.add_patch(circle_x)
    ax.add_patch(circle_y)

    plt.legend()
    plt.show()
    fig.savefig(f"{algorithm}_clouds.png", dpi=300)

