import numpy as np
import ACAs as aca

# Create two separate clouds of points
N = 200
M = 300
t_coord = np.random.rand(N, 2)
s_coord = np.random.rand(M, 2)
t_coord += np.array([1.5,0])

# Set low-rank approximation parameters
tolerance = 1e-3
min_pivot = 1e-10
max_rank = min(N, M)
green_kernel_power = 1
green_kernel_factor = 1

# Run the ACA algorithm
U,V,error,rank,Jk,Ik,history = aca.aca_gp(t_coord, s_coord, green_kernel_factor, tolerance, max_rank, min_pivot, green_kernel_power)

# Compute the approximated matrix
approx_matrix = np.dot(U,V)

# Compute the error
full_matrix = np.zeros((N, M))
for i in range(N):
    full_matrix[i] = aca.line_kernel(t_coord[i], s_coord, green_kernel_factor, green_kernel_power)
norm_full_matrix = np.linalg.norm(full_matrix,"fro")
aca_gp_error = np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix

print("\n/ ACA-GP algorithm")
print(" Approximation rank: {0:2d} ".format(rank))
print(" Storage fraction:   {0:.2f} %".format(100*rank*(N+M)/(N*M)))
print(" Relative error:     {0:.2e} ".format(aca_gp_error))
print(" Approximate error:  {0:.2e} < {1:.2e}".format(error, tolerance))