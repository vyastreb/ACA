import numpy as np
import time
import ACAs as aca

# Parameters
algorithm = "ACA-GP"  # "ACA" or "ACA-GP"
seed = 128
np.random.seed(seed)

# Number of points
N = 1000
M = 1500

# Low-rank approximation parameters
tolerance = 1e-3
min_pivot = 1e-12
max_rank = min(min(N, M), 15)
Green_kernel_power = 1
Green_kernel_factor = 1
# ACA-GP specific parameters
central_fraction = 0.1
square_shape = True # If True, the shape of the target and source clouds should be square-like, if False, the clouds can have arbitrary shape.

# Create clouds of points
x_coord = np.random.rand(N, 2) + np.array([1.5, 0.5])
y_coord = np.random.rand(M, 2)

# Run ACA algorithm
start_time = time.time()
if algorithm == "ACA":
    U, V, error, rank, _, _, _ = \
        aca.aca(x_coord, y_coord, tolerance, max_rank, min_pivot, Green_kernel_factor, Green_kernel_power)
elif algorithm == "ACA-GP":
    U, V, error, rank, _, _, _, _, _ = \
        aca.aca_gp(x_coord, y_coord, tolerance, max_rank, min_pivot, Green_kernel_factor, Green_kernel_power,
                    central_fraction, square_shape)
else:
    raise ValueError("Invalid algorithm")
end_time = time.time()

# Compute approximation and error
approx_matrix = np.dot(U, V)
full_matrix = np.array([aca.line_kernel(x, y_coord, Green_kernel_factor, Green_kernel_power) for x in x_coord])
norm_full_matrix = np.linalg.norm(full_matrix, "fro")
aca_error = np.linalg.norm(approx_matrix - full_matrix, "fro") / norm_full_matrix

# Print results
print(f"/ Algorithm: {algorithm} for {N} x {M} clouds")
print(f" Time (s):              {end_time - start_time:<10.4f}")
print(f" Approximation rank:    {rank+1:<10d}")
print(f" Storage fraction (%):  {(rank+1)*(N+M)/(N*M)*100:<10.2f}")
print(f" Requested tolerance:   {tolerance:<10.2e}")
print(f" Approximate error:     {error:<10.2e}")
print(f" True relative error:   {aca_error:<10.2e}")
