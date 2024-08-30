# Adaptive Cross Approximation: ACA and ACA-GP

This repository contains the implementation of the Adaptive Cross Approximation (ACA) and the Adaptive Cross Approximation with Geometrical Pivot selection (ACA-GP) algorithms. The ACA algorithm is a low-rank matrix approximation algorithm that is particularly well-suited for matrices that arise in the context of integral equations. The ACA-GP algorithm is a variant of the ACA algorithm that is designed to enhance the purely algebraic ACA algorithm by incorporating a geometrical pivot selection strategy. The ACA-GP algorithm is only marginally more expensive but constantly ensures a better accuracy than the classical ACA. 

**License:** BSD 3-Clause

**Language:** Python

## Usage

Clone the repository 
```bash
git clone git@github.com:vyastreb/ACA.git
```

Install the requirements
```bash
pip install -r requirements.txt
```

Usage of the ACA algorithm (see `test.py`)
```python
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
```

## Full test

Script `test_for_point_clouds.py` contains an elaborated test of the ACA and ACA-GP algorithms for two rectangular clouds of a given aspect ratio and rotated one with respect to another by a random angle. The user can define the target distance between these clouds and define the type of points' distribution (uniform or normal). The script computes ACA and ACA-GP low-rank approximations and computes an SVD for comparison.

## References

Vladislav A. Yastrebov. ACA-GP: Adaptive Cross Approximation with a Geometrical Pivot Choice. arXiv preprint [arXiv:??.??](https://arxiv.org/abs/??.??), 2024.



