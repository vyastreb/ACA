import numpy as np
import ACAs as aca

# Uncomment the two following lines to use a fixed seed
seed = 124  
np.random.seed(seed)

# Create two separate clouds of points
algorithm = "ACA-GP" # "ACA" or "ACA-GP"
N = 200
M = 300
t_coord = np.random.rand(N, 2)
s_coord = np.random.rand(M, 2)
Delta_X = 1.5
Delta_Y = .5
t_coord += np.array([Delta_X,Delta_Y])


# Set low-rank approximation parameters
tolerance = 1e-3
min_pivot = 1e-10
max_rank = min(N, M)
green_kernel_power = 1
green_kernel_factor = 1

# Run the ACA algorithm
if algorithm == "ACA":
    U,V,error,rank,Jk,Ik,history = aca.aca(   t_coord, s_coord, green_kernel_factor, \
                                           tolerance, max_rank, min_pivot, green_kernel_power)
elif algorithm == "ACA-GP":
    U,V,error,rank,Jk,Ik,history = aca.aca_gp(t_coord, s_coord, green_kernel_factor, \
                                              tolerance, max_rank, min_pivot, green_kernel_power, \
                                              Rank3SpecialTreatment=True, convex_hull_distance=("const",1.))
else:
    raise ValueError("Invalid algorithm")


## BO plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Keep only positive arguments of Jk
Jk = Jk[:rank]
dtext = 0.01
fig,ax = plt.subplots()
ax.set_aspect('equal', 'box')
plt.scatter(t_coord[:,0],t_coord[:,1],s=2.5,c="r",label="t_coord",alpha=0.1)
plt.scatter(s_coord[:,0],s_coord[:,1],s=2.5,c="b",label="s_coord",alpha=0.1)
for j in range(len(Jk)):
    plt.plot(s_coord[Jk[j],0],s_coord[Jk[j],1],"o",markersize=2.5,c="k",zorder=10)
    plt.text(s_coord[Jk[j],0]+dtext,s_coord[Jk[j],1]+dtext,f"{j+1}",fontsize=12)
for i in range(len(Jk)):
    plt.plot(t_coord[Ik[i],0],t_coord[Ik[i],1],"o",markersize=2.5,c="k",zorder=10)
    plt.text(t_coord[Ik[i],0]+dtext,t_coord[Ik[i],1]+dtext,f"{i+1}",fontsize=12)
    
rect2 = patches.Rectangle((0,0), 1, 1,linewidth=1,edgecolor='#990000',linestyle="dashed",facecolor='none', zorder=3) 
ax.add_patch(rect2)

rect3 = patches.Rectangle((Delta_X,Delta_Y), 1, 1,linewidth=1,edgecolor='#990000',linestyle="dashed",facecolor='none', zorder=3)
ax.add_patch(rect3)

plt.legend()
plt.show()
fig.savefig(f"{algorithm}_classical.png", dpi=300)
## EO plot


# Compute the approximated matrix
approx_matrix = np.dot(U,V)

# Compute the error
full_matrix = np.zeros((N, M))
for i in range(N):
    full_matrix[i] = aca.line_kernel(t_coord[i], s_coord, green_kernel_factor, green_kernel_power)
norm_full_matrix = np.linalg.norm(full_matrix,"fro")
aca_gp_error = np.linalg.norm(approx_matrix - full_matrix,"fro")/norm_full_matrix

print("/ Algorithm: {0}".format(algorithm))
print(" Approximation rank: {0:2d} ".format(rank))
print(" Storage fraction:   {0:.2f} %".format(100*rank*(N+M)/(N*M)))
print(" Relative error:     {0:.2e} ".format(aca_gp_error))
print(" Approximate error:  {0:.2e} < {1:.2e}".format(error, tolerance))

