import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import numba as nb
import os
from scipy.sparse import csgraph
from scipy.integrate import solve_ivp

# Solve ODE system for disk model with effective hydrodynamic interactions

np.random.seed()

# Path to save all data
sv_file = 'Data/'

# Simulation ID
simID = "test"
print("sim ID: "+simID)

# Simulation time for ODE model
simtimes = np.linspace(0, 2000, 1000)

# Number of disks
N = 7

# Size of periodic box (in units of disk size)
L = 100

# Periodic domain?
per_dom = False

# Compressing surface potential?
Surf_Pot = False
WellLength = 30  # Length scale of the well potential

# Stokeslet strength (must be >0, sets the attraction strength between disks)
Fg = 219 + 70 * np.random.randn(N)  # Units: [radius^2]/second

# Maximum interaction distance for attractive Stokeslet interaction
RFg_int = 2.8  # 2*sqrt(2) is the second next nearest neighbour in hexagonal grid

# Strength of rotational near-field interactions of neighbouring particles
# Free spinning calibration
f0 = -0.06
tau0 = 0.12

# Minimal distance of disk boundaries from which near-field interactions start
Rnf_int = 0.5

# Single disk angular frequency (= time-scale)
omega0 = 0.05 * 2 * np.pi * (0.72 + 0.17 * np.random.randn(N))

# Flow interactions between disks
# F: each disk will only interact with its image
# T: each disk interacts with all other disks and with its image
Flowinteract = True

# Lateral steric repulsion
Sterinteract = True

# Spin-Frequency near-field interactions to slow down nearby disks?
NFinteract = True

# Symmetrize the flow interactions?
# Unsymmetrization cause noise which can drive the translational motion
Symmetrize = False

# Far-field attraction from disks with up to two neighbours
# (otherwirse only near-field interactions)
SelectFarField = True

# Modulation of intrinsic torques through presence of nearby disks
ModOmega0 = True
N0_damping = 80

###### GRAVITY (=Stokeslet direction) (DON'T CHANGE IN DISK MODEL) ######
grav_vec = np.array([0, 0, -1],dtype=float)
grav_vec /= np.linalg.norm(grav_vec)

# Distance of flow singularity below the surface
h = 1

# Strength of steric repulsion
Frep_str = 3200 * (1 + 0.4 * (np.random.rand(N) - 0.5)) # For 1/r^12 repulsive potential (CORRECT ONE)

###################### SET INITIAL POSITIONS ############################

# % %%%%%%% RANDOM INITIAL CONDITIONS (PAPER) %%%%%%%
# % %%%%% (Starts with disks far outside) %%%%%%%
phi_part = 2 * np.pi * np.random.rand(N)  # Random angles
R_part = 0.8 * (350 + 200 * np.sqrt(np.random.rand(N)) ** (1 / 6))  # Random radii
Pos_init = np.column_stack((R_part * np.cos(phi_part), R_part * np.sin(phi_part)))

# % % Apply random stretch factors
for i in range(1):
    rand_stretch = 1 + 0.4 * np.random.randn(N)
    Pos_init = np.column_stack((rand_stretch, rand_stretch)) * Pos_init

# % % Move to center of domain
Pos_init -= np.mean(Pos_init, axis=0)
Pos_init %= L
# % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# % % Plot initial positions
fig, ax = plt.subplots()
ax.clear()
ax.scatter(Pos_init[:,0], Pos_init[:,1], c='r', s=3, edgecolors='black')
plt.show()
# % % Flatten for ode solver
Pos_init = Pos_init.reshape(-1,1)

################### END SET INITIAL POSITIONS ###########################

###################### Singularity implementation  ######################
# Functions require input of position vectors [rx1, ry1, rz1; rx2, ry2, rz2; ...]
# and similarly for the parameterizing Stokeslet unit vector f
@nb.njit
def norm2d(arr, ax):
    return np.sqrt((arr ** 2).sum(axis=ax))

################%% STOKESLET flow %%%%%%%%%%%%%%%%%%
@nb.njit
def u_st(f, r):
    r_norm = norm2d(r, 1).reshape(-1, 1)
    return (f / r_norm + r * np.sum(f * r, axis=1).reshape(-1, 1) / (r_norm ** 3).reshape(-1, 1)) / (8 * np.pi)

################% Steric repulsion of nearby disks %%%%%%%%%%%%%%%%%%%%%
# 1/r^12 potential repulsion between midpoints -> 1/r^13 force
# (written \vec{r}/r^14 because of vec(r) in nominator!!)
@nb.njit
def Frep(r):
    return 12 * r / norm2d(r, 1).reshape(-1, 1) ** 14

############### Radial dependence of near-field forces %%%%%%%%%%%%%%%%%##
@nb.njit
def feta(r):
    return np.log(Rnf_int / np.abs(r - 2))

############## Radial dependence of near-field torques %%%%%%%%%%%%%%%%%##
@nb.njit
def taueta(r):
    return np.log(Rnf_int / np.abs(r - 2))

# Transformations of image vector and pseudo-vector orientations
@nb.njit
def vec_img(e):
    return np.column_stack((e[:, 0], e[:, 1], -e[:, 2]))  # Stokeslet, force- and source-dipole

def disk_dynamics(t, y):
    # Print time for progress
    # if t % 0.001 < 0.00005:
    #    print(t)

    if per_dom:
        # Periodic boundary conditions
        y %= L

    # ODE function of disk dynamics for axisymmetric disks:
    # Input vector y contains for each disk 2D position and 3D orientation
    # [x1; y1; x2; y2; ... xN; yN]
    # Extract and format positions as needed for flow functions
    Pos3D = np.column_stack((y[:2 * N].reshape((N, 2)), -h * np.ones(N)))
    # Signed distance matrices r_i - r^0_i where flows from
    # singularities placed at r^0_i are evaluated at r_i
    dist_x = Pos3D[:, 0].reshape(1, -1) - Pos3D[:, 0].reshape(-1, 1)
    dist_y = Pos3D[:, 1].reshape(1, -1) - Pos3D[:, 1].reshape(-1, 1)
    # Stokeslet force orientation (DON'T CHANGE FOR DISK MODEL)
    grav = np.tile(grav_vec, (N, 1))
    grav /= np.linalg.norm(grav, axis=1).reshape(-1, 1)
    # Fixed global orientation of gravity
    fst_img = vec_img(grav)
    # Determine angular frequency of each disk
    rij = np.hypot(dist_x,dist_y)  # Distance matrix
    NH_matrix = (rij < (2 + Rnf_int)).astype(int) - np.eye(N, dtype=int)  # Adjacency matrix of particles within near-field interaction distance
    r,p = csgraph.connected_components(NH_matrix, directed=False)
    omega_all = omega0.copy()

    if SelectFarField:
        # Assume all particles participate in far-field interactions
        idx_FF = np.ones(N, dtype=bool)

    # ANGULAR FREQUENCY CALCULATION
    if NFinteract:
        for j in range(r): # Loop through connected components        
            idx_num = []

            if np.count_nonzero(p == j) > 1: # If at least two elements in connected component
                # Extract numeric indices of all disks in this component
                idx_num = np.nonzero(p == j)[0]
            
            if len(idx_num):
                # Number of disks in current connected component
                nrd_cc = len(idx_num)
                # Linear matrix of the torque balance for given connected component
                M = np.zeros((nrd_cc, nrd_cc))
                
                # Loop through those disks to build linear system
                for l in range(nrd_cc):
                    # Current disk
                    curr_disk = idx_num[l]
                    # Near-field interaction neighbours for current disk
                    nh_vec = np.nonzero(NH_matrix[curr_disk, :] > 0)[0]
                    # Torque interactions strengths for pair-wise distances of disks within interaction distance
                    tau = tau0 * taueta(rij[curr_disk, nh_vec])
                    # Fill linear system matrix row for given particle
                    M[l, l] = 1 + np.sum(tau)

                    for n in range(len(nh_vec)):
                        M[l, np.nonzero(idx_num == nh_vec[n])[0]] = tau[n]

                omega0_M = omega0[idx_num]

                if ModOmega0:
                    # Renormalize intrinsic rotation frequencies
                    omega0_M /= (1 + (nrd_cc / N0_damping)**2)

                # Add angular frequencies in this connected component into array according to disk IDs
                omega_all[idx_num] = np.linalg.solve(M, omega0_M)

                if SelectFarField:
                    # If these disks belong to a group with more than 3 disks
                    # remove those indices from the far-field interaction list
                    if len(idx_num) > 3:
                        idx_FF[idx_num] = False

    # Sum up all flow contributions that affect a given disk
    # Cross-product vectors for near-field force interactions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RotForce_dir_x = dist_y / rij
        RotForce_dir_y = -dist_x / rij

    u_e = np.zeros((N, 2))  # Initiate translational velocities array

    for j in range(N):  # Compute velocity for each disk
        if Flowinteract:  # Full flow interactions
            # Logical index for disks whose lateral interaction is taken into account
            idx_lat = np.ones(N, dtype=bool)
            idx_lat[j] = False

            if not (SelectFarField and idx_FF[j]):
                # Find all disks further than a distance away
                # and exclude them from Stokeslet interactions
                idx_far = rij[:, j] > RFg_int
                idx_lat[idx_far] = False

            # Logical index for disks whose image interaction is taken into account
            idx_img = idx_lat  # Only images that participate in lateral interactions

        else:  # Each disk only interacts with its image
            idx_lat = np.zeros(N, dtype=bool)  # No lateral interactions
            idx_img = np.zeros(N, dtype=bool)
            idx_img[j] = True  # Disk interacts only with its image

        # Because all disks are at the same distance below
        # the surface only image flow interaction plays a role
        r_curr = np.column_stack((dist_x[idx_img, j], dist_y[idx_img, j], -2 * h * np.ones(np.sum(idx_img))))
        # Prepare according arrays of vectors parameterizing flow singularities
        fst_curr = fst_img[idx_img]

        # Collect all attractive Stokeslet flow interactions (no additional weighting)
        if Symmetrize:
            u_star = 0.5 * np.sum(np.tile((Fg[j] + Fg[idx_img]).reshape(-1,1),(1,3)) * u_st(fst_curr, r_curr),axis=0)  # SYMMETRIZED
        else:
            u_star = np.sum(np.tile((Fg[idx_img]).reshape(-1,1),(1,3)) * u_st(fst_curr, r_curr),axis=0)  # UNSYMMETRIZED
        u_e[j] = u_star[:2]  # Only vx and vy are relevant
        
        # Steric repulsion only laterally between disks
        if Sterinteract:
            # Same neighbourhood as Stokeslet interaction
            r_curr_ster = r_curr.copy()
            r_curr_ster[:, 2] = 0
            u_rep = 0.5 * np.sum((Frep_str[j] + Frep_str[idx_lat]).reshape(-1,1) * Frep(r_curr_ster), axis=0)
            u_e[j] += u_rep[:2]
        
        # Contributions from transverse force interactions
        idx_neighb = NH_matrix[j, :] > 0  # Neighbour-indices (no diagonals!)

        if np.sum(idx_neighb) != 0:
            feta_curr = feta(rij[j, idx_neighb])
            # Build omega array that can be used for rotation force summation
            omega_full = np.tile(omega_all[j], (np.sum(idx_neighb), 1)).T + omega_all[idx_neighb]
            # Sum up all transverse force contributions for given disks
            u_e[j, 0] += f0 * np.sum(feta_curr * omega_full * RotForce_dir_x[j, idx_neighb])
            u_e[j, 1] += f0 * np.sum(feta_curr * omega_full * RotForce_dir_y[j, idx_neighb])

    # Fill RHS output vector of ODE system
    dydt = u_e.flatten().T

    # If well curvature effect is included
    if Surf_Pot:
        # Cylindrical coordinates of the disk positions
        R_pos = np.linalg.norm(Pos3D[:, :2], axis=1)
        Phi_pos = np.pi + np.arctan2(-Pos3D[:, 1], -Pos3D[:, 0])
        # Emulate centering effect of well curvature
        u_pot = -WellLength ** (-2) * np.column_stack((R_pos * np.cos(Phi_pos), R_pos * np.sin(Phi_pos))).T
        dydt += u_pot.flatten()

    return dydt

# Solve ODEs
start_time = time.time()
ysol = solve_ivp(disk_dynamics, [simtimes[0], simtimes[-1]], Pos_init.flatten(), method="RK23", t_eval=simtimes, rtol=1e-3, atol=1e-3, vectorized=True)
end_time = time.time()
print("Total time: ", end_time - start_time)
print(ysol)

# Save data
directory = os.path.dirname(sv_file+simID+'/')

if not os.path.exists(directory):
    os.makedirs(directory)

if per_dom:
    np.save(sv_file + simID +'/' + simID + '_pos.npy', ysol.y % L)
else:
    np.save(sv_file + simID +'/' + simID + '_pos.npy', ysol.y)

np.save(sv_file + simID +'/' + simID + '_time.npy', ysol.t)
np.save(sv_file + simID +'/' + simID + '_omega0.npy', omega0)
np.save(sv_file + simID +'/' + simID + '_params.npy', [N, L, Rnf_int, tau0, ModOmega0, N0_damping, NFinteract])
