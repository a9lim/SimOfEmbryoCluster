import numpy as np
import time
from matplotlib import animation
from matplotlib.colors import Normalize
from scipy import sparse
import matplotlib.pyplot as plt

# Path to save all data
data_dir = 'Data/'

# Simulation ID
sim_id = "bigSim"

fg0 = 6
f0 = 6
tau0 = 6

sim_id += f"_fg{fg0}_f0{f0}_tau0{tau0}"

pos = np.loadtxt(data_dir + sim_id + '/' + sim_id + '_pos.csv', delimiter=',')
t = np.loadtxt(data_dir + sim_id + '/' + sim_id + '_time.csv', delimiter=',')
omega0 = np.loadtxt(data_dir + sim_id + '/' + sim_id + '_omega0.csv', delimiter=',')
omega_alltime = np.loadtxt(data_dir + sim_id + '/' + sim_id + '_omega_alltime.csv', delimiter=',')
(N, L, rnf_int, tau0, mod_omega0, n0_damping, nf_interact) = np.loadtxt(data_dir + sim_id + '/' + sim_id + '_params.csv', delimiter=',')
N = int(N)
L = int(L)


def compute_omega(pos):
    # Signed distance matrices r_i - r^0_i where flows from singularities placed at r^0_i are evaluated at r_i
    dist_x = pos[:, 0].reshape(1, -1) - pos[:, 0].reshape(-1, 1)
    dist_y = pos[:, 1].reshape(1, -1) - pos[:, 1].reshape(-1, 1)
    # Determine angular frequency of each disk
    rij = np.hypot(dist_x, dist_y)  # Distance matrix
    nh_matrix = (rij < 3.5).astype(int) - np.eye(N, dtype=int)
    return sparse.lil_array(nh_matrix).rows


# approximately reasonable scaling
# around 100000/L^2
size = (512/L)**2

fig, ax = plt.subplots()
sc = ax.scatter([], [], c=[], cmap='jet', s=size, linewidths=0)
cbar = plt.colorbar(sc)
cbar.set_label('Neighbor Exchanges')
cbar.mappable.set_norm(Normalize(vmin=1, vmax=250))
ax.set_xlim(-L * 0.2, L * 1.2)
ax.set_ylim(-L * 0.2, L * 1.2)

horse = np.zeros(N)

r = sparse.lil_array((N, 1)).rows
oldr = r[0].copy()
st = 500
fatglorp = []

for j in range(len(t)):
    smallhorse = np.zeros(N)
    glorp = [[]]
    data = pos[:, j].reshape(-1, 2)
    oldr = r
    r = compute_omega(data)
    for i in range(N):
        if (r[i] != oldr[i]).count_nonzero() == 0:
            smallhorse[i] += 1
    I = list(range(N))
    for i in I:
        if len(r[i]) < 6:
            glorp.append(i)
    for i in glorp:
        I.remove(i)
    k = 1
    while len(I) > 0:
        glorp.append([])
        for i in I:
            if not set(glorp[k - 1]).intersection(set(r[i])):
                glorp[k].append(i)
        for i in glorp[k]:
            I.remove(i)
        k += 1
    snip = []
    for i in range(len(glorp)):
        for k in glorp[i]:
            snip[i] += omega_alltime[k, j]
        snip[i] /= len(glorp[i])
    fatglorp = np.column_stack((fatglorp, snip))
    horse = np.column_stack((horse, smallhorse))

def update(frame):
    data = pos[:, frame].reshape(-1, 2)
    sc.set_offsets(np.c_[data[:, 0], data[:, 1]])
    sc.set_array(horse[:, frame])
    return sc,


print('Rendering')
true_start_time = time.time()
ani = animation.FuncAnimation(fig, update, frames=range(st, len(t)), interval=20)
end_time = time.time()
print("Rendering time: ", end_time - true_start_time)

print('Writing')
start_time = time.time()
ani.save(data_dir + sim_id + '/' + sim_id + '_animation-graph.gif', fps=30, codec='hevc_nvenc')
end_time = time.time()
print("Write time: ", end_time - start_time)
print("Total time: ", end_time - true_start_time)

plt.show()

