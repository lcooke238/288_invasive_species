# data generation. Based on work from https://github.com/SteevenJanny/DeepKKL/blob/master/data_generation/datagen_LotkaVolterra.py

# imports
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
import random 
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

#Constants: Simulation Parameters
ALPHA = 2 / 3
BETA = 4 / 3
GAMMA = 1
DELTA = 1

# Spatial Lotka Volterra Model Parameters
D = np.array([0.2, 0.2])
R = np.array([0.5, 0.5])
K = np.array([10, 10])
A = np.array([1.5, 1.5])
G = np.zeros(2)

def spatial_dynamics(t, y, n, m):
    """Spatial Lotka Volterra Model as defined in BRANTINGHAM's OG paper
        D - diffusion constant
        R - fundamental growth rate
        K - carrying capacity
        A - competition coefficient (how big of an impact one group's activites have on the competiting group)
        G - decay constant
    """
    delta = 1
    # OK, so because scipy only works with 1d-arrays, we have to represent our 2d-space in a flattened, 1d representation
    # Instead of being an array with shape (n, m, 2), we can think of it as a flattened, 1d-array
    # To access what would be [i, j, k], we would use 2 * (n * i + j) + k.
    out = np.zeros((2 * n * m))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            k = 2 * (n * i + j)
            # Contribution from diffusion
            # diffusion is estimated through finite differencing method
            # according to https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
            delta_t = (delta ** 2)/(4 * D)
            diffusion = (D * delta_t) / (delta ** 2) * (y[k + 2:k + 4] + y[k - 2:k] + y[k + 2 * n:k + 2 * n + 2] + y[k - 2 * n:k - 2 * n + 2] - 4 * y[k:k + 2])
            # Contribution from decay
            decay = G * y[k:k + 2]

            out[k: k + 2] = diffusion - decay

            # Computing growth of both groups with respect to competition
            growth_prey = R[0] * y[k] * (1 - y[k] / K[0] - A[0] * y[k + 1] / K[1])
            growth_pred = R[1] * y[k + 1] * (1 - y[k + 1] / K[1] - A[1] * y[k] / K[1])
            
            out[k] += growth_prey
            out[k + 1] += growth_pred
    return out


'''generate(): creates np file simulating multiple trajectories of Lotka Volterra's system. The args are:
    num_traj: number of trajectories to simulate. default 200
    len_traj: length of a trajectory (in seconds). default 25
    pts_per_sec: number of points per second. default 40
    save_loc: where to save created files
'''
def generate(num_traj=200,len_traj=50, pts_per_sec=100, save_loc='../Data/val.npy', prey_range=(1, 5), predator_range=(1, 3)):
    n = 25
    m = 25
    dataset = np.zeros((num_traj, n * m * 2, len_traj * pts_per_sec))  # That will store each simulation
    t_span = [0, len_traj]
    t_eval = np.linspace(0, len_traj, len_traj * pts_per_sec)  # Time vector

    # Change this line to configure how much you downsample the data, and the final time range
    downsample_rate = int(len(t_eval) / (len_traj * pts_per_sec))
    idx = np.arange(0, len(t_eval), downsample_rate)

    # Generate random initial values for prey and predator populations within the specified ranges
    y0 = np.zeros((n * m * 2))

    # Generating inital points for both populations
    for _ in range(3):
        # Prey
        i = np.random.randint(1, n - 1)
        j = np.random.randint(1, m - 1)
        y0[2 * (n * i + j)] = 5
    for _ in range(3):
        # Pred
        i = np.random.randint(1, n - 1)
        j = np.random.randint(1, m - 1)
        y0[2 * (n * i + j) + 1] = 5

    # sol.y has shape (2 * n * m, 2500) with one row representing the prey(t) function and the other representing the predator(t)function 
    sol = solve_ivp(spatial_dynamics, y0=y0, t_span=t_span, t_eval=t_eval, args=(n, m))

    np.save(save_loc, sol.y)
    return sol.y


def plot_predator_locations(grid, timestep):
    #Get the min and max of all your data
    _min, _max = np.amin(grid), np.amax(grid)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    #Add the vmin and vmax arguments to set the color scale
    ax.imshow(grid[:, :, 0, timestep], cmap=plt.cm.YlGn, vmin = _min, vmax = _max)
    ax2 = fig.add_subplot(2, 1, 2)
    #Add the vmin and vmax arguments to set the color scale
    ax2.imshow(grid[:, :, 1, timestep], cmap=plt.cm.YlGn, vmin = _min, vmax = _max)
    plt.show()


# Function to plot predator locations at a specified timestep
def plot_predator_locations_at_timestep(file_loc, timestep):
    dataset = np.load(file_loc)
    data = dataset.reshape((25, 25, 2, 5000))
    plot_predator_locations(data, timestep * 100)


# Define the maximum timestep based on the dataset
max_timestep = 49


# Spatial Lotka Volterra Model Parameters traps
D = np.array([0.2, 0.001])
R = np.array([0.5, 0])
K = np.array([10, 10])
A = np.array([1.5, 0])
G = np.zeros(2)

def spatial_dynamics_traps(t, y, n, m):
    """Spatial Lotka Volterra Model as defined in BRANTINGHAM's OG paper
        D - diffusion constant
        R - fundamental growth rate
        K - carrying capacity
        A - competition coefficient (how big of an impact one group's activites have on the competiting group)
        G - decay constant
    """
    delta = 1
    # OK, so because scipy only works with 1d-arrays, we have to represent our 2d-space in a flattened, 1d representation
    # Instead of being an array with shape (n, m, 2), we can think of it as a flattened, 1d-array
    # To access what would be [i, j, k], we would use 2 * (n * i + j) + k.
    out = np.zeros((2 * n * m))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            k = 2 * (n * i + j)
            # Contribution from diffusion
            # diffusion is estimated through finite differencing method
            # according to https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
            delta_t = (delta ** 2)/(4 * D)
            diffusion = (D * delta_t) / (delta ** 2) * (y[k + 2:k + 4] + y[k - 2:k] + y[k + 2 * n:k + 2 * n + 2] + y[k - 2 * n:k - 2 * n + 2] - 4 * y[k:k + 2])
            # Contribution from decay
            decay = G * y[k:k + 2]

            out[k: k + 2] = diffusion - decay

            # Computing growth of both groups with respect to competition
            growth_prey = R[0] * y[k] * (1 - y[k] / K[0] - A[0] * y[k + 1] / K[1])
            growth_pred = R[1] * y[k + 1] * (1 - y[k + 1] / K[1] - A[1] * y[k] / K[1])
            
            out[k] += growth_prey
            out[k + 1] += growth_pred
    return out


'''generate(): creates np file simulating multiple trajectories of Lotka Volterra's system. The args are:
    num_traj: number of trajectories to simulate. default 200
    len_traj: length of a trajectory (in seconds). default 50
    pts_per_sec: number of points per second. default 40
    save_loc: where to save created files
'''
def generate_traps(init_data_loc='../Data/val.npy', num_traps=20, num_traj=200, num_replacements=10, len_traj=50, pts_per_sec=100, save_loc='../Data/val_traps.npy', prey_range=(1, 5), predator_range=(1, 3)):
    # initialization stuff
    n = 25
    m = 25
    replacement_window = int(len_traj/num_replacements)
    dataset = np.zeros((num_traj, n * m * 2, len_traj * pts_per_sec))  # That will store each simulation
    t_span = [0, replacement_window]
    t_eval = np.linspace(0, replacement_window, replacement_window * pts_per_sec)  # Time vector

    # Change this line to configure how much you downsample the data, and the final time range
    downsample_rate = int(len(t_eval) / (replacement_window * pts_per_sec))
    idx = np.arange(0, len(t_eval), downsample_rate)

    # Generate random initial values for prey and predator populations within the specified ranges
    y0 = np.zeros((n, m, 2))

    # Generating inital points for both populations
    # import prey information
    dataset_init = np.load(init_data_loc)
    data_init = dataset_init.reshape((25, 25, 2, 5000))
    prey_data, predator_data = data_init[:, :, 0, :], data_init[:, :, 1, :]
    # set prey (invasive species) location at last timestep of initialization for new object where the traps will be the predators
    y0[:,:,0] = predator_data[:, :, -1]
    prey_data = prey_data[:, :, -1]

    # initialize trap locations based on number of desired traps and density, re initialize per replacement time
    # flatten pred array
    pred_flatten = y0[:,:,0].copy().flatten()
    num_traps_use = -1 * num_traps
    top_flat_indices = np.argpartition(pred_flatten, num_traps_use)[num_traps_use:]
    # top_flat_indices // m, top_flat_indices % m
    for idx in top_flat_indices:
        i = idx // m
        j = idx % m
        y0[i, j, 1] = 10
    y0 = y0.flatten()

    # modify a run s.t. each timestep consists of two solvers: one for traps and one for prey
    # sol.y has shape (2 * n * m, 2500) with one row representing the prey(t) function and the other representing the predator(t)function
    # sol_use = None
    master_sol = np.ndarray((n*m*2,pts_per_sec))

    for _ in range(num_replacements):
        # replace trap positions, creating new y0
        y0 = np.reshape(y0, (n, m, 2))
        pred_flatten = y0[:,:,0].copy().flatten()
        num_traps_use = -1 * num_traps
        top_flat_indices = np.argpartition(pred_flatten, num_traps_use)[num_traps_use:]
        for idx in top_flat_indices:
            i = idx // m
            j = idx % m
            y0[i, j, 1] = 10
        y0 = y0.flatten()
        for _ in range(replacement_window):
            # trap solver, only grab single timestep
            sol = solve_ivp(spatial_dynamics_traps, y0=y0, t_span=[0,1], t_eval=np.linspace(0, 1, pts_per_sec), args=(n, m))
            # prey solver y0 creation
            y_prey = np.zeros((n, m, 2))
            last_dim = int(pts_per_sec)
            sol_use = sol.y.reshape((n, m, 2, last_dim))
            pred_data_new, trap_data_new = sol_use[:, :, 0, :], sol_use[:, :, 1, :]
            # set prey from timestep of interest as predator in y_prey
            y_prey[:,:,1] = pred_data_new[:, :, -1]
            # grab predator information from prey data
            y_prey[:,:,0] = prey_data
            y_prey = y_prey.flatten()
            # prey solver, only grab single timestep
            sol_prey = solve_ivp(spatial_dynamics, y0=y_prey, t_span=[0,1], t_eval=np.linspace(0, 1, pts_per_sec), args=(n, m))
            # create y0 for next run of trap solver, overwrite y0 and prey_data
            y0 = np.zeros((n, m, 2))
            sol_prey_use = sol_prey.y.reshape((n, m, 2, last_dim))
            prey_data, predator_data = sol_prey_use[:, :, 0, :], sol_prey_use[:, :, 1, :]
            y0[:,:,0] = predator_data[:, :, -1]
            prey_data = prey_data[:, :, -1]
            # initialize trap locations based on number of desired traps and density, re initialize per replacement time
            y0[:,:,1] = trap_data_new[:,:,-1]
            y0 = y0.flatten()
            master_sol = np.concatenate((master_sol, sol_prey.y), 1)

    # save master sol object
    print(master_sol[:,100:].shape)
    np.save(save_loc, master_sol[:,100:])



