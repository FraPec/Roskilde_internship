import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import cuda
import rumdpy as rp
from float32.SAAP_potential import saap as saap32
from float64.SAAP_potential import saap as saap64

cuda.select_device(0)

def run(sim):
    # Run simulation
    sim.run()
    return time()


# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
configuration['m'] = 1.0
configuration.randomize_velocities(T=1.0)

# Setup pair potential
pair_func = rp.apply_shifted_force_cutoff(saap32)  

# Setting parameters of the potential and cutoff
sigma, eps = 1.0, 1.0
pot_params = [65214.64725, -9.452343340, -19.42488828, 
          -1.958381959, -2.379111084, 1.051490962, 
          sigma, eps]
cut = [4.0 * sigma]
params = pot_params + cut

pair_pot = rp.PairPotential(pair_func, params, max_num_nbs=1000)

# Setup integrator: NVE
dt = 0.005
integrator = rp.integrators.NVE(dt=dt)

# Setup Simulation.
num_timeblocks = 16
steps_per_timeblock = 512
steps = num_timeblocks * steps_per_timeblock
scalar_output = 16
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='memory', 
                    scalar_output=scalar_output)

num_sim = 200
t_float32 = np.zeros(num_sim)
t_float64 = np.zeros(num_sim)

for i in range(num_sim):
    t_float32[i] = run(sim)
    
# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
configuration['m'] = 1.0
configuration.randomize_velocities(T=1.0)

# Setup pair potential
pair_func = rp.apply_shifted_force_cutoff(saap64)  

# Setting parameters of the potential and cutoff
sigma, eps = 1.0, 1.0
pot_params = [65214.64725, -9.452343340, -19.42488828, 
          -1.958381959, -2.379111084, 1.051490962, 
          sigma, eps]
cut = [4.0 * sigma]
params = pot_params + cut

pair_pot = rp.PairPotential(pair_func, params, max_num_nbs=1000)

# Setup integrator: NVE
dt = 0.005
integrator = rp.integrators.NVE(dt=dt)

sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='memory', 
                    scalar_output=scalar_output)    
    
for i in range(num_sim):
    t_float64[i] = run(sim)

t_float32 = np.diff(t_float32)
t_32_mean = np.mean(t_float32)
std_t_32 = np.std(t_float32)
np.savetxt('times_float32_NVE.txt', t_float32, header=f'times [s], mean(t) = {t_32_mean:.1f}, std(t) = {std_t_32:.1f}')

t_float64 = np.diff(t_float64)
t_64_mean = np.mean(t_float64)
std_t_64 = np.std(t_float64)
np.savetxt('times_float64_NVE.txt', t_float64, header=f'times [s], mean(t) = {t_64_mean:.1f}, std(t) = {std_t_64:.1f}')

# Plot of the histogram of performaces for float32
plt.figure(1)
plt.title(f'{num_sim} NVE simulations: N particles = 2048, integration steps = {int(steps)}', fontsize=25)
plt.hist(t_float32, alpha=0.6, color='blue', bins=5, label='float32')
plt.axvline(t_32_mean, color='blue', label=f'mean(t) = {t_32_mean:.1f}, std(t) = {std_t_32:.1f}')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
# Plot of the histogram of performaces for float64
plt.hist(t_float64, alpha=0.6, color='red', bins=5, label='float64')
plt.axvline(t_64_mean, color='red', label=f'mean(t) = {t_64_mean:.1f}, std(t) = {std_t_64:.1f}')
plt.xlabel('time [s]', fontsize=30)
plt.ylabel('occurrencies', fontsize=30)
plt.show()


cuda.close()