import numpy as np 
import matplotlib.pyplot as plt
import rumdpy as rp
from SAAP_potential import saap

# Setup configuration: FCC Lattice
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
configuration['m'] = 1.0
configuration.randomize_velocities(T=1.0)

# Setup pair potential
pair_func = rp.apply_shifted_force_cutoff(saap)  

# Setting parameters of the potential and cutoff
sigma, eps = 1.0, 1.0
pot_params = [65214.64725, -9.452343340, -19.42488828, 
          -1.958381959, -2.379111084, 1.051490962, 
          sigma, eps]
cut = [4.0 * sigma]
params = pot_params + cut

pair_pot = rp.PairPotential(pair_func, params, max_num_nbs=1000)

# Setup integrator: NVE
dt = 0.002
integrator = rp.integrators.NVE(dt=dt)

# Setup Simulation.
num_timeblocks = 64
steps_per_timeblock = 1024
steps = num_timeblocks * steps_per_timeblock
scalar_output = 16
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='memory', 
                    scalar_output=scalar_output)

# Saving starting configuration to see if it's an FCC 
# rp.tools.save_configuration(configuration, "initial.xyz")

# Run simulation
sim.run()

# Saving final configuration to see if it's an FCC 
# rp.tools.save_configuration(configuration, "final.xyz")

# See if it's a correct NVE
output = sim.output
U, K = rp.extract_scalars(output, ['U', 'K'])
U = U[len(U)//4:]
K = K[len(K)//4:]
t = np.arange(0, dt*(len(U)), dt)
mean_U = np.mean(U)
mean_K = np.mean(K)
rescaled_U = U / mean_U
rescaled_K = K / mean_K
E = U + K
mean_E = np.mean(E)
rescaled_E = E / mean_E

plt.figure(1)
plt.plot(t, rescaled_U, label='$U/U_{mean}$')
plt.plot(t, rescaled_K, label='$K/K_{mean}$')
plt.plot(t, rescaled_E, label='$E/E_{mean}$')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=30)
plt.grid()
plt.xlabel('time [$\sigma \sqrt{m/ \epsilon}$]', fontsize=35)
plt.ylabel('Rescaled energies', fontsize=35)
plt.show()

