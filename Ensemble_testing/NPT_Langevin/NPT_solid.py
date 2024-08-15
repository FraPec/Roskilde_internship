from SAAP_potential import saap
import rumdpy as rp
from numba import cuda

cuda.select_device(1)

# Temperature we want to reach
T = 2.000

# Setup configuration: FCC Lattice
rho = 1.0/0.87489 # solid density
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(T=T)

# Simulation parameters
dt = 0.004
num_timeblocks = 1024
steps_per_timeblock = 4096
steps = num_timeblocks * steps_per_timeblock
scalar_output = 32


# Setup pair potential
pair_func = rp.apply_shifted_potential_cutoff(saap)  

# Setting parameters of the potential and cutoff
sigma, eps = 1.0, 1.0
pot_params = [65214.64725, -9.452343340, -19.42488828, 
          -1.958381959, -2.379111084, 1.051490962, 
          sigma, eps]
cut = [4.0 * sigma]
params = pot_params + cut

pair_pot = rp.PairPotential(pair_func, params, max_num_nbs=1000)

# Setup integrator: NVT
integrator = rp.integrators.NVT(dt=dt, tau=0.2, temperature=T)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks // 4,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='memory', 
                    scalar_output=scalar_output)

# Run NVT simulation to prepare for NPT
sim.run()

# Pressure we want to reach
P = 22.591

# Setup integrator: NPT Langevin
# Setting parameters for the NPT Langevin 
alpha_thermo = 0.5
alpha_baro = 0.005
m_baro = 0.001
integrator = rp.integrators.NPT_Langevin(temperature=T, pressure=P, 
                                          alpha=alpha_thermo, alpha_baro=alpha_baro,
                                          mass_baro=m_baro, dt=dt, 
                                          volume_velocity=0.0, barostatModeISO=True,
                                          boxFlucCoord=2, seed=2023)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='NPT2_solid.h5', 
                    scalar_output=scalar_output)

# Run simulation
sim.run()

cuda.close()