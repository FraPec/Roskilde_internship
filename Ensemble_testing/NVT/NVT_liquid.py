from SAAP_potential import saap
import rumdpy as rp

# Starting temperature
T_start = 5.000 # Temperature at which we have a liquid
# Temperature we want to reach
T = 2.000

# Setup configuration: FCC Lattice
rho = 1.0/0.92612 # liquid density
configuration = rp.Configuration(D=3)
configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(T=T_start)

# Simulation parameters
dt = 0.004
num_timeblocks = 1024
first_block = num_timeblocks // 3
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

running_time = dt*num_timeblocks*steps_per_timeblock

print('High Temperature followed by cooling and equilibration:')
Ttarget_function = rp.make_function_ramp(value0=T_start, x0=running_time*(1/12), 
                                         value1=T, x1=running_time*(1/3))


# Setup integrator: NVT
integrator = rp.integrators.NVT(dt=dt, tau=0.2, temperature=Ttarget_function)

# Setup Simulation.
sim = rp.Simulation(configuration, pair_pot, integrator,
                    steps_between_momentum_reset=100,
                    num_timeblocks=num_timeblocks,
                    steps_per_timeblock=steps_per_timeblock,
                    storage='NVT2_liquid.h5', 
                    scalar_output=scalar_output)

# Run simulation
sim.run()
