import numpy as np
import rumdpy as rp
from numba import cuda
from SAAP_potential import saap

def run_NVT(T_start, T_target, tau_T, rho, pot_params, m, sim_params, saving_path):     
    # Setup configuration: FCC Lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
    configuration['m'] = m
    configuration.randomize_velocities(T=T_start)
    N = configuration.N    
    
    # Extract params for simulation
    dt = sim_params[0]
    num_timeblocks = sim_params[1]
    first_block = sim_params[2]
    steps_per_timeblock = sim_params[3]
    scalar_output = sim_params[4]
    
    # Setup pair potential
    pair_func = rp.apply_shifted_potential_cutoff(saap) 
    pair_pot = rp.PairPotential(pair_func, pot_params, max_num_nbs=1000)

    # Setup integrator: NVT
    integrator = rp.integrators.NVT(dt=dt, tau=tau_T, temperature=T_target)
    
    # Setup Simulation.
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=num_timeblocks,
                        steps_per_timeblock=steps_per_timeblock,
                        storage=saving_path, 
                        scalar_output=scalar_output)
    
    # Run simulation
    sim.run()
    
    if saving_path=='memory':
        # Loading output from simulation object
        output = sim.output
        U, K, W, V = rp.extract_scalars(output, ['U', 'K', 'W', 'Vol'], first_block=first_block)
    else:
        # Loading output from h5 file
        output = rp.tools.load_output(saving_path)
        U, K, W, V = rp.extract_scalars(output, ['U', 'K', 'W', 'Vol'], first_block=first_block)
    
    D = sim.configuration.D
    dof = D * N - D
    T_kin = 2*K/dof
    rho = N/V
    P = rho*T_kin + W/V

    return np.mean(U), np.mean(T_kin), np.mean(P), N, sim

if __name__ == '__main__':
    # Setting parameters of the potential and cutoff
    sigma, eps = 1.0, 1.0
    pot_params = [65214.64725, -9.452343340, -19.42488828, 
              -1.958381959, -2.379111084, 1.051490962, 
              sigma, eps]
    cut = [4.0 * sigma]
    pot_params = pot_params + cut
    m = 1.0
    T = 2.0

    rho = 1.0 / 0.874908 # solid density
    t0 = rho**(1/3) / (T / m)**(1/2)
    print(t0)
    # Simulation parameters
    dt = 0.004*t0
    num_timeblocks = 12
    first_block = num_timeblocks // 3
    steps_per_timeblock = 4096
    scalar_output = 32
    sim_params = [dt, num_timeblocks, first_block, steps_per_timeblock, scalar_output]
    
    # Setting parameters for the NPT Langevin 
    tau_T = 0.2*t0
    
    #### SOLID ####
    T_target = 2.0
    T_start = 2.0
    P_target = 22.59078 
    P_start = P_target
    

    # Saving path for h5 file
    saving_path = 'memory'
    # Running
    U_s, T_s, P_s, N_s, sim = run_NVT(T_start, T_target, tau_T, rho, pot_params, m,
                                 sim_params, saving_path)
    print(f'Mean pot. energy  per particle = {U_s/N_s:.2f}, mean kin. energy per particle = {T_s/N_s:.5f}, mean pressure = {P_s:.2f}')
    print(f'Simulation obj: {sim.interactions}')
    
    
    