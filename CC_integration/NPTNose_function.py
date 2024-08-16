import numpy as np
import rumdpy as rp
from SAAP_potential import saap
from postprocessing_NPT import postprocessing_NPT

def run_NPT(T_start, T_target, P, tau_T, tau_P, rho_start, pot_params, m, sim_params, saving_path):
    # Setup configuration: FCC Lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho_start)
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
    
    # Setup integrator: preliminary NVT
    integrator = rp.integrators.NVT(dt=dt, tau=tau_T, temperature=T_start)

    # Setup Simulation.
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=num_timeblocks // 4,
                        steps_per_timeblock=steps_per_timeblock,
                        storage='memory', 
                        scalar_output=scalar_output)

    # Preliminary NVT run to prepare for NPT
    sim.run()

    # Setup integrator: NPT
    integrator = rp.integrators.NPT_Atomic(temperature=T_target, tau=tau_T, pressure=P, tau_p=tau_P, dt=dt)
    
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
    
    return np.mean(U), np.mean(V), np.mean(P), N

if __name__ == '__main__':
    # Setting parameters of the potential and cutoff
    sigma, eps = 1.0, 1.0
    pot_params = [65214.64725, -9.452343340, -19.42488828, 
              -1.958381959, -2.379111084, 1.051490962, 
              sigma, eps]
    cut = [4.0 * sigma]
    pot_params = pot_params + cut
    m = 1.0
    
    # Simulation parameters
    dt = 0.001
    num_timeblocks = 12
    first_block = num_timeblocks // 3
    steps_per_timeblock = 4096
    scalar_output = 32
    sim_params = [dt, num_timeblocks, first_block, steps_per_timeblock, scalar_output]
    
    #### SOLID ####
    T_target = 4.0
    T_start = 4.0
    P_target = 75.252930 
    P_start = P_target
    
    rho = 1.0 /0.73880 # solid density
    # Saving path for h5 file
    saving_path = 'solid_NPT.h5'
    # Setting tau for the barostat
    tau_T = 0.1
    tau_P = 2.0
    # Running
    U_s, V_s, P_s, N = run_NPT(T_start, T_target, P_target, tau_T, tau_P, rho, pot_params, m, sim_params, saving_path)
    print(f'Mean pot. energy = {U_s:.2f}, mean volume = {V_s:.2f}, mean pressure = {P_s:.2f}')
    postprocessing_NPT(sim_params, P_target, rho, saving_path)
    
    #### LIQUID ####
    # Setting parameters of the simulation
    T_start = 5.0
    T_target = 2.0
    running_time = dt*num_timeblocks*steps_per_timeblock
    Ttarget_function = rp.make_function_ramp(value0=T_start, x0=running_time*(1/12), 
                                              value1=T_target, x1=running_time*(1/3))
    P = 22.591
    rho = 1.0/0.92614549 # liquid density
    # Saving path for h5 file
    saving_path = 'liquid_NPT.h5'
    # Setting tau for the barostat and the thermostat
    tau_T = 0.2
    tau_P = 5.0
    # Running
    U_l, V_l, P_l, rho_l = run_NPT(T_start, Ttarget_function, P, tau_T, tau_P, rho, pot_params, m, sim_params, saving_path)
    print(f'Mean pot. energy = {U_l:.2f}, mean volume = {V_l:.2f}, mean pressure = {P_l:.2f}')
    postprocessing_NPT(sim_params, P, rho, saving_path)