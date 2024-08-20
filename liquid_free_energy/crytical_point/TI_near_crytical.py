#!/usr/bin/env python3
#PBS -l nodes=bead57
#PBS -e error_ti.e
#PBS -o output_ti.o
import os, sys
if "PBS_O_WORKDIR" in os.environ:
    os.chdir(os.environ["PBS_O_WORKDIR"])
    sys.path.append(".")
sys.path.insert(0, "/net/debye/francescop/rumdpy-dev/")

import numpy as np
import rumdpy as rp
from NVTNose_function import run_NVT
from SAAP_potential import saap

def widom_insertion(num_particles, sim, T, first_block):
    '''

    Parameters
    ----------
    num_particles : int
        Number of ghost particles for the Montecarlo evaluation of the excess 
        chemical potential. mu_tot = mu_id + mu_ex
    sim : rumdpy simulation class
        Simulation.
    pair_pot : rumdpy potential
        Pair potential.from SAAP_potential import saap
    T : float
        Temperature.
    first_block : int
        First block from which compute the mean chem. pot.

    Returns
    -------
    mu : float
        Mean chemical potential.
    sigma : float
        Std. dev chemical potential.

    '''
    conf = sim.configuration
    # interactions is a list of things, the first of which is the pair potential
    pair_pot = sim.interactions[0] 
    
    ghost_positions = np.random.rand(num_particles, conf.D) * conf.simbox.lengths
    calc_widom = rp.CalculatorWidomInsertion(conf, pair_pot, T, ghost_positions)
    print('Production run:')
    for block in sim.timeblocks():
        if block >= first_block:
            calc_widom.update()
            print(f'block{block+1}\n', end='', flush=True)
        
    calc_widom_data = calc_widom.read()
    print(f"\nExcess chemical potential: {calc_widom_data['chemical_potential']}")

    # Error estimation assuming that timeblocks are statistically independent
    mu = calc_widom_data['chemical_potential']
    sigma = np.std(calc_widom_data['chemical_potentials']) / np.sqrt(len(calc_widom_data['chemical_potentials']))
    print(f"95 % confidence interval: [{mu - 1.96*sigma}, {mu + 1.96*sigma}]")
    
    return mu, sigma



if __name__=='__main__':
    # T and V (rho) for NVT
    T_start = 2.00
    T = 2.00
    rho_start = 0.35 # low density
    rho_end = 0.5 # density at coexistence point
    num_rhos = 100
    rho_v = np.linspace(rho_start, rho_end, num=num_rhos)
    
    # Setting parameters of the potential and cutoff
    sigma, eps = 1.0, 1.0
    pot_params = [65214.64725, -9.452343340, -19.42488828, 
              -1.958381959, -2.379111084, 1.051490962, 
              sigma, eps]
    cut = [4.0 * sigma]
    params = pot_params + cut
    m = 1.0
    
    # Setup pair potential
    pair_func = rp.apply_shifted_potential_cutoff(saap) 
    pair_pot = rp.PairPotential(pair_func, pot_params, max_num_nbs=1000)
    
    # Let's work in reduced units
    t0 = rho_v**(1/3) / (T / m)**(1/2)    
    # Coupling of thermostat
    tau_T = 0.2 * t0
    # Time step
    dt = 0.002 * t0

    # Params for simulation
    num_timeblocks = 128
    first_block = num_timeblocks//4
    steps_per_timeblock = 512
    scalar_output = 4
    sim_params = [dt[0], num_timeblocks, first_block, steps_per_timeblock, scalar_output]
    # Running times for ramp of T (vector!!!)
    running_time = dt*num_timeblocks*steps_per_timeblock

    # Saving path for simulations
    saving_path = 'memory'
    
    # Pressure array 
    P_v = np.zeros(rho_v.shape)
    # Chemical potential array
    mu_v = np.zeros(rho_v.shape)
    sigma_v = np.zeros(rho_v.shape)
    # Volumes array
    V_v = np.zeros(rho_v.shape)

    num_ghost_particles = 500_000

    for rho, i in zip(rho_v, range(len(rho_v))):
        # Change time step for reduced units simulation
        sim_params[0] = dt[i]
        # Temperature ramp
        Ttarget_function = rp.make_function_ramp(value0=T_start, x0=running_time[i]*(1/13), 
                                         value1=T, x1=running_time[i]*(1/12))
        print(f'Simulation at density: {rho_v[i]:.3f}') 
        # Simulation run
        U, T, P_v[i], N, sim = run_NVT(T_start=T_start, T_target=Ttarget_function, tau_T=tau_T[i], 
                                       rho=rho_v[i], pot_params=params, m=m, sim_params=sim_params, saving_path=saving_path)
        # Widom insertion to compute chemical potential
        mu_v[i], sigma_v[i] = widom_insertion(num_ghost_particles, sim, T, first_block)
        # Save volume
        V_v[i] = N / rho_v[i]  
    
    fname = 'TI_crytical.txt'
    header = 'density, pressure, chemical potential, std. dev. chemical potential, volume'
    tosave = np.transpose(np.array([rho_v, P_v, mu_v, sigma_v, V_v]))
    np.savetxt(fname, tosave, header=header)
        
        
    

    