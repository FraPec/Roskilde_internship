#!/usr/bin/env python3
#PBS -l nodes=bead57
#PBS -e error_widom.e
#PBS -o output_widom.o
import os, sys
if "PBS_O_WORKDIR" in os.environ:
    os.chdir(os.environ["PBS_O_WORKDIR"])
    sys.path.append(".")
sys.path.insert(0, "/net/debye/francescop/rumdpy-dev/")

from time import time
import numpy as np
import rumdpy as rp
from numba import cuda
from thermodynamic_integration import widom_insertion
from NVTNose_function import run_NVT
from SAAP_potential import saap

if __name__=='__main__':
    cuda.select_device(1)
    # Setting the NVT state point
    rho = 0.1
    T = 2.0
    
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
    t0 = rho**(1/3) / (T / m)**(1/2)    
    # Coupling of thermostat
    tau_T = 0.2 * t0
    # Time step
    dt = 0.002 * t0
    
    # Params for simulation
    num_timeblocks = 64
    first_block = 1
    steps_per_timeblock = 512
    scalar_output = 4
    sim_params = [dt, num_timeblocks, first_block, steps_per_timeblock, scalar_output]
    # Saving path for simulations
    saving_path = 'memory'

    # Array of number of ghost particles
    num_ghost_particles = np.logspace(1, 7, num=600, dtype=np.int32)
    
    # Array for mean mu, std mu (between the number of blocks) and timing
    mu_v = np.zeros(num_ghost_particles.shape)
    sigma_v = np.zeros(num_ghost_particles.shape)
    time_v = np.zeros(num_ghost_particles.shape)


    # Simulation run
    U, T, P, N, sim = run_NVT(T_start=T, T_target=T, tau_T=tau_T, 
                              rho=rho, pot_params=params, m=m, sim_params=sim_params, saving_path=saving_path)
    for i in range(len(num_ghost_particles)):
        # Widom insertion to compute chemical potential
        time_v[i] = time()
        mu_v[i], sigma_v[i] = widom_insertion(num_ghost_particles[i], sim, T, first_block) 
        time_v[i] = time() - time_v[i]
    
    cuda.close()
    fname = 'N_mu_sigma_time.txt'
    header = 'Num ghost particles, chemical potential, std. dev. chemical potential, times'
    tosave = np.transpose(np.array([num_ghost_particles, mu_v, sigma_v, time_v]))
    np.savetxt(fname, tosave, header=header)
    
