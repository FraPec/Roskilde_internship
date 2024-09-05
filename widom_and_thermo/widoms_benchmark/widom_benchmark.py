#!/usr/bin/env python3
#PBS -l nodes=bead52
#PBS -e error_widom.e
#PBS -o output_widom.o
import os, sys
if "PBS_O_WORKDIR" in os.environ:
    os.chdir(os.environ["PBS_O_WORKDIR"])
    sys.path.append(".")
sys.path.insert(0, "/net/debye/francescop/rumdpy-dev/")

import numpy as np
import rumdpy as rp
from SAAP_potential import saap
from numba import cuda

if __name__=='__main__':
    cuda.select_device(1)
    # Setting the NVT state point
    rho = 0.9
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
    num_timeblocks = 256
    steps_per_timeblock = 512
    scalar_output = 4
    # Saving path for simulations
    saving_path = 'memory'

    # Array of number of ghost particles
    num_ghost_particles = np.logspace(1, 7, num=100, dtype=np.int32)
    
    # Array for mean mu, std mu (between the number of blocks) and timing
    mu_v = np.zeros(num_ghost_particles.shape)
    sigma_v = np.zeros(num_ghost_particles.shape)
    time_v = np.zeros(num_ghost_particles.shape)

    # Setup configuration: FCC Lattice
    configuration = rp.Configuration(D=3)
    configuration.make_lattice(rp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
    configuration['m'] = m
    configuration.randomize_velocities(T=T)
    N = configuration.N 
    
    # Setup pair potential
    pair_func = rp.apply_shifted_potential_cutoff(saap) 
    pair_pot = rp.PairPotential(pair_func, pot_params, max_num_nbs=1000)

    # Setup integrator: NVT
    integrator = rp.integrators.NVT(dt=dt, tau=tau_T, temperature=T)
    
    # Setup Simulation.
    sim = rp.Simulation(configuration, pair_pot, integrator,
                        steps_between_momentum_reset=100,
                        num_timeblocks=num_timeblocks,
                        steps_per_timeblock=steps_per_timeblock,
                        storage=saving_path, 
                        scalar_output=scalar_output)
    # Preliminary run to initializate the timing_numba attribute in sim
    sim.run()
    
    for i in range(len(num_ghost_particles)):
        ghost_positions = np.random.rand(num_ghost_particles[i], configuration.D) * configuration.simbox.lengths
        calc_widom = rp.CalculatorWidomInsertion(configuration, pair_pot, T, ghost_positions)
        mu = np.zeros(num_timeblocks)
        sigma_mu = np.zeros(num_timeblocks)
        print('Production run:')
        # Let's simulate again, but using also the Widom's insertion
        for block in sim.timeblocks():
            calc_widom.update()
            print(f'.', end='', flush=True)
            calc_widom_data = calc_widom.read()
            # Widom insertion to compute chemical potential
            mu[block] = calc_widom_data['chemical_potential']
        mu_v[i] = np.mean(mu)
        # If we assume that blocks are independent, then we can compute std dev and divide it by sqrt of len(mu)
        sigma_v[i] = np.std(mu) / np.sqrt(len(mu))
        print(f"mu_excess = {mu_v[i]:.5f} +- {sigma_v[i]:.5f}.")
        # We extract the simulation time and the total time. In the total there is also the time for Widom's insertion
        time_total = sim.timing_numba / 1000 # time is in milliseconds
        time_sim = np.sum(sim.timing_numba_blocks) / 1000
        time_v[i] = time_total - time_sim
        print(f"Widom's insertion timing for {num_ghost_particles[i]}: {time_v[i]:.5f}")
    fname = 'widom_benchmark_data_liquid.txt'
    header = f'Density = {rho}, T = {T}.\nNum ghost particles, chemical potential, std. dev. chemical potential, times'
    tosave = np.transpose(np.array([num_ghost_particles, mu_v, sigma_v, time_v]))
    np.savetxt(fname, tosave, header=header)
    cuda.close()
    
