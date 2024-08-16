import numpy as np
import rumdpy as rp
from NPTLangevin_function import run_NPT
import matplotlib.pyplot as plt
import pandas as pd

if __name__=='__main__':
    # Defining starting T, final T, number of steps and starting P
    T0 = 2.0
    Tf = 4.0
    steps = 11
    P0 = 22.591
    
    # Defining starting densities
    rho_s = 1.0/0.87489 # solid density 
    rho_l = 1.0/0.92612 # liquid density
    
    # Use numpy arrays
    T_v = np.linspace(T0, Tf, num=steps)
    dT = np.mean(np.diff(T_v))
    P_v = np.zeros(T_v.shape)
    rho_s_v = np.zeros(T_v.shape)
    rho_l_v = np.zeros(T_v.shape)
    P_v[0] = P0
    rho_s_v[0] = rho_s
    rho_l_v[0] = rho_l
    
    # Setting parameters of the potential and the cutoff
    sigma, eps = 1.0, 1.0
    pot_params = [65214.64725, -9.452343340, -19.42488828, 
              -1.958381959, -2.379111084, 1.051490962, 
              sigma, eps]
    cut = [4.0 * sigma]
    pot_params = pot_params + cut
    m = 1.0
    
    # Simulation parameters
    dt = 0.004
    num_timeblocks = 8
    first_block = num_timeblocks // 3
    steps_per_timeblock = 2024
    scalar_output = 8
    running_time = dt*num_timeblocks*steps_per_timeblock # Ramps parameter
    sim_params = [dt, num_timeblocks, first_block, steps_per_timeblock, scalar_output]
    # Setting parameters for the NPT Langevin 
    alpha_thermo = 0.5
    alpha_baro = 0.005
    m_baro = 0.001
    T_start_l = 5.0 # Sim of L starts at high T, so that we are sure it's a liquid phase

    # Array to collect slopes, differences in volume and entropy
    slope_v = np.zeros(T_v.shape[0]) # slope
    dvolume_v = np.zeros(T_v.shape[0]) # dVm
    dentropy_v = np.zeros(T_v.shape[0]) # dSm
    
    #### STARTING THE CLAUSIUS CLAPEYRON INTEGRATION ####
    for i in range(0, steps-1):
        print(f'STEP {i} : Starting first set of simulation for the prediction.')
        # Simulation for the prediction
        ### SOLID ###
        T_start_s = T_v[i]-1.0
        T_target_s = T_v[i]
        # Let's use a ramp from below for the solid
        Ttarget_function_s = rp.make_function_ramp(value0=T_start_s, x0=running_time*(1/12), 
                                                  value1=T_target_s, x1=running_time*(1/6))
        P = P_v[i]
        U_s, V_s, P_s, N_s = run_NPT(T_start_s, Ttarget_function_s, P, alpha_thermo, alpha_baro,
                                     m_baro, rho_s, pot_params, m, sim_params, 'memory')
        ### LIQUID ###
        T_target_l = T_v[i]
        # Let's use a ramp from above for the liquid
        Ttarget_function_l = rp.make_function_ramp(value0=T_start_l, x0=running_time*(1/12), 
                                                  value1=T_target_l, x1=running_time*(1/6))
        
        U_l, V_l, P_l, N_l = run_NPT(T_start_l, Ttarget_function_l, P, alpha_thermo, alpha_baro,
                                     m_baro, rho_l, pot_params, m, sim_params, 'memory')
        ### PREDICTOR ###
        dVm = V_l - V_s
        dSm = (U_l - U_s + dVm * P_v[i]) / T_v[i]
        k0 = dSm / dVm
        P_pred = P_v[i] + k0 * dT
        print(f'STEP {i} : P_predicted = {P_pred}')
        # Collect dVolume and dEntropy
        dvolume_v[i] = dVm
        dentropy_v[i] = dSm
        
        # Simulation for the correction
        ### SOLID ###
        T_start_s = T_v[i+1]-1.0
        T_target_s = T_v[i+1]
        # Let's use a ramp from below for the liquid
        Ttarget_function_s = rp.make_function_ramp(value0=T_start_s, x0=running_time*(1/12), 
                                                  value1=T_target_s, x1=running_time*(1/6))
        U_s, V_s, P_s, N_s = run_NPT(T_start_s, Ttarget_function_s, P_pred, alpha_thermo, alpha_baro,
                                     m_baro, rho_s, pot_params, m, sim_params, 'memory')
        ### LIQUID ###
        T_target_l = T_v[i+1]
        # Let's use a ramp from above for the liquid
        Ttarget_function_l = rp.make_function_ramp(value0=T_start_l, x0=running_time*(1/12), 
                                                  value1=T_target_l, x1=running_time*(1/6))
        U_l, V_l, P_l, N_l = run_NPT(T_start_l, Ttarget_function_l, P_pred, alpha_thermo, alpha_baro,
                                     m_baro, rho_l, pot_params, m, sim_params, 'memory')
        ### CORRECTOR ###
        dVm = V_l - V_s
        dSm = (U_l - U_s + dVm * P_pred) / T_v[i+1]
        k1 = dSm / dVm
        # Collect the average of dVolume and dEntropy
        dvolume_v[i] = (dVm + dvolume_v[i]) / 2
        dentropy_v[i] = (dSm + dentropy_v[i]) / 2
        ### FINAL RESULT ###
        P_v[i+1] = P_v[i] + dT / 2 * (k0 + k1)       
        slope_v[i+1] = (k0 + k1) / 2
        
        # New densities for step i+1
        rho_s = V_s / N_s
        rho_l = V_l / N_l
        print(f'STEP {i} : New P = {P_v[i+1]}')
    
    # Saving T, P
    to_save = np.array([T_v, P_v, slope_v, dvolume_v, dentropy_v]).transpose()
    file_path = 'CC_data.txt'
    header = 'T, P, slope (no slope at the beginning --> 0), Delta Volume, Delta Entropy'
    np.savetxt(file_path, to_save, header=header)
    
    # Import data for comparison
    df = pd.read_csv('CC_Argon_comparison.csv')
    T_comparison = np.array(df['T'])
    P_comparison = np.array(df['p'])
    
    plt.figure(1)
    plt.title('Phase diagram for Argon', fontsize=30)
    plt.plot(T_comparison, P_comparison, label='Paper', color='red')
    plt.scatter(T_v, P_v, label='My results', color='blue', marker='o')
    plt.ylabel('Pressure [$\epsilon \sigma^{-3}$]', fontsize=25)
    plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
    plt.grid()
    plt.legend
    plt.show()   
    
    plt.figure(2)
    plt.title('$\Delta V_m \equiv V_L - V_S$', fontsize=30)
    plt.scatter(T_v, dvolume_v, label='Argon')
    plt.ylabel('$\Delta V_m$ [$\sigma^{3}$]', fontsize=25)
    plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
    plt.grid()
    plt.show()  
    
    plt.figure(3)
    plt.title('$\Delta S_m \equiv (U_L - U_S + p \Delta V_m) / T$', fontsize=30)
    plt.scatter(T_v, dentropy_v, label='Argon')
    plt.ylabel('$\Delta S_m$ [$k_B$]', fontsize=25)
    plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
    plt.grid()
    plt.show()
        
    
    
        

