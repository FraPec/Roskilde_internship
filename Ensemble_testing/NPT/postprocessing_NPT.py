import numpy as np
import rumdpy as rp
import matplotlib.pyplot as plt
from numba import cuda

cuda.select_device(0)
# Simulation parameters
dt = 0.004
num_timeblocks = 3
first_block = 1
steps_per_timeblock = 4096
steps = num_timeblocks * steps_per_timeblock
scalar_output = 32

# Target pressure
P_target = 22.591

# Target rho
target_rho = 1.0/0.87489

# Loading output from h5 file
output = rp.tools.load_output('NPT2_solid.h5')

nblocks, nconfs, _, N, D = output['block'].shape

conf = rp.Configuration(D=D, N=N)

conf.simbox = rp.Simbox(D, output['attrs']['simbox_initial'])
calc_rdf = rp.CalculatorRadialDistribution(conf, num_bins=1000)
positions = output['block'][nblocks//2:,:,0,:,:]
positions = positions.reshape(nblocks//2*nconfs, N, D)
for pos in positions[nconfs-1::nconfs]:
    conf['r'] = pos
    conf.copy_to_device()
    calc_rdf.update()
calc_rdf.save_average()
rdf = calc_rdf.read()
plt.figure(0)
rdf['rdf'] = np.mean(rdf['rdf'], axis=0)
plt.plot(rdf['distances'], rdf['rdf'], '-')
plt.xlabel('r [$\sigma$]', fontsize=25)
plt.ylabel('g(r)', fontsize=25)
plt.grid()
plt.show()
# NVT energetic output
U, K, W, V = rp.extract_scalars(output, ['U', 'K', 'W', 'Vol'], first_block=first_block)
print(V)
print(N)
t = np.arange(0, len(U)) * dt

# Extracting useful variables for NVT and computing pressures
# rho and volume
rho = N / V

# Kinetic temperature
dof = D * N - D # degrees of freedom
T_kin = 2 * K / dof

# Temperature plot 
plt.figure(1)
plt.plot(t, T_kin)
plt.grid()
plt.axhline(np.mean(T_kin), label=f'mean temperature = {np.mean(T_kin):.2f} +- {np.std(T_kin):.2f}', color='black')
plt.axhline(np.mean(T_kin) + np.std(T_kin), color='r')
plt.axhline(np.mean(T_kin) - np.std(T_kin), color='r')
plt.xlabel('time', fontsize=25)
plt.ylabel('T [$\epsilon / k_B$]', fontsize=25)
plt.legend()
plt.show()

# Compute instantaneous pressure
P = rho * T_kin + W / V

# Pressure plot 
plt.figure(2)
plt.plot(t, P)
plt.grid()
plt.axhline(np.mean(P), label=f'mean pressure = {np.mean(P):.1f} +- {np.std(P):.1f}', color='black')
plt.axhline(P_target, label=f'Target pressure = {P_target}', color='green')
plt.axhline(np.mean(P) + np.std(P), color='r')
plt.axhline(np.mean(P) - np.std(P), color='r')
plt.xlabel('time', fontsize=25)
plt.ylabel('P [$\epsilon \sigma^{3}$]', fontsize=25)
plt.legend()
plt.show()

# Volume plot 
plt.figure(2)
plt.plot(t, rho)
plt.grid()
plt.axhline(np.mean(rho), label=f'mean rho = {np.mean(rho):.3f} +- {np.std(rho):.3f}', color='black')
plt.axhline(np.mean(rho) + np.std(rho), color='r')
plt.axhline(target_rho, color='green', label=f'target rho = {target_rho:.3f}')
plt.axhline(np.mean(rho) - np.std(rho), color='r')
plt.xlabel('time', fontsize=25)
plt.ylabel(' rho [$\sigma^{-3}$]', fontsize=25)
plt.legend()
plt.show()

print(f"Mean pressure: {np.mean(P):.1f} +- {np.std(P):.1f}")
print(f"Mean kinetic temperature: {np.mean(T_kin):.2f}+- {np.std(T_kin):.2f}")
print(f"Mean potential per particle: {(np.mean(U) / N):.1f} +- {(np.std(U) / N):.1f}")

cuda.close()
