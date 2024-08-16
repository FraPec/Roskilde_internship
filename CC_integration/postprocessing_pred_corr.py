import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'CC_data.txt'
T_v, P_v, slope_v, dV_v, dS_v = np.transpose(np.loadtxt(file_path))
N = 2048
dv_v = dV_v / N
ds_v = dS_v / N

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
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.show()

plt.figure(2)
plt.title('$\Delta v_m \equiv [V_L - V_S] / N$', fontsize=30)
plt.scatter(T_v[:-1], dv_v[:-1], label='Argon')
plt.ylabel('$\Delta v_m$ [$\sigma^{3}$]', fontsize=25)
plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
plt.grid()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.show()

plt.figure(3)
plt.title('$\Delta s_m \equiv [(U_L - U_S + p \Delta V_m) / T] / N$', fontsize=30)
plt.scatter(T_v[:-1], ds_v[:-1], label='Argon')
plt.ylabel('$\Delta s_m$ [$k_B$]', fontsize=25)
plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
plt.grid()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.show()
    