import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = 'data/CC_data.txt'
T_v, P_v, slope_v, dV_v, dS_v = np.transpose(np.loadtxt(file_path))
N = 2048
dv_v = dV_v / N
ds_v = dS_v / N

# Import data for comparison
df = pd.read_csv('data/CC_Argon_comparison.csv')
T_comparison = np.array(df['T'])
P_comparison = np.array(df['p'])
v_fcc_ref = np.array(df['v_fcc'])
v_liquid_ref = np.array(df['v_liquid'])
u_fcc_ref = np.array(df['u_fcc'])
u_liquid_ref = np.array(df['u_liquid'])

# Define reference entropy diff and volume diff for comparison
dv_v_ref = v_liquid_ref - v_fcc_ref
ds_v_ref = 1 / T_comparison * (P_comparison * dv_v_ref + u_liquid_ref - u_fcc_ref)

# Save my results and the one in reference in a csv, such that is easyly accessible
mask = T_comparison>=2.0
T_comparison_tosave = T_comparison[mask]
P_comparison_tosave = P_comparison[mask]
dv_v_ref_tosave = dv_v_ref[mask]
ds_v_ref_tosave = ds_v_ref[mask]
tosave = np.transpose([T_comparison_tosave, P_comparison_tosave, P_v, ds_v_ref_tosave, ds_v, dv_v_ref_tosave, dv_v])
columns = ['T', 'ref p', 'my p', 'ref ds', 'my ds', 'ref dv', 'my dv']
df_tosave = pd.DataFrame(tosave, columns=columns)
df_tosave.to_csv('data/my_data_vs_reference.csv', sep=',')

plt.figure(1)
plt.title('Phase diagram for Argon', fontsize=30)
plt.plot(T_comparison, P_comparison, label='Results of reference', color='red', zorder=0)
plt.scatter(T_v, P_v, label='My results', color='blue', marker='o', zorder=1)
plt.ylabel('Pressure [$\epsilon \sigma^{-3}$]', fontsize=25)
plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
plt.grid()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)

plt.figure(2)
plt.title('$\Delta v_m \equiv [V_L - V_S] / N$ for Argon', fontsize=25)
plt.scatter(T_v[:-1], dv_v[:-1], label='My results', zorder=1)
plt.plot(T_comparison, dv_v_ref, label='Results of reference', color='red', zorder=0)
plt.ylabel('$\Delta v_m$ [$\sigma^{3}$]', fontsize=25)
plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
plt.grid()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)

plt.figure(3)
plt.title('$\Delta s_m \equiv [(U_L - U_S + p \Delta V_m) / T] / N$ for Argon', fontsize=25)
plt.scatter(T_v[:-1], ds_v[:-1], label='My results', zorder=1)
plt.plot(T_comparison, ds_v_ref, label='Results of reference', color='red', zorder=0)
plt.ylabel('$\Delta s_m$ [$k_B$]', fontsize=25)
plt.xlabel('Temperature [$\epsilon / k_B$]', fontsize=25)
plt.grid()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
#plt.show()
    