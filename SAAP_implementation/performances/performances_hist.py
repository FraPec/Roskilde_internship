import numpy as np
import matplotlib.pyplot as plt

t_float32 = np.loadtxt('times_float32_NVE.txt')
t_32_mean = np.mean(t_float32)
std_t_32 = np.std(t_float32)
t_float64 = np.loadtxt('times_float64_NVE.txt')
t_64_mean = np.mean(t_float64)
std_t_64 = np.std(t_float64)

# Plot of the histogram of performaces for float32
plt.figure(1)
plt.title('200 float32-NVE simulations: N particles = 2048, integration steps = 8192', fontsize=25)
plt.xlabel('time [s]', fontsize=30)
plt.ylabel('occurrencies', fontsize=30)
plt.hist(t_float32, alpha=0.6, color='blue', bins=5, label='float32')
plt.axvline(t_32_mean, color='blue', label=f'mean(t) = {t_32_mean:.3f}, std(t) = {std_t_32:.3f}')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.grid()
plt.show()


# Plot of the histogram of performaces for float64
plt.figure(1)
plt.title('200 float64-NVE simulations: N particles = 2048, integration steps = 8192', fontsize=25)
plt.hist(t_float64, alpha=0.6, color='red', bins=5, label='float64')
plt.axvline(t_64_mean, color='red', label=f'mean(t) = {t_64_mean:.3f}, std(t) = {std_t_64:.3f}')
plt.xlabel('time [s]', fontsize=30)
plt.ylabel('occurrencies', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
plt.grid()
plt.show()

