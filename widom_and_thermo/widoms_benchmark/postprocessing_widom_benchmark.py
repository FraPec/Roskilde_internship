import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__=='__main__':
    fname = 'widom_benchmark_data_gas.txt'
    num_ghost_particles, mu_v, sigma_v, time_v = np.transpose(np.loadtxt(fname))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    for ax in [ax1, ax2]:
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    
    fig.suptitle(r'$time$ and $\sigma_{\mu_{ex}}$ VS number of $ghost$ particles, $\rho=0.1$, $T=2.0$', fontsize=25)
    ### Plot of times against number of ghost particles
    ax1.scatter(num_ghost_particles, time_v, marker='x')
    ax1.set_yscale('log', base=10)
    ax1.set_xscale('log', base=10)

    ax1.grid()
    ax1.set_ylabel(r'global time [s]', fontsize=25)

    ### Plot of sigma against number of ghost particles
    ax2.scatter(num_ghost_particles, sigma_v, marker='x')
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=10)
    ax2.grid()
    ax2.set_xlabel(r'num. ghost particles', fontsize=30)
    ax2.set_ylabel(r'$\sigma [\epsilon]$', fontsize=25)
 
    plt.show()
 