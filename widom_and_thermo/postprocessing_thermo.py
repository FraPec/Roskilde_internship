import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    fname = 'data/thermodyn_int.txt'
    T = 2.0
    m = 1.0
    rho_v, P_v, mu_ex_v, sigma_v, V_v = np.transpose(np.loadtxt(fname))
    N = rho_v[0] * V_v[0] # N is a constant
    
    # chemical potential of ideal gas 
    de_broglie = 1 / np.sqrt(2 * np.pi * m * T)
    mu_ideal_v = - T * np.log(V_v / (de_broglie * (N+1) )) 
    mu_tot_v = (mu_ex_v + mu_ideal_v)
    # total chemical potential is proportional to G by N
    G_v = N * (mu_ex_v + mu_ideal_v) 
    # Helmholtz free energy
    F_v = G_v - P_v * V_v
    V_der_v_widom = np.gradient(G_v, P_v) # Derivative of G at constant T is V
    F_int_v = np.zeros(F_v.shape)
    for i in range(len(P_v)): # Thermodynamic integration to obtain F
        F_int_v[i] = np.trapz(-P_v[:i+1], V_v[:i+1])

    # Let's define a G obtained from the thermodynamic integration
    G_int_v = F_int_v + P_v * V_v
    constant = G_v[0] - G_int_v[0] # It's an integration, so we have an unknown constants that we estimate from Widom's insertion first point
    G_int_v = G_int_v + constant

    ### Plot of Volumes against pressures
    plt.figure()
    plt.title('$V/N \ vs\ P$, T = 2.0', fontsize=35)
    plt.plot(P_v, V_v/N, label=r'v, NVT simulation', zorder=0)
    plt.scatter(P_v, V_der_v_widom/N, label=r"$\frac{1}{N} \frac{\partial G}{\partial P} \|_{T} = v$, Widoms's insertion", color='orange', marker='x', zorder=2)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'p[$\epsilon / \sigma^{3}$]', fontsize=35)
    plt.ylabel(r'v $\equiv \ V/N$ [$\sigma^3$]', fontsize=35)
    plt.legend(fontsize=25)
    
    
    
    tosave = np.array([P_v, G_int_v / N])
    np.savetxt('G_liquid.txt', tosave)

    ### Plot of mu (G per particle) against pressures
    plt.figure()
    plt.title('Widom $vs$ thermod. int., T = 2.0', fontsize=35)
    plt.errorbar(P_v, mu_tot_v, yerr=sigma_v, label=r"$\mu_{tot}$ Widom's insertion", linestyle=' ', marker='.')
    plt.scatter(P_v, G_int_v/N, label=r'$\mu_{tot}$ thermodynamic integration for F', color='orange', marker='x')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'p[$\epsilon / \sigma^{3}$]', fontsize=35)
    plt.ylabel(r'$\mu_{tot} [\epsilon]$', fontsize=35)
    plt.legend(fontsize=25)

    ### Plot of excess potential
    plt.figure()
    plt.title('Chemical potentials, T = 2.0', fontsize=35)
    plt.scatter(P_v, mu_ex_v, label=r"$\mu_{excess}$")
    plt.scatter(P_v, mu_ideal_v, label=r"$\mu_{ideal}$")
    plt.scatter(P_v, mu_tot_v, label=r"$\mu_{tot}$")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'p[$\epsilon / \sigma^{3}$]', fontsize=35)
    plt.ylabel(r'$\mu [\epsilon]$', fontsize=35)
    plt.legend(fontsize=25)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for ax in [ax1, ax2]:
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    fig.suptitle(r'$\sigma_{\mu_{ex}}$ and $\mu_{ex}$ vs $\rho$, $T=2.0$', fontsize=30)
    ### Plot of sigma against rho
    ax2.scatter(rho_v, sigma_v, label=r"{\sigma}", marker='.')
    ax2.grid()
    ax2.set_ylabel(r'$\sigma [\epsilon]$', fontsize=25)
    ### Plot of sigma against rho
    ax1.errorbar(rho_v, mu_ex_v, label=r"{\mu_{ex}}", yerr=sigma_v, linestyle=' ', marker='.')
    ax1.grid()
    ax1.set_ylabel(r'$\mu_{ex} [\epsilon]$', fontsize=25)
    ax2.set_xlabel(r'$\rho [\sigma^{-3}]$', fontsize=25)

    plt.show()
