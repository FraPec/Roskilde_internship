import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__=='__main__':
    fname = 'TI_crytical.txt'
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
    V_der_v = np.gradient(G_v, P_v) # Derivative of G at constant T is V
    F_int_v = np.zeros(F_v.shape)
    for i in range(len(P_v)): # Thermodynamic integration to obtain F
        F_int_v[i] = np.trapz(-P_v[:i+1], V_v[:i+1])
    

    ### Plot of Volumes against pressures
    plt.figure()
    plt.title('$V/N \ vs\ P$, T = 2.0', fontsize=35)
    plt.plot(P_v, V_v/N, label=r'v, NVT simulation')
    plt.scatter(P_v, V_der_v/N, label=r"$\frac{1}{N} \frac{\partial G}{\partial P} \|_{T} = v$, Widoms's insertion", color='orange', marker='x')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'p[$\epsilon / \sigma^{3}$]', fontsize=35)
    plt.ylabel(r'v $\equiv \ V/N$ [$\sigma^3$]', fontsize=35)
    plt.legend(fontsize=25)
    
    # Let's define a G obtained from the thermodynamic integration
    G_int_v = F_int_v + P_v * V_v
    constant = np.mean(G_v - G_int_v) # It's an integration, so we have an unknown constants
    G_int_v = G_int_v + constant 
    
    ### Plot of mu (G per particle) against pressures
    plt.figure()
    plt.title('Widom $vs$ thermod. int., T = 2.0', fontsize=35)
    plt.scatter(P_v, mu_tot_v, label=r"$\mu_{tot}$ Widom's insertion")
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
    plt.plot(P_v, mu_ex_v, label=r"$\mu_{excess}$")
    plt.plot(P_v, mu_ideal_v, label=r"$\mu_{ideal}$")
    plt.plot(P_v, mu_tot_v, label=r"$\mu_{tot}$")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'p[$\epsilon / \sigma^{3}$]', fontsize=35)
    plt.ylabel(r'$\mu [\epsilon]$', fontsize=35)
    plt.legend(fontsize=25)

    ### Plot of sigma esteemed from blocks
    plt.figure()
    plt.title('Standard deviation on $\mu_{ex}$, T = 2.0', fontsize=35)
    plt.scatter(rho_v, sigma_v/np.abs(mu_ex_v))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid()
    plt.xlabel(r'$\rho$[$\sigma^{-3}$]', fontsize=35)
    plt.ylabel(r'$\sigma_{\mu}/\mu_{ex}$', fontsize=35)
    plt.show()
