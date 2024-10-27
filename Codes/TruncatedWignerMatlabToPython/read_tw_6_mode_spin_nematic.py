from multiprocessing import Pool, cpu_count

import numpy as np
# import time as timemodule
import matplotlib.pyplot as plt

from Codes.TruncatedWignerMatlabToPython.tw_for_Npoints_Nrealiz_n_seeds import tw_for_Npoints_Nrealiz_n_seeds



# np.random.seed(0)

# Calculate delta_p for given system parameters
def get_delta_p(eta, chi_p, kappa):
    return eta ** 2 / (2 * chi_p) - np.sqrt(eta ** 4 / (4 * chi_p ** 2) - kappa ** 2)


# Calculate eta for given system parameters
def get_eta(delta, chi, kappa):  # works with (delta_p, chi_p) and (delta_m, chi_m)
    return np.sqrt(chi * (delta ** 2 + kappa ** 2) / delta)

def time_in_micro_s_to_index(t1_micro_s,time):
    t1_milli_s = t1_micro_s /1000
    t_0 = time[0]
    t_end = time[-1]
    timesteps = len(time)
    t_frac = (t1_milli_s-time[0]) / (time[-1] - time[0])
    index_frac = t_frac * (timesteps-1)
    index_frac = round(index_frac)
    return index_frac





if __name__ == '__main__':
    hbar = 1.054571628 * (10 ** (-34))
    mRb87 = 1.443160648 * (10 ** (-25))
    epsilon0 = 8.8541878128e-12
    # ?
    lam_M = 790.02e-9  # doesnt matter whether it is k in x or z direction, see notes.
    E0 = 403  # V/m electric field per cavity photon;
    c = 2.9979e8
    omegaRec = hbar * 2 * np.pi * np.pi / (mRb87 * lam_M * lam_M)
    omegaR = omegaRec / 1000  # everythig in kHz and ms
    p = 0.7024 * 1e6  # First order splitting of Rb87 in Hz
    q = 144  # 2md second order splitting of Rb87 in Hz

    N = 80000  # 30000#80000 #80000 # atom number
    DeltaN = 0.045 * N  # Fluctuations of Atom Number
    tbounds = np.array([0, 0.2])  # np.array([0,0.2]) # Evoution time in ms
    # eta = 2*np.pi*3.4e3 # Raman coupling


    Npoints = 2000
    time = np.linspace(tbounds[0], tbounds[1], Npoints)

    # Initialize system parameters
    eta_old_setup = 2 * np.pi * 1.7e3  # Raman coupling
    Kappa = 2 * np.pi * 1.25e6  # Cavity losses in Hz
    omegaZ = 2 * np.pi * 7.09e6  # 2*np.pi * 1.09e6 #Zeemansplitting in Hz
    x_p = - 0.15 * 2 * np.pi#-0.0009662061050309727 * 1000  # Coupling of plus channel in Hz
    delta_p_old_setup = get_delta_p(eta_old_setup, x_p, Kappa)

    # Define maximal eta, delta_p possible
    eta_max_multi = np.sqrt(3)
    eta_max = eta_max_multi * eta_old_setup
    delta_p_max = get_delta_p(eta_max, x_p, Kappa)

    # Define corresponding vectors (equally spaced in delta_p)
    N_delta_p_points = 30
    delta_p = np.linspace(delta_p_old_setup, delta_p_max, N_delta_p_points)
    eta = get_eta(delta_p, x_p, Kappa)

    # Define corresponding vectors of other system parameters

    # Two Photon Detunings
    deltaC = delta_p - omegaZ
    delta_m = (deltaC - omegaZ)  # Two Photon Detungs for - channel
    # what about points and where does fromula come from? (not the same as in my latex)
    omega0 = 0.5 * (4 * omegaRec + 2 * np.pi * q * (omegaZ / 2 / np.pi / p) ** 2) / 1000

    # Calculate Two Photon And Four Photon Couplings
    # what about points and where does fromula come from?
    Gamma_p = (eta ** 2 * Kappa / (delta_p ** 2 + (Kappa) ** 2)) / 1000  # Dissipative coupling for chi_+ Channel
    x_m = (eta ** 2 * delta_m / (delta_m ** 2 + (Kappa) ** 2)) / 1000  # Pair coupling for chi_- Channel
    Gamma_m = (eta ** 2 * Kappa / (delta_m ** 2 + (Kappa) ** 2)) / 1000  # Pair coupling for chi_+ Channel

    ### DO SIMULATIONS ###

    # ?
    n_list = np.array([0])  # List of average initial pair number from classical seeds
    Nrealiz = 1000  # Number of TW simulations pro setting (used for averaging)
    # ?
    scalecoupling_k2 = 1  # 1  # Scale Coupling to m=0, k_x=+-2k mode, set to 0 to supress coupling to this mode and 1 to consider it correctily

    # EOM
    # still to check, whether following constants are defined useful!
    a = 2 / 3
    a = (scalecoupling_k2) ** 2 * a

    filename = 'tw_30_1000_045.bin'

##end HEADER


# Shape of tw_matrix_variable_detuning (N_delta_p_points,      6,         Npoints,     Nreliz,         length(n_list)
#                                       detunings               modes       time        realization     seeds
# axis                                  0                       1           2               3           4


font = {'size': 14}
plt.rc('font', **font)

squeezing_vec = np.zeros((N_delta_p_points),dtype=np.csingle)
time_loose_wineland_vec = np.zeros(N_delta_p_points)
newfp = np.memmap(filename, dtype=np.csingle, mode='r', shape=(N_delta_p_points, 6, Npoints, Nrealiz, 1))



for i in range(29,30):#N_delta_p_points):

    tw_matrix = newfp[i]

    # Shape of tw_matrix: ( 6,         Npoints,    Nreliz,         length(n_list)
    #                     modes       time        realization     seeds
    # axis                0           1           2               3

    phi0_vec = tw_matrix[0]
    phi1_vec = tw_matrix[1]
    phi2_vec = tw_matrix[2]

    # Shape of S_x´: (      Npoints,        Nrealiz,            length(n_list)

    #def following [chapman,2012] #double checked!
    S_x = 1/ 2**0.5 * (np.conj(phi1_vec) * phi0_vec + np.conj(phi0_vec) * phi2_vec + \
                       np.conj(phi0_vec) * phi1_vec + np.conj(phi2_vec) * phi0_vec)
    print(np.shape(S_x))

    S_x_mean = np.mean(S_x, axis = 1)
    #S_x_var = np.var(S_x, axis=1) # this is too big by factor 2

    S_y = 1 / (2**0.5 * 1j) * ( np.conj(phi1_vec) * phi0_vec - np.conj(phi0_vec) * phi1_vec + \
                                np.conj(phi0_vec) * phi2_vec - np.conj(phi2_vec) * phi0_vec )
    #S_y_var = np.var(S_y, axis=1) # this is too big by factor 2

    #def following [wang,2020] #double checked!
    S_z = np.conj(phi1_vec) * phi1_vec - np.conj(phi2_vec) * phi2_vec
    #S_z_var = np.var(S_z,axis=1) # if I once use this term: here I would have to think about factor two, in rns we defined j_z with a factor 1/2
    # S_z_var = np.var(S_z, axis=1)

    #def following wang paper 2020 #double checked!
    Q_yz = 1/ 2**0.5 * ( -1j * np.conj(phi1_vec) * phi0_vec + 1j* np.conj(phi0_vec) * phi2_vec + \
                       1j * np.conj(phi0_vec) * phi1_vec -1j* np.conj(phi2_vec) * phi0_vec)
    # Q_xz = 2 * S_x * S_z

    #calculated on my own (maybe double check calc with matrix calc) #note, that this operators is not symmetric and not of the quadratic from c_q^* c_q but sort of third order
    comm_S_x_Q_yz = 1j * (np.conj(phi1_vec)*phi1_vec + np.conj(phi1_vec) * phi2_vec - 2 * np.conj(phi0_vec) * phi0_vec + np.conj(phi2_vec) * phi1_vec + np.conj(phi2_vec) * phi2_vec)
    #absolute value of comm expecatation value for t=0, 80000 atoms is 160,000. we are squeezed, if the variance of one quatdrature is below 160,000 / 2 . at a certain time this value shrinks from 160,000 to 43,000


    comm_S_x_Q_yz_exp_absolute_val = np.abs(np.mean(comm_S_x_Q_yz, axis = 1))

    #shape comm_S_x_Q_yz_exp_absolute_val:
    #  axis     0               1
    #           npoints         seeds



    #is this really called heisenberg limit? anyways. is the variance of one quadrature is lower than this, then we have squeezing
    heis_limit_for_one_quadrature_var = comm_S_x_Q_yz_exp_absolute_val/2


    Q_yz_var = np.var(Q_yz, axis=1)
    # Q_xz_var = np.var(Q_xz, axis=1)
    # for some reason this var starts with 150,000 (so around 160,000) and not around 80,000 where it should start

    #i did the csv data with 300 theta values
    theta = np.linspace(0, 2 * np.pi, 300)

    quadrature_operator_matrix = np.array([np.cos(theta[i]) * S_x + np.sin(theta[i]) * Q_yz for i in range(len(theta))])


    #shape quadrature operator matrix:
    #  axis     0               1               2               3
    #           theta           npoints         nrealiz         seeds



    # ASSUME that the VAR-FACTOR-2-PROBLEM is due to our initialization of the truncated wigner:
    # this here is the first point where we calculate a variance and therefor we just add the factor 2 here.
    # in any further calcuations this problem should be solved.
    quadrature_operator_matrix_var = np.var(quadrature_operator_matrix, axis=2) /2
    print(np.shape(quadrature_operator_matrix_var))

    #shape quadrature operator matrix:
    #  axis     0               1               2
    #           theta           npoints         seeds



    quadrature_operator_matrix_var_min_theta = np.min(quadrature_operator_matrix_var, axis=0)
    print(np.shape(quadrature_operator_matrix_var_min_theta))

    # shape quadrature_operator_matrix_var_min_theta:
    #  axis     0             1
    #           npoints       seeds



    #define min_theta wineland parameter including misterious 2
    wineland_param_min_theta = 4 * N * quadrature_operator_matrix_var_min_theta[0:1000] / comm_S_x_Q_yz_exp_absolute_val[0:1000] ** 2

    # shape wineland_param_min_theta:
    #  axis     0             1
    #           npoints       seeds

    ## find the time when wineland parameter gets above 1
    idx_loose_wineland = np.argwhere(np.diff(np.sign(wineland_param_min_theta[:, 0] - 1)))[-1][0]
    time_loose_wineland_vec[i] = time[idx_loose_wineland]


    ### Plot maximal entanglement over detuning Part A

    #we include the misterious 2round_delta_str = str(round(delta_p[i] / 2 / np.pi / 10 ** 6, 2))
    wineland_param_min_theta_time = np.min(wineland_param_min_theta, axis=0)
    squeezing_vec[i] = wineland_param_min_theta_time



    ### Plot Sx Sy at different times
    #
    # timearray = np.array([0, 100, 180])
    #
    # for t_1 in timearray:
    #     t_index = time_in_micro_s_to_index(t_1,time)
    #     plt.scatter(np.real(S_x[t_index,:,0])/1000,np.real(S_y[t_index,:,0])/1000,s=4)
    #     # plt.xlabel('$\\langle S_x \\rangle \\, \\,\\, (10^3)$')
    #     plt.xlabel('$ S_x \\, \\,\\, (10^3)$')
    #     # plt.ylabel('$\\langle Q_{yz} \\rangle\\, \\, \\,(10^3)$')
    #     plt.ylabel('$ S_{y} \\, \\, \\,(10^3)$')
    #     # plt.figure(figsize=(5, 4))  # Adjust width and height as needed
    #     plt.tight_layout()
    #     plt.show()
    #     # plt.savefig(plot_name)
    #
    # ### Plot scatter at four different times shape of the squeezing
    # timearray= np.array([0,20,70])
    #
    # for t_1 in timearray:
    #     t_index = time_in_micro_s_to_index(t_1,time)
    #     plt.scatter(np.real(S_x[t_index,:,0])/1000,np.real(Q_yz[t_index,:,0])/1000,s=4)
    #     # plt.xlabel('$\\langle S_x \\rangle \\, \\,\\, (10^3)$')
    #     plt.xlabel('$ S_x \\, \\,\\, (10^3)$')
    #     # plt.ylabel('$\\langle Q_{yz} \\rangle\\, \\, \\,(10^3)$')
    #     plt.ylabel('$ Q_{yz} \\, \\, \\,(10^3)$')
    #     # plt.figure(figsize=(5, 4))  # Adjust width and height as needed
    #     plt.tight_layout()
    #     plot_name = 'plots/sns_qyz_sx_' + str(t_1) +'_bla.svg'
    #     plt.savefig(plot_name)
    #     plt.show()


    ### Plot wineland parameter over theta Part A
    # wineland_param = 4 * N * quadrature_operator_matrix_var / comm_S_x_Q_yz_exp_absolute_val**2
    # t_index_20 = time_in_micro_s_to_index(20,time)
    # print(np.shape(wineland_param))
    # plt.plot(theta, wineland_param[:,t_index_20,0])
    # ax = plt.gca()
    # plt.ylabel('$\\xi^2(\\theta)$')
    # plt.xlabel('$\\theta$',labelpad=0)
    # plt.yscale('log')
    # plt.hlines(1,0,2*np.pi,linestyles='dashed', label = 'SQL', colors='grey')
    # ax.set_xticks(np.array([0,np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi]))
    # ax.set_xticklabels([r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$',
    #              r'$\frac{4\pi}{3}$', r'$\frac{5\pi}{3}$', r'$2\pi$'])
    # plt.tight_layout()
    #
    # plt.savefig('plots/sns_wineland_theta_bla.svg')
    # plt.show()
    #




    ###  Plot Wineland parameter over time part A

    #the devided by two is needed, because for mysterious reasons the variance of Sx is initially around 160,000 and not 80,000
    # plt.plot(time, S_x_var/2/heis_limit_for_one_quadrature_var, label='S_x_var')
    # plt.plot(time,Q_yz_var/2/heis_limit_for_one_quadrature_var, label='Q_yz_var')
    # plt.plot(time,quadrature_operator_matrix_var_min/2/heis_limit_for_one_quadrature_var, label='quad_op_min_var') # minimal ariance over different theta


    #BUT better to plot Wineland parameter instead (again I add the misterious 2)
    #note, that N actually also has variance (which is experimental caused)

    if i == 0 or i == N_delta_p_points - 1:
        round_delta_str = str(round(delta_p[i] / 2 / np.pi / 10 ** 6, 2))
        plt.plot(time[0:1000]*1000, 4 * N * quadrature_operator_matrix_var_min_theta[0:1000]/comm_S_x_Q_yz_exp_absolute_val[0:1000]**2,label='$\\delta_+ / 2 \\pi = $' + round_delta_str + ' MHz')

    # plt.plot(time,Q_xz_var, label='Q_xz_var')
    # plt.plot(time,S_y_var, label='S_y_var')
    # plt.plot(time,S_z_var, label='S_z_var')


## plot Wineland parameter over time part B
plt.hlines(1,0,100,linestyles='dashed', colors='grey')
plt.ylabel('min$_\\theta \\, \\, \\xi^2(\\theta)$')
plt.xlabel('$t \\, \\, (\\mu s$)')
plt.yscale('log')
# plt.hlines(1,0,200,linestyles='dashed', label = 'SQL', colors='grey')
plt.legend()
plt.savefig('plots/sns_wineland_time_bla.svg')
plt.show()


### plot maximal entanglement over detuning Part B
# np.savetxt("sns_squeezing_vec_bla.csv", np.real(squeezing_vec), delimiter=",")
# np.savetxt("sns_detuning_vec_bla.csv", delta_p, delimiter=",")
# np.savetxt("sns_time_loose_wineland_vec_bla.csv", time_loose_wineland_vec, delimiter=",")



#### Plot  population ratio
# for i in range(N_delta_p_points):
#
#     rho0_vec = np.abs(newfp[i][0]) ** 2
#     rho1_vec = np.abs(newfp[i][1]) ** 2
#     rho2_vec = np.abs(newfp[i][2]) ** 2
#     rho3_vec = np.abs(newfp[i][3]) ** 2
#     rho4_vec = np.abs(newfp[i][4]) ** 2
#     rho5_vec = np.abs(newfp[i][5]) ** 2
#
#     rho0_mean = np.mean(rho0_vec, axis=1)
#     rho1_mean = np.mean(rho1_vec, axis=1)
#     rho2_mean = np.mean(rho2_vec, axis=1)
#     rho3_mean = np.mean(rho3_vec, axis=1)
#     rho4_mean = np.mean(rho4_vec, axis=1)
#     rho5_mean = np.mean(rho5_vec, axis=1)
#
#     #
#     if i % 14 == 0:
#         #     print(i)
#         #     plt.plot(time, rho0_mean, label='0,m=0')
#         plt.plot(time, rho2_mean / rho3_mean, label='2mode by 3 mode')
#         plt.plot(time, rho2_mean / rho5_mean, label='2mode by 5 mode')
# plt.title('mode population ratio for different detunings')
# plt.ylabel('mode population ratio')
# plt.xlabel('t (milli seconds)')
# plt.yscale('log')
# plt.legend()
# plt.savefig('plots/sns_3mode_assumtion_bla.svg') # corr means that we realize, that eta**2 prop to power
# plt.show()
# # # #
