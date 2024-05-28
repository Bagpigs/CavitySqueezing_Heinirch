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
    DeltaN = 0.00 * N  # Fluctuations of Atom Number
    tbounds = np.array([0, 0.2])  # np.array([0,0.2]) # Evoution time in ms
    # eta = 2*np.pi*3.4e3 # Raman coupling


    Npoints = 2000
    time = np.linspace(tbounds[0], tbounds[1], Npoints)

    # Initialize system parameters
    eta_old_setup = 2 * np.pi * 1.7e3  # Raman coupling
    Kappa = 2 * np.pi * 1.25e6  # Cavity losses in Hz
    omegaZ = 2 * np.pi * 7.09e6  # 2*np.pi * 1.09e6 #Zeemansplitting in Hz
    x_p = -0.0009662061050309727 * 1000  # Coupling of plus channel in Hz
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

    filename = 'tw_detuning_02_30_1000_6_corr.bin'

squeezing_vec = np.zeros((N_delta_p_points),dtype=np.csingle)
newfp = np.memmap(filename, dtype=np.csingle, mode='r', shape=(N_delta_p_points, 6, Npoints, Nrealiz, 1))

# index_min_squeezing_vec = np.zeros((N_delta_p_points))
pairs_max_squeezing_vec = np.zeros((N_delta_p_points))
pairs_loose_squeezing_vec = np.zeros((N_delta_p_points))

for i in range(N_delta_p_points):

    rho0_vec = np.abs(newfp[i][0]) ** 2
    rho1_vec = np.abs(newfp[i][1]) ** 2
    rho2_vec = np.abs(newfp[i][2]) ** 2
    rho3_vec = np.abs(newfp[i][3]) ** 2
    rho4_vec = np.abs(newfp[i][4]) ** 2
    rho5_vec = np.abs(newfp[i][5]) ** 2

    rho0_mean = np.mean(rho0_vec, axis=1)
    rho1_mean = np.mean(rho1_vec, axis=1)
    rho2_mean = np.mean(rho2_vec, axis=1)
    rho3_mean = np.mean(rho3_vec, axis=1)
    rho4_mean = np.mean(rho4_vec, axis=1)
    rho5_mean = np.mean(rho5_vec, axis=1)
    #
    if i %14  == 0:
    #     print(i)
    #     plt.plot(time, rho0_mean, label='0,m=0')
        plt.plot(time, rho2_mean/rho3_mean, label='2mode by 3 mode')
        plt.plot(time, rho2_mean/rho5_mean, label='2mode by 5 mode')
        # plt.plot(time, rho3_mean, label='k,m=-1')
    # # # plt.plot(time, rho4_mean, label='-k,m=1')
    #     plt.plot(time, rho5_mean, label='2k,m=0')
    #     plt.grid()
    # plt.legend()
    # plt.show()


    # #
    J_z_vec = 1 / 2 * (rho1_vec - rho2_vec)
    J_z_var = (np.var(J_z_vec, axis=1))

    xi_N_squared = 4 * J_z_var / N
    xi_N_squared_coh = 4 * rho2_mean / 2 / N
    number_squeezing = xi_N_squared / xi_N_squared_coh

    # Plot squeezing over time
    # if i % 8 == 0:
    #     plt.plot(time*1000, number_squeezing)
    #     # plt.show()
    #
    squeezing_vec[i] = number_squeezing.min()
    # pairs_max_squeezing_vec[i] = rho4_mean[number_squeezing.argmin()]
    #
    idx_loose_squeezing = np.argwhere(np.diff(np.sign(number_squeezing[:,0]-1)))[-1][0]
    pairs_loose_squeezing_vec[i] = rho2_mean[idx_loose_squeezing]



    #plot squeezing over number of pairs
    # if i == N_delta_p_points-1:
    #     plt.plot(rho2_mean[0:440],number_squeezing[0:440])



#plot squeezing over number of pairs
# plt.title('+ channel squeezing for max power')
# plt.yscale('log')
# plt.ylabel('squeezing')
# plt.xlabel('<Npairs>')
# plt.savefig('plots/sq_Npairs_P_corr_max_power_high_res.svg')
# plt.show()
#


# plot number of pairs at loose squeezing: +channel
# plt.plot(-delta_p/2/np.pi/10**6, pairs_loose_squeezing_vec)
# plt.title('+ channel number of pairs at loose squeezing')
# plt.ylabel('<number pairs>')
# plt.xlabel('d_p in 2 pi MHz')
# plt.legend()
# plt.savefig('plots/Npair_sq_loose_P_corr_high_res.svg')
# plt.show()


# Plot number pairs
plt.title('+ channel for different detunings')
plt.ylabel('<number pairs>')
plt.xlabel('t (micro sec)')
plt.yscale('log')
plt.legend()
#plt.savefig('plots/sqP_corr_high_res_bla.svg') # corr means that we realize, that eta**2 prop to power
plt.show()
# # #

# plot squeezing over detuing


# # plot squeezing
# plt.title('+ channel for different detunings')
# plt.ylabel('squeezing')
# plt.xlabel('t (micro sec)')
# plt.yscale('log')
# plt.legend()
# plt.savefig('plots/sqP_all_corr_high_res.svg')
# plt.show()



