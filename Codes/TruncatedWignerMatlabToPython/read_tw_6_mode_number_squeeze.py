# import time as timemodule
import matplotlib.pyplot as plt
import numpy as np


# np.random.seed(0)

# Calculate delta_p for given system parameters
def get_delta_p(eta, chi_p, kappa):
    return eta ** 2 / (2 * chi_p) - np.sqrt(eta ** 4 / (4 * chi_p ** 2) - kappa ** 2)


# Calculate eta for given system parameters
def get_eta(delta, chi, kappa):  # works with (delta_p, chi_p) and (delta_m, chi_m)
    return np.sqrt(chi * (delta ** 2 + kappa ** 2) / delta)


def time_in_micro_s_to_index(t1_micro_s, time):
    t1_milli_s = t1_micro_s / 1000
    t_0 = time[0]
    t_end = time[-1]
    timesteps = len(time)
    t_frac = (t1_milli_s - time[0]) / (time[-1] - time[0])
    index_frac = t_frac * (timesteps - 1)
    index_frac = round(index_frac)
    return index_frac


def power_func(y, a, r):
    return a * y ** r


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
    x_p = - 0.15 * 2 * np.pi  # -0.0009662061050309727 * 1000  # Coupling of plus channel in Hz
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

squeezing_vec = np.zeros((N_delta_p_points), dtype=np.csingle)
newfp = np.memmap(filename, dtype=np.csingle, mode='r', shape=(N_delta_p_points, 6, Npoints, Nrealiz, 1))

# index_min_squeezing_vec = np.zeros((N_delta_p_points))
pairs_max_squeezing_vec = np.zeros((N_delta_p_points))
pairs_loose_squeezing_vec = np.zeros((N_delta_p_points))
time_loose_squeezing_vec = np.zeros((N_delta_p_points))

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

    ### plot populations for small and high laser powers
    # if i  == 0 or i == N_delta_p_points-1:
    # if i == 0:
    # #     print(i)
    # #     plt.plot(time, rho0_mean, label='0,m=0')
    #
    #     font = {'size': 13}
    #     plt.rc('font', **font)
    #
    #     plt.plot(time*1000,rho0_mean, label='$ |0\\rangle_0 $', color='green')
    #     plt.plot(time*1000,rho1_mean, label='$ |+k\\rangle_{+1} $', color='blue')
    #     plt.plot(time*1000,rho2_mean, label='$ |-k\\rangle_{-1} $', color='red')
    #     plt.plot(time*1000,rho3_mean, label='$ |+k\\rangle_{-1} $', color='purple')
    #     plt.plot(time*1000,rho4_mean, label='$ |-k\\rangle_{+1} $',  color='orange')
    #     plt.plot(time*1000,rho5_mean, label='$ |+2k_x\\rangle_{0} $', color='grey')
    #     # plt.grid()
    #
    #     plt.yscale('log')
    #     plt.xlabel('$t \\, \\, (\\mu s$)')
    #     plt.ylabel('Mode occupation')
    #     plt.legend(loc = 4)
    #
    #     plt.savefig('plots/mod_occ_time_low_power_bla.svg')
    #     plt.show()

    # #
    J_z_vec = 1 / 2 * (rho1_vec - rho2_vec)
    J_z_var = (np.var(J_z_vec, axis=1))

    xi_N_squared = 4 * J_z_var / N
    xi_N_squared_coh = 4 * rho2_mean / 2 / N
    number_squeezing = xi_N_squared / xi_N_squared_coh


    ## evaluate number squeezing
    # if i == 0 or  i == N_delta_p_points -1:
    #     print(np.min(number_squeezing))



    ## Plot squeezing over time Part A
    # font = {'size': 15}
    # plt.rc('font', **font)
    # if i == 0 or i == N_delta_p_points - 1:
    #     round_delta_str = str(round(delta_p[i] / 2 / np.pi / 10 ** 6, 2))
    #     plt.plot(time * 1000, number_squeezing, label='$\\delta_+ / 2 \\pi = $' + round_delta_str + ' MHz')
    #


    squeezing_vec[i] = number_squeezing.min()
    # pairs_max_squeezing_vec[i] = rho4_mean[number_squeezing.argmin()]
    #time
    idx_loose_squeezing = np.argwhere(np.diff(np.sign(number_squeezing[:, 0] - 1)))[-1][0]
    pairs_loose_squeezing_vec[i] = rho2_mean[idx_loose_squeezing]
    time_loose_squeezing_vec[i] = time[idx_loose_squeezing]

    ### plot squeezing over number of pairs Part A
    # #
    # font = {'size': 15}
    # plt.rc('font', **font)
    # if i == 0 or i == N_delta_p_points-1:
    #     round_delta_str = str(round(delta_p[i] / 2 / np.pi / 10**6,2))
    #     plt.plot(rho2_mean[0:440],number_squeezing[0:440],label = '$\\delta_+ / 2 \\pi = $' + round_delta_str +  ' MHz')

### plot squeezing over number of pairs Part B
# plt.yscale('log')
# plt.xlabel('$\\langle N_p \\rangle$')
# plt.ylabel('$\\xi_N^2 / \\xi_{N,coh}^2$')
# plt.hlines(1,0,300,linestyles='dashed', label = 'SQL', colors='grey')
# plt.tight_layout()
# plt.legend()
# plt.savefig('plots/rns_sq_npair_bla.svg')
# plt.show()


### plot number of pairs at loose squeezing: +channel # and fititng included
#
#
# font = {'size': 16}
# plt.rc('font', **font)
#
# # FIND FIT TO CURVE WITH np.polyfit
# #
# # coefficients1 = np.polyfit(delta_p, pairs_loose_squeezing_vec, 1)
# # coefficients2 = np.polyfit(delta_p, pairs_loose_squeezing_vec, 2)
# # coefficients3 = np.polyfit(delta_p, pairs_loose_squeezing_vec, 3)
# # coefficients4 = np.polyfit(delta_p, pairs_loose_squeezing_vec, 4)
# #
# # print(coefficients2)
# # print(coefficients3)
# # print(coefficients4)
#
# #create polynomial function from the coefficients
# # polyfit1_func = np.poly1d(coefficients1)
# # polyfit2_func = np.poly1d(coefficients2)
# # polyfit3_func = np.poly1d(coefficients3)
# # polyfit4_func = np.poly1d(coefficients4)
# #
# # # Generate y values
# # polyfit1 = polyfit1_func(delta_p)
# # polyfit2 = polyfit2_func(delta_p)
# # polyfit3 = polyfit3_func(delta_p)
# # polyfit4 = polyfit4_func(delta_p)
# #
# # mse1 = np.mean((pairs_loose_squeezing_vec- polyfit1) **2)
# # mse2 = np.mean((pairs_loose_squeezing_vec- polyfit2) **2)
# # mse3 = np.mean((pairs_loose_squeezing_vec- polyfit3) **2)
# # mse4 = np.mean((pairs_loose_squeezing_vec- polyfit4) **2)
# # print('hi',mse1,mse2,mse3,mse4)
#
# # FIND FIT TO CURVE WITH curve_fit
#
# ##Initial guess for the coefficients
#
# #[ 3.38361502e-15 -2.18497152e-07 -1.85816865e+01]
# # initial_guess = [3.38*10**(-15) , 2]
# #
# # # # Fit the model to the data
# # params, params_covariance = curve_fit(power_func, -delta_p, pairs_loose_squeezing_vec, p0=initial_guess)
# # print('power params',params)
# #
# # power_func_fit = power_func(-delta_p, *params)
# #
# # mse1 = np.mean((pairs_loose_squeezing_vec- power_func_fit) **2)
# # print('approx mse', np.mean((pairs_loose_squeezing_vec- 1.331*(10**(-14))* (-delta_p)**(1.937)) **2))
# # print('powermse',mse1)
# # plt.plot(-delta_p/2/np.pi/10**6, power_func_fit)
# #
# # # plt.plot(-delta_p/2/np.pi/10**6, power_func(delta_p,initial_guess[0],initial_guess[1]))
# #
# plt.plot(delta_p/2/np.pi/10**6, pairs_loose_squeezing_vec)
# # plt.plot(delta_p/2/np.pi/10**6, polyfit1)
# # plt.plot(delta_p/2/np.pi/10**6, polyfit2)
# # plt.plot(delta_p/2/np.pi/10**6, polyfit3)
# plt.ylabel('$\\langle N_p \\rangle_{max,sq}$')
# plt.xlabel('$\\delta_+/2\\pi \\,\\,$(MHz)')
# plt.legend()
# # plt.xscale('log')
# plt.xlim((delta_p/2/np.pi/10**6)[0],(delta_p/2/np.pi/10**6)[-1])
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('plots/rns_npair_det_sq_loose_bla.svg')
# plt.show()

### Plot t_loose_squeezing over detuning

font = {'size': 16}
plt.rc('font', **font)

plt.plot(delta_p/2/np.pi/10**6, time_loose_squeezing_vec*1000)
plt.ylabel('$t_L \\, \\, (\\mu s)$')
plt.xlabel('$\\delta_+/2\\pi \\,\\,$(MHz)')
plt.xlim((delta_p/2/np.pi/10**6)[0],(delta_p/2/np.pi/10**6)[-1])
plt.tight_layout()
plt.legend()
plt.savefig('plots/rns_t_det_sq_loose_bla.svg')
plt.show()


# Plot number pairs
# plt.title('+ channel for different detunings')
# plt.ylabel('<number pairs>')
# plt.xlabel('t (micro sec)')
# plt.yscale('log')
# plt.legend()
# #plt.savefig('plots/sqP_corr_high_res_bla.svg') # corr means that we realize, that eta**2 prop to power
# plt.show()
# # #


### plot squeezing over detuing
### and Find fits to squeezing behaviour


# def laurent_polynomial(y, *coefficients):
#     return sum(c * y**(-i) for i, c in enumerate(coefficients))
#
#
#
# ##First order
#
# # Initial guess for the coefficients
# initial_guess = [0,0]
#
# # Fit the model to the data
# params, params_covariance = curve_fit(power_func, -delta_p, squeezing_vec, p0=initial_guess)
# print(params)
#
# power_fit_1 = power_func(-delta_p, *params)
#
# power_fit_rounded = power_func(-delta_p, 2.13*10**7,-0.99)
#
# mse1 = np.mean((squeezing_vec- power_fit_1) **2)
# mse_rounded = np.mean((squeezing_vec- power_fit_rounded) **2)
# print('mse power', mse1)
# print('mse power rounded', mse_rounded)
#
# ##Second order
# # Initial guess for the coefficients
# initial_guess = [0,0,0]
#
# # Fit the model to the data
# params, params_covariance = curve_fit(laurent_polynomial, delta_p, squeezing_vec, p0=initial_guess)
# print(params)
#
# laurent_fit_2 = laurent_polynomial(delta_p, *params)
#
# mse2 = np.mean((squeezing_vec- laurent_fit_2) **2)
# print('mse laurent2',mse2)
#
# plt.plot(delta_p/2/np.pi/10**6, squeezing_vec)
# # plt.plot(delta_p/2/np.pi/10**6, power_fit_rounded)
# # plt.plot(delta_p/2/np.pi/10**6, laurent_fit_2)
# plt.xlim((delta_p/2/np.pi/10**6)[0],(delta_p/2/np.pi/10**6)[-1])
#
#
# # plt.title('+ channel over detuning')
# # plt.plot(delta_p/2/np.pi/10**6, 10*np.log10(squeezing_vec))
# plt.ylabel('$(\\xi_N^2 / \\xi_{N,coh}^2)_{min}$')
# plt.xlabel('$\\delta_+/2\\pi \\,\\,$(MHz)')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.tight_layout()
#
# plt.savefig('plots/rns_sq_max_det_bla.svg')
# plt.show()


### plot squeezing over time Part B
# plt.ylabel('$\\xi_N^2 / \\xi_{N,coh}^2$')
# plt.xlabel('$t \\, \\, (\\mu s$)')
# plt.yscale('log')
# plt.hlines(1, 0, 200, linestyles='dashed', label='SQL', colors='grey')
# plt.tight_layout()
#
# plt.legend()
# plt.savefig('plots/rns_sq_time_bla.svg')
# plt.show()
