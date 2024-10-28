from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.integrate import solve_ivp
import time as tm


### FUNCTIONS ###


# The following two functions allow to change delta_p and eta simultaneously,
# such that the pair coupling of the chi_+ channel can stay constant (c.f. work H. Jaeger)

# Calculate delta_p for given system parameters
def get_delta_p(eta, chi_p, kappa):
    return eta ** 2 / (2 * chi_p) - np.sqrt(eta ** 4 / (4 * chi_p ** 2) - kappa ** 2)

# Calculate eta for given system parameters
def get_eta(delta, chi, kappa):  # works with (delta_p, chi_p) and (delta_m, chi_m)
    return np.sqrt(chi * (delta ** 2 + kappa ** 2) / delta)


# Define the equations of motion
def eoms(t, phi, gamma, gammaM, chi, chiM,omega0, a, omegaR):
    dphidt = np.zeros(6, dtype=np.csingle)

    dphidt[0] = -1j * chi * (2 * np.conj((phi[0]+np.sqrt(a)*phi[5])) * phi[1] * phi[2] + np.conj(phi[1]) * (phi[0]+np.sqrt(a)*phi[5]) * phi[1] + np.conj(phi[2]) * (phi[0]+np.sqrt(a)*phi[5]) * phi[2]) + \
                gamma * (np.conj(phi[2]) * phi[2] * (phi[0]+np.sqrt(a)*phi[5]) - np.conj(phi[1]) * phi[1] * (phi[0]+np.sqrt(a)*phi[5])) - \
                1j * chiM * (2 * np.conj((phi[0]+np.sqrt(a)*phi[5])) * phi[3] * phi[4] + np.conj(phi[3]) * (phi[0]+np.sqrt(a)*phi[5]) * phi[3] + np.conj(phi[4]) * (phi[0]+np.sqrt(a)*phi[5]) * phi[4]) + \
                gammaM * (np.conj(phi[4]) * phi[4] * (phi[0]+np.sqrt(a)*phi[5]) - np.conj(phi[3]) * phi[3] * (phi[0]+np.sqrt(a)*phi[5]))
    dphidt[1] = -1j * omega0 * phi[1] - 1j * chi * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[1] + np.conj(phi[2]) * (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5])) + \
                gamma * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[1] + (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5]) * np.conj(phi[2]))
    dphidt[2] = -1j * omega0 * phi[2] - 1j * chi * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[2] + np.conj(phi[1]) * (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5])) - \
                gamma * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[2] + (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5]) * np.conj(phi[1]))
    dphidt[3] = -1j * omega0 * phi[3] - 1j * chiM * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[3] + np.conj(phi[4]) * (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5])) + \
                gammaM * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[3] + (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5]) * np.conj(phi[4]))
    dphidt[4] = -1j * omega0 * phi[4] - 1j * chiM * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[4] + np.conj(phi[3]) * (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5])) - \
                gammaM * (np.conj((phi[0]+np.sqrt(a)*phi[5])) * (phi[0]+np.sqrt(a)*phi[5]) * phi[4] + (phi[0]+np.sqrt(a)*phi[5]) * (phi[0]+np.sqrt(a)*phi[5]) * np.conj(phi[3]))
    dphidt[5] = -1j * 4 * omegaR * phi[5] \
                            -1j*np.sqrt(a)*chi*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[1]*phi[2]+ np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*phi[1] + np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]) \
                        + gamma*np.sqrt(a)*(np.conj(phi[2])*phi[2]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[1])*phi[1]*(phi[0]+np.sqrt(a)*phi[5])) \
                        -1j*np.sqrt(a)*chiM*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[3]*phi[4]+ np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*phi[3] + np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*phi[4]) \
                        + gammaM*np.sqrt(a)*(np.conj(phi[4])*phi[4]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[3])*phi[3]*(phi[0]+np.sqrt(a)*phi[5]))
    return dphidt


# "tw_for_Npoints_Nrealiz_n_seeds" does the actual TW simulation
# For given initial values this function returns a matrix that contains
# all modes at all times, realizations and seeds.
def tw_for_Npoints_Nrealiz_n_seeds(initial_values_dic):

    # Each mode gets its own matrix containing all times, realizations and seeds.
    phi0_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 0, 0     0_vec
    phi1_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf =1 ,+k     1_vec
    phi2_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = -1 , -k  M1_vec
    phi3_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = -1, +k   M1_M:vec
    phi4_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 1 ,-k    1_M_vec
    phi5_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 0, +-2k  5_vec

    # for each classical seed
    for seed_index in range(0, len(initial_values_dic['n_list'])):

        nSeed = initial_values_dic['n_list'][seed_index] # current classical seed

        # for each realization
        for realiz_index in range(0, initial_values_dic['Nrealiz']):

            N_temp = np.random.normal(loc=initial_values_dic['N'], scale=initial_values_dic['DeltaN'], size=1)[0]  # N mean, DeltaN standard deviation

            # switch seed type usually there is also an option for poison seed
            # deterministic:
            N_temp_therm = nSeed
            N_temp_therm2 = nSeed

            phi_initial = np.zeros(6, dtype=np.csingle)
            phi_initial[0] = np.sqrt(N_temp)  # All atoms in mF = 0
            phi_initial[1] = np.sqrt(N_temp_therm)  # Atoms in mF = 1, +k
            phi_initial[2] = np.sqrt(N_temp_therm2)  # Atoms in mF = -1 ,- k

            phi_initial[3] = np.sqrt(N_temp_therm)  # Atoms in mF = -1, +k
            phi_initial[4] = np.sqrt(N_temp_therm2)  # Atoms in mF = 1, -k
            phi_initial[5] = 0 # Atoms in mF = 0, +-2k_x

            # Sample Quantum 1/2 noise
            phi_initial[0] = phi_initial[0] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[1] = phi_initial[1] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[2] = phi_initial[2] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[3] = phi_initial[3] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[4] = phi_initial[4] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[5] = phi_initial[5] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
                             np.random.normal(loc=0, scale=1, size=1)[0]

            #scipy solver
            t_1 = tm.time()
            sol = solve_ivp(fun=eoms, t_span=[initial_values_dic['time'][0],initial_values_dic['time'][-1]], args=(initial_values_dic['gamma'],initial_values_dic['gammaM'],initial_values_dic['chi'],initial_values_dic['chiM'],initial_values_dic['omega0'],initial_values_dic['a'],initial_values_dic['omegaR']), y0=phi_initial, method='DOP853',atol=1e-8,rtol=1e-8, t_eval=initial_values_dic['time'])#atol=1e-6,rtol=1e-4, t_eval=tbounds)
            t_2 = tm.time()

            phi0_vec[:, realiz_index, seed_index] = sol.y[0]
            phi1_vec[:, realiz_index, seed_index] = sol.y[1]
            phi2_vec[:, realiz_index, seed_index] = sol.y[2]
            phi3_vec[:, realiz_index, seed_index] = sol.y[3]
            phi4_vec[:, realiz_index, seed_index] = sol.y[4]
            phi5_vec[:, realiz_index, seed_index] = sol.y[5]

            if realiz_index % 50 == 1:
                print('NpSeed=', initial_values_dic['n_list'][seed_index], ', EOMs ', realiz_index, ' out of ', initial_values_dic['Nrealiz'],
                      ', Time for single EOM=', t_2-t_1,'s')
    return np.array([phi0_vec,phi1_vec,phi2_vec,phi3_vec,phi4_vec,phi5_vec])








### MAIN PART ###

if __name__ == '__main__':
    hbar = 1.054571628 * (10 ** (-34))
    mRb87 = 1.443160648 * (10 ** (-25))
    epsilon0 = 8.8541878128e-12
    lam_M = 790.02e-9  # k_x and k_z both yield lam_M
    E0 = 403  # V/m electric field per cavity photon;
    c = 2.9979e8 # speed of light
    omegaRec = hbar * 2 * np.pi * np.pi / (mRb87 * lam_M * lam_M)
    omegaR = omegaRec / 1000  # everything in kHz and ms
    p = 0.7024 * 1e6  # First order splitting of Rb87 in Hz
    q = 144  # 2md second order splitting of Rb87 in Hz

    N = 80000  # atom number
    DeltaN = 0.045 * N  # Fluctuations of Atom Number
    tbounds = np.array([0, 0.2])  # Evoution time in ms
    # eta = 2*np.pi*3.4e3 # Raman coupling

    Npoints = 2000 # time points
    time = np.linspace(tbounds[0], tbounds[1], Npoints) # time array

    # Initialize system parameters
    eta_old_setup = 2 * np.pi * 1.7e3  # original laser power yields Raman coupling "eta_old_setup"
    Kappa = 2 * np.pi * 1.25e6  # Cavity losses in Hz
    omegaZ = 2 * np.pi * 7.09e6  # Zeemansplitting in Hz
    x_p = - 0.15 * 2 * np.pi  # Pair coupling for chi_- channel in Hz
    delta_p_old_setup = get_delta_p(eta_old_setup, x_p, Kappa) # original laser power yields detuning "delta_p_old_setup"

    eta_max = eta_old_setup * np.sqrt(3) # strong laser power yields raman coupling "eta_max"
    delta_p_max = get_delta_p(eta_max, x_p, Kappa) # strong laser power yields detuning "delta_p_max"

    # Decide for original laser strength or strong laser drive
    # (here decision for stronger laser strength)
    delta_p = delta_p_max
    eta = eta_max

    # Two Photon Detunings
    deltaC = delta_p - omegaZ
    delta_m = (deltaC - omegaZ)  # Two Photon Detungs for - channel
    omega0 = 0.5 * (4 * omegaRec + 2 * np.pi * q * (omegaZ / 2 / np.pi / p) ** 2) / 1000

    # Calculate Two Photon And Four Photon Couplings
    x_p = x_p / 1000 # Pair coupling for chi_- Channel in kHz
    Gamma_p = (eta ** 2 * Kappa / (delta_p ** 2 + (Kappa) ** 2)) / 1000  # Dissipative coupling for chi_+ Channel in kHz
    x_m = (eta ** 2 * delta_m / (delta_m ** 2 + (Kappa) ** 2)) / 1000  # Pair coupling for chi_- Channel in kHz
    Gamma_m = (eta ** 2 * Kappa / (delta_m ** 2 + (Kappa) ** 2)) / 1000  # Dissipative coupling for chi_- Channel in kHz






    ### DO SIMULATIONS ###

    # This script creates a bin file that contains the results of the TW simulation.
    # The script "read_data_tw.py" is used to analyse the results.

    # We make up a name for the bin file
    filename = 'tw_2000_npoints_1000_nrealiz_max_laser_power_non_seeded.bin'


    n_list = np.array([0])  # List of average initial pair number from classical seeds
    Nrealiz = 1000  # Number of TW simulation realizations

    scalecoupling_k2 = 1  # Scale Coupling to m=0, k_x=+-2k mode, set to 0 to supress coupling to this mode and 1 to consider it correctily

    a = 2 / 3
    a = (scalecoupling_k2) ** 2 * a

    # create a dictionary containing all parameters for the TW simulation
    inital_values_dic = {'Npoints': Npoints,
                          'Nrealiz': Nrealiz,
                          'n_list': n_list,
                          'N': N,
                          'DeltaN': DeltaN,
                          'time': time,
                          'gamma': Gamma_p,
                          'gammaM': Gamma_m,
                          'chi': x_p,
                          'chiM': x_m,
                          'omega0': omega0,
                          'a': a,
                          'omegaR': omegaR
                          }

    # We use multiple processors to speed up the calculation.
    # Attention: if we use some shared computer replace "(cpu_count() - 1)" by a reasonable number
    with Pool(processes=(cpu_count() - 2)) as pool:
        # Do the actual TW simulation and write all the results in the "tw_matrix"
        result = pool.map(tw_for_Npoints_Nrealiz_n_seeds, [inital_values_dic])

    # This is still part of the multi-processing
    pool.close()
    pool.join()
    # Multi-processing is finished


    # Shape of tw_matrix: ( 6,         Npoints,    Nreliz,         length(n_list)
    #                     modes       time        realization     seeds
    # axis                0           1           2               3
    tw_matrix = result[0]
    shape_tw_matrix = (6, Npoints, Nrealiz, len(n_list))


    # Create empty bin file with correct shape to contain tw_matrix
    bin_file = np.memmap(filename, mode='w+', dtype=np.csingle, shape=(shape_tw_matrix))

    # Write data from tw_matix in the bin file
    # I think,we could just do bin_file = tw_martix. For bigger tw_matrices,
    # it takes less memory to write the bin_file with a for-loop.
    # Here we write each mode separately. The bin file is the same as in the
    # case of writing the hole bin_file at once.
    for mode_index, mode_matrix in enumerate(tw_matrix):
        bin_file[mode_index] = mode_matrix
    del bin_file

