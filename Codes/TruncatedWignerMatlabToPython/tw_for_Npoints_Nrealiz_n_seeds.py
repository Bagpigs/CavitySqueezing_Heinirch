import numpy as np
from scipy.integrate import solve_ivp
import time as tm

from Codes.TruncatedWignerMatlabToPython.eoms import eoms


def tw_for_Npoints_Nrealiz_n_seeds(initial_values_dic):
    #initialize different matrices to carry all information about different simulations and different times
    phi0_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 0        0_vec
    phi1_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf =1 ,+k     1_vec
    phi2_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = -1 , -k  M1_vec
    phi3_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = -1, +k   M1_M:vec
    phi4_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 1 ,-k    1_M_vec
    phi5_vec = np.zeros((initial_values_dic['Npoints'],initial_values_dic['Nrealiz'],len(initial_values_dic['n_list'])),dtype=np.csingle) #     # mf = 0, +-2k  5_vec



    ###### #initialize phi (which wont be squared later)

    # for each classical seed
    for seed_index in range(0, len(initial_values_dic['n_list'])):
        # Choose average pair occupation, what is this?
        nSeed = initial_values_dic['n_list'][seed_index]
        for realiz_index in range(0, initial_values_dic['Nrealiz']):

            N_temp = np.random.normal(loc=initial_values_dic['N'], scale=initial_values_dic['DeltaN'], size=1)[0]  # N mean, DeltaN standard deviation

            # switch seed type usually there is also an option for poison seed
            # deterministic:
            N_temp_therm = nSeed
            N_temp_therm2 = nSeed

            # initialize Phi

            # why in matlab script 5?
            phi_initial = np.zeros(6, dtype=np.csingle)
            phi_initial[0] = np.sqrt(N_temp)  # All atoms in mF = 0
            phi_initial[1] = np.sqrt(N_temp_therm)  # Atoms in mF = 1, +k
            phi_initial[2] = np.sqrt(N_temp_therm2)  # Atoms in mF = -1 ,- k

            phi_initial[3] = np.sqrt(N_temp_therm)  # Atoms in mF = -1, +k
            phi_initial[4] = np.sqrt(N_temp_therm2)  # Atoms in mF = 1, -k
            phi_initial[5] = 0 # Atoms in  mF = 0, +-2k_x

            # Sample Quantum 1/2 noise
            phi_initial[0] = phi_initial[0] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[1] = phi_initial[1] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[2] = phi_initial[2] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[3] = phi_initial[3] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[4] = phi_initial[4] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]
            phi_initial[5] = phi_initial[5] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                             np.random.normal(loc=0, scale=1, size=1)[0]

            # test
            t_1 = tm.time()

            #scipy solver
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