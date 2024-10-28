# why in matlab script 5?
import numpy as np


N_temp = 80000
Nrealiz = 50000

phi0_vec = np.zeros(
        Nrealiz,
        dtype=np.csingle)  # # mf = 0        0_vec
phi1_vec = np.zeros(
    Nrealiz,
    dtype=np.csingle)  # # mf =1 ,+k     1_vec
phi2_vec = np.zeros(
    Nrealiz,
    dtype=np.csingle)  # # mf = -1 , -k  M1_vec
phi3_vec = np.zeros(
    Nrealiz,
    dtype=np.csingle)  # # mf = -1, +k   M1_M:vec
phi4_vec = np.zeros(
    Nrealiz,
    dtype=np.csingle)  # # mf = 1 ,-k    1_M_vec
phi5_vec = np.zeros(
    Nrealiz,
    dtype=np.csingle)  # # mf = 0, +-2k  5_vec

for realiz_index in range(0, Nrealiz):
    
    phi_initial = np.zeros(6, dtype=np.csingle)
    phi_initial[0] = np.sqrt(N_temp)  # All atoms in mF = 0
    # phi_initial[1] = np.sqrt(0.5)
    # phi_initial[2] = np.sqrt(0.5)
    # phi_initial[3] = np.sqrt(0.5)
    # phi_initial[4] = np.sqrt(0.5)
    # phi_initial[5] = np.sqrt(0.5)

    #
    # Sample Quantum 1/2 noise # like it is written here, the variance for each quantity is 1 (0.5 real part variance + 0.5 imag part variance)
    # correct would be to replace np.sqrt(0.5) by 0.5 everywhere in the lines below
    # phi_initial[0] = phi_initial[0] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[1] = phi_initial[1] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[2] = phi_initial[2] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[3] = phi_initial[3] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[4] = phi_initial[4] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[5] = phi_initial[5] + 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
    #                  np.random.normal(loc=0, scale=1, size=1)[0]
    #
    #
    phi_initial[0] = phi_initial[0] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]
    phi_initial[1] = phi_initial[1] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]

    phi_initial[2] = phi_initial[2] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]
    # phi_initial[1] =  phi_initial[1] + np.sqrt(0.5) * \
    #                   (np.random.normal(loc=0, scale=np.sqrt(0.5), size=1)[0] + 1j* np.random.normal(loc=0, scale=np.sqrt(0.5), size=1)[0])
    #
    # phi_initial[2] = phi_initial[2] +  np.sqrt(0.5) * \
    #                  (np.random.normal(loc=0, scale=np.sqrt(0.5), size=1)[0] + 1j *
    #                   np.random.normal(loc=0, scale=np.sqrt(0.5), size=1)[0])
    phi_initial[3] = phi_initial[3] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]
    phi_initial[4] = phi_initial[4] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]
    phi_initial[5] = phi_initial[5] + np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
                     np.random.normal(loc=0, scale=1, size=1)[0]



    phi0_vec[realiz_index] = phi_initial[0]
    phi1_vec[realiz_index] = phi_initial[1]
    phi2_vec[realiz_index] = phi_initial[2]
    phi3_vec[realiz_index] = phi_initial[3]
    phi4_vec[realiz_index] = phi_initial[4]
    phi5_vec[realiz_index] = phi_initial[5]


rho0_vec = np.abs(phi0_vec) ** 2
rho1_vec = np.abs(phi1_vec) ** 2
rho2_vec = np.abs(phi2_vec) ** 2
rho3_vec = np.abs(phi3_vec) ** 2
rho4_vec = np.abs(phi4_vec) ** 2
rho5_vec = np.abs(phi5_vec) ** 2


rho0_mean = np.mean(rho0_vec, axis=0)
rho1_mean = np.mean(rho1_vec, axis=0)
rho2_mean = np.mean(rho2_vec, axis=0)
rho3_mean = np.mean(rho3_vec, axis=0)
rho4_mean = np.mean(rho4_vec, axis=0)
rho5_mean = np.mean(rho5_vec, axis=0)

#we calculate the variance of the (-k,-1) mode population
lhs = np.mean(rho2_vec**2 +1/4 - rho2_vec)
rhs = np.var(rho2_vec - 1/2)
middle = np.mean( (rho2_vec - 1/2)**2 ) - np.mean(rho2_vec-1/2)**2
x = np.mean(rho2_vec) #HÄÄÄÄ? = 0.05????


J_z_vec = 1 / 2 * (rho1_vec - rho2_vec)
J_z_var = (np.var(J_z_vec))

#y = np.var(rho1_vec) = 0.95
#1 - 1/4  -> physikalisch 3/4
#z = np.var(phi1_vec) = 1 ->physikalisch 1

# the (-k,-1) occupation i.e. rho2_mean is used to define the number of pairs in the chi_+ channel
exp_pair_number = rho2_mean
J_z_var_coh = exp_pair_number / 2

xi_N_squared = 4 * J_z_var / N_temp
xi_N_squared_coh = 4 * J_z_var_coh / N_temp
number_squeezing = xi_N_squared / xi_N_squared_coh

print('hi')
#
# J_z_vec =  1/2* (np.conj(phi1_vec)*phi1_vec - np.conj(phi2_vec)*phi2_vec)
# N_p = np.mean(np.conj(phi2_vec)*phi2_vec)
#
# print('J_z_var', np.var(J_z_vec))
# print('N_p', N_p)
# print('J_z_var_coh', N_p / 2)
#
# rns= 4 * np.var(J_z_vec) / N_p
#
# print('rns', rns)
# print(np.mean(J_z_vec))
#
#
# print(np.var(J_z_vec)) # 1/8
# print(np.mean(np.conj(phi1_vec)*phi1_vec))
# print(np.mean(np.conj(phi2_vec)*phi2_vec))
# print(np.mean(np.conj(phi0_vec)*phi0_vec))
# print(np.var(phi0_vec))
# print(np.var(np.conj(phi1_vec)*phi1_vec))
# print(np.var(np.conj(phi2_vec)*phi2_vec*np.conj(phi1_vec)*phi1_vec))
