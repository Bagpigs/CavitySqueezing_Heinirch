# why in matlab script 5?
import numpy as np


N_temp = 80000

phi0_vec = np.zeros(
        10000,
        dtype=np.csingle)  # # mf = 0        0_vec
phi1_vec = np.zeros(
    10000,
    dtype=np.csingle)  # # mf =1 ,+k     1_vec
phi2_vec = np.zeros(
    10000,
    dtype=np.csingle)  # # mf = -1 , -k  M1_vec
phi3_vec = np.zeros(
    10000,
    dtype=np.csingle)  # # mf = -1, +k   M1_M:vec
phi4_vec = np.zeros(
    10000,
    dtype=np.csingle)  # # mf = 1 ,-k    1_M_vec
phi5_vec = np.zeros(
    10000,
    dtype=np.csingle)  # # mf = 0, +-2k  5_vec

for realiz_index in range(0, 10000):
    
    phi_initial = np.zeros(6, dtype=np.csingle)
    phi_initial[0] = np.sqrt(N_temp)  # All atoms in mF = 0
    
    # Sample Quantum 1/2 noise # like it is written here, the variance for each quantity is 1 (0.5 real part variance + 0.5 imag part variance)
    # correct would be to replace np.sqrt(0.5) by 0.5 everywhere in the lines below
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

    phi0_vec[realiz_index] = phi_initial[0]
    phi1_vec[realiz_index] = phi_initial[1]
    phi2_vec[realiz_index] = phi_initial[2]
    phi3_vec[realiz_index] = phi_initial[3]
    phi4_vec[realiz_index] = phi_initial[4]
    phi5_vec[realiz_index] = phi_initial[5]

J_z_vec =  1/2* (np.conj(phi1_vec)*phi1_vec - np.conj(phi2_vec)*phi2_vec)
print(np.mean(J_z_vec))
print(np.var(J_z_vec)) # 1/8
print(np.mean(np.conj(phi1_vec)*phi1_vec))
print(np.mean(np.conj(phi2_vec)*phi2_vec))
print(np.mean(np.conj(phi0_vec)*phi0_vec))
print(np.var(phi0_vec))
print(np.var(np.conj(phi1_vec)*phi1_vec))
print(np.var(np.conj(phi2_vec)*phi2_vec*np.conj(phi1_vec)*phi1_vec))
