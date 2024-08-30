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


newfp = np.memmap(filename, dtype=np.csingle, mode='r', shape=(N_delta_p_points, 6, Npoints, Nrealiz, 1))




tw_matrix = newfp[0]

# Shape of tw_matrix: ( 6,         Npoints,    Nreliz,         length(n_list)
#                     modes       time        realization     seeds
# axis                0           1           2               3
x = np.zeros(1000,dtype=np.csingle)
for j in range(1000):
    x[j] = 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5*  np.random.normal(loc=0, scale=1, size=1)[0]
#np.sqrt(0.5) * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * np.sqrt(0.5) * \
           #          np.random.normal(loc=0, scale=1, size=1)[0]
phi0_vec = tw_matrix[0]
phi1_vec = tw_matrix[1]
initial_phi0_vec = phi0_vec[0,:,0]
initial_phi1_vec = phi1_vec[0,:,0]
print(np.sqrt(np.var(initial_phi0_vec * np.conj(initial_phi0_vec),axis=0)))
print(80000*0.045)
print(2* (0.045 * 80000)**4)
print(np.var(initial_phi0_vec ,axis=0))
print('theo', 0.045 * 1/np.sqrt(2) * 80000)
print(np.mean(x*np.conj(x),axis=0))
print('var',np.var(x,axis=0))
print(np.mean(initial_phi1_vec*np.conj(initial_phi1_vec),axis=0))
