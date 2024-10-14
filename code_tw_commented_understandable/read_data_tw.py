import numpy as np
from matplotlib import pyplot as plt



###   DEFINE INITIAL PARAMETERS ###
### Make sure they are the same as in "create_data_tw.py"!!! ###

N = 80000 # atom number
tbounds = np.array([0, 0.2])  # Evoution time in ms
Npoints = 2000  # time points
time = np.linspace(tbounds[0], tbounds[1], Npoints) # time array

n_list = np.array([0])  # List of average initial pair number from classical seeds
Nrealiz = 800 # realizations of TW simulation

filename = 'tw_2000_npoints_800_nrealiz_max_laser_power.bin'




### READ BIN FILE ###

# Read bin file into a matrix with correct shape
tw_matrix = np.array(np.memmap(filename, dtype=np.csingle, mode='r', shape=(6, Npoints, Nrealiz, len(n_list))))

# Shape of tw_matrix: ( 6,         Npoints,    Nreliz,         length(n_list)
#                     modes       time        realization     seeds
# axis                0           1           2               3




### EVALUATE DATA ###

### RELATIVE NUMBER SQUEEZING ###

rho0_vec = np.abs(tw_matrix[0]) ** 2
rho1_vec = np.abs(tw_matrix[1]) ** 2
rho2_vec = np.abs(tw_matrix[2]) ** 2
rho3_vec = np.abs(tw_matrix[3]) ** 2
rho4_vec = np.abs(tw_matrix[4]) ** 2
rho5_vec = np.abs(tw_matrix[5]) ** 2

rho0_mean = np.mean(rho0_vec, axis=1)
rho1_mean = np.mean(rho1_vec, axis=1)
rho2_mean = np.mean(rho2_vec, axis=1)
rho3_mean = np.mean(rho3_vec, axis=1)
rho4_mean = np.mean(rho4_vec, axis=1)
rho5_mean = np.mean(rho5_vec, axis=1)


J_z_vec = 1 / 2 * (rho1_vec - rho2_vec)
J_z_var = (np.var(J_z_vec, axis=1))

# the (-k,-1) occupation i.e. rho2_mean is used to define the number of pairs in the chi_+ channel
exp_pair_number = rho2_mean
J_z_var_coh = exp_pair_number / 2

xi_N_squared = 4 * J_z_var / N
xi_N_squared_coh = 4 * J_z_var_coh / N
number_squeezing = xi_N_squared / xi_N_squared_coh


### Plot relative number squeezing over time

font = {'size': 15}
plt.rc('font', **font)
plt.plot(time * 1000, number_squeezing, label='$\\delta_+ / 2 \\pi = $ ? MHz')
plt.ylabel('$\\xi_N^2 / \\xi_{N,coh}^2$')
plt.xlabel('$t \\, \\, (\\mu s$)')
plt.yscale('log')
plt.hlines(1, 0, 200, linestyles='dashed', label='SQL', colors='grey')
plt.tight_layout()
plt.legend()
# plt.savefig('foo.svg')
plt.show()






### SPIN-NEMATIC SQUEEZING ###

phi0_vec = tw_matrix[0]
phi1_vec = tw_matrix[1]
phi2_vec = tw_matrix[2]

S_x = 1 / 2 ** 0.5 * (np.conj(phi1_vec) * phi0_vec + np.conj(phi0_vec) * phi2_vec + \
                      np.conj(phi0_vec) * phi1_vec + np.conj(phi2_vec) * phi0_vec)

# Shape of S_x´: (      Npoints,        Nrealiz,            length(n_list)

S_y = 1 / (2**0.5 * 1j) * ( np.conj(phi1_vec) * phi0_vec - np.conj(phi0_vec) * phi1_vec + \
                                np.conj(phi0_vec) * phi2_vec - np.conj(phi2_vec) * phi0_vec )

S_z = np.conj(phi1_vec) * phi1_vec - np.conj(phi2_vec) * phi2_vec

Q_yz = 1 / 2 ** 0.5 * (-1j * np.conj(phi1_vec) * phi0_vec + 1j * np.conj(phi0_vec) * phi2_vec + \
                       1j * np.conj(phi0_vec) * phi1_vec - 1j * np.conj(phi2_vec) * phi0_vec)

comm_S_x_Q_yz = 1j * (np.conj(phi1_vec) * phi1_vec + np.conj(phi1_vec) * phi2_vec - 2 * np.conj(phi0_vec) * phi0_vec + np.conj(phi2_vec) * phi1_vec + np.conj(phi2_vec) * phi2_vec)


comm_S_x_Q_yz_exp_absolute_val = np.abs(np.mean(comm_S_x_Q_yz, axis=1))

# shape comm_S_x_Q_yz_exp_absolute_val:
#  axis     0               1
#           npoints         seeds

heis_limit_for_one_quadrature_var = comm_S_x_Q_yz_exp_absolute_val / 2

Q_yz_var = np.var(Q_yz, axis=1)

theta = np.linspace(0, 2 * np.pi, 300)

quadrature_operator_matrix = np.array([np.cos(theta[i]) * S_x + np.sin(theta[i]) * Q_yz for i in range(len(theta))])

# shape quadrature operator matrix:
#  axis     0               1               2               3
#           theta           npoints         nrealiz         seeds

quadrature_operator_matrix_var = np.var(quadrature_operator_matrix, axis=2)

quadrature_operator_matrix_var_min_theta = np.min(quadrature_operator_matrix_var, axis=0)

# shape quadrature_operator_matrix_var_min_theta:
#  axis     0             1
#           npoints       seeds

wineland_param_min_theta = 4 * N * quadrature_operator_matrix_var_min_theta[0:1000] / comm_S_x_Q_yz_exp_absolute_val[0:1000] ** 2

# shape wineland_param_min_theta:
#  axis     0             1
#           npoints       seeds


###  Plot Wineland parameter over time part A

plt.plot(time[0:1000]*1000, 4 * N * quadrature_operator_matrix_var_min_theta[0:1000]/comm_S_x_Q_yz_exp_absolute_val[0:1000]**2/2,label='$\\delta_+ / 2 \\pi = $ ? MHz')
plt.hlines(1,0,100,linestyles='dashed', colors='grey', label='SQL')
plt.ylabel('min$_\\theta \\, \\, \\xi^2(\\theta)$')
plt.xlabel('$t \\, \\, (\\mu s$)')
plt.yscale('log')
plt.tight_layout()
plt.legend()
# plt.savefig('wineland_foo.svg')
plt.show()
