import numpy as np
from matplotlib import pyplot as plt
#import tensorflow as tf
#import tensorflow_probability as tfp
from scipy.integrate import solve_ivp, ode, odeint
import time as tm
print('input check')
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp
#np.random.seed(0)


hbar = 1.054571628*(10**(-34))
mRb87 = 1.443160648*(10**(-25))
epsilon0 = 8.8541878128e-12
#?
lam_M=790.02e-9 #doesnt matter whether it is k in x or z direction, see notes.
E0=403 #V/m electric field per cavity photon;
c=2.9979e8
omegaRec = hbar * 2 * np.pi * np.pi / (mRb87 * lam_M * lam_M)
omegaR = omegaRec / 1000 #everythig in kHz and ms
p = 0.7024*1e6 # First order splitting of Rb87 in Hz
q=144  #2md second order splitting of Rb87 in Hz

N = 80000 # atom number
DeltaN = 0.00*N # Fluctuations of Atom Number
tbounds = np.array([0,0.19])#np.array([0,0.2]) # Evoution time in ms
eta = 2*np.pi*1.7e3 # Raman coupling

Npoints = 2000
time = np.linspace(tbounds[0],tbounds[1], Npoints)
Kappa = 2*np.pi*1.25e6 #Cavity losses in Hz
omegaZ = 2*np.pi * 1.09e6 #Zeemansplitting in Hz
deltaC = -2*np.pi *25.8e6 # Cavity detuning in Hz
print(omegaRec)


# Two Photon Detunings
delta_p = (deltaC+omegaZ) # Two Photon Detunings for + channel
delta_m = (deltaC - omegaZ) # Two Photon Detungs for - channel
#what about points and where does fromula come from?
omega0=0.5*(4*omegaRec+2*np.pi*q*(omegaZ/2/np.pi/p)**2)/1000

# Calculate Two Photon And Four Photon Couplings
#what about points and where does fromula come from?
x_p=(eta**2*delta_p/(delta_p**2+(Kappa)**2))/1000 # Pair coupling for chi_+ Channel
Gamma_p=(eta**2*Kappa/(delta_p**2+(Kappa)**2))/1000 # Dissipative coupling for chi_+ Channel
x_m=(eta**2*delta_m/(delta_m**2+(Kappa)**2))/1000 # Pair coupling for chi_- Channel
Gamma_m=(eta**2*Kappa/(delta_m**2+(Kappa)**2))/1000 # Pair coupling for chi_+ Channel



### DO SIMULATIONS ###

# ?
n_list = np.array([0])  # List of average initial pair number from classical seeds
Nrealiz = 1  # Number of TW simulations pro setting (used for averaging)
# ?
scalecoupling_k2 = 1  # Scale Coupling to m=0, k_x=+-2k mode, set to 0 to supress coupling to this mode and 1 to consider it correctily

# EOM
# still to check, whether following constants are defined useful!
chi = x_p
chiM = x_m
gamma = Gamma_p
gammaM = Gamma_m
a = 2 / 3
a = (scalecoupling_k2) ** 2 * a


def eoms(t, phi, args = None):
    dphidt = np.zeros(6, dtype=np.csingle)
    # dphidt[0] = - 1j * chi * (2 * np.conj(phi[0]) * phi[1]+       \
    #                          phi[2] * np.conj(phi[2]) * (phi[0])  \
    #         + np.conj(phi[1]) * (phi[0]) * phi[1])                \
    #         + gamma * (np.conj(phi[2]) * phi[2] * (phi[0])        \
    #                    - np.conj(phi[1]) * phi[1] * (phi[0]))     \
    #         - 1j * chiM * (2 * np.conj(phi[0]) * phi[4] * phi[3]  \
    #                        + np.conj(phi[3]) * (phi[0]) * phi[3]  \
    #                        + phi[4] * (phi[0]) * np.conj(phi[4])) \
    #         + gammaM * (np.conj(phi[4]) * phi[4] * (phi[0])       \
    #                     - np.conj(phi[3]) * phi[3] * (phi[0]))

    # dphidt[1] = -1j*omega0*phi[1] -1j*chi*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[1]+np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
    #                               + gamma*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[1]+np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

    # dphidt[2] = -1j*omega0*phi[2] -1j*chi*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]+np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
    #                               - gamma*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]+np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

    # dphidt[3] = -1j*omega0*phi[3] -1j*chiM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[3]+np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
    #                               + gammaM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[3]+np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

    # dphidt[4] = -1j*omega0*phi[4] -1j*chiM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[4]+np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
    #                               - gammaM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[4]+np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

    # dphidt[5] = -1j*4*phi[5]
    dphidt[5] = -1j * 4 * omegaR * phi[5] \
                            -1j*np.sqrt(a)*chi*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[1]*phi[2]+ np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*phi[1] + np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]) \
                        + gamma*np.sqrt(a)*(np.conj(phi[2])*phi[2]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[1])*phi[1]*(phi[0]+np.sqrt(a)*phi[5])) \
                        -1j*np.sqrt(a)*chiM*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[3]*phi[4]+ np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*phi[3] + np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*phi[4]) \
                        + gammaM*np.sqrt(a)*(np.conj(phi[4])*phi[4]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[3])*phi[3]*(phi[0]+np.sqrt(a)*phi[5]))

    # # dphidt[6] = -1j*np.pi*0.3*phi[6]
    ####THE CURRENT STATE OF ODES #THEY ALIGNED WITH MATLAB
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
    ## END CURRENT STATE

    # dphidt[0] = - 1j * chi * (2 * np.conj(phi[0]) * phi[1]+       \
    #                          phi[2] * np.conj(phi[2]) * (phi[0])  \
    #         + np.conj(phi[1]) * (phi[0]) * phi[1])                \
    #         + gamma * (np.conj(phi[2]) * phi[2] * (phi[0])        \
    #                    - np.conj(phi[1]) * phi[1] * (phi[0]))     \
    #         - 1j * chiM * (2 * np.conj(phi[0]) * phi[4] * phi[3]  \
    #                        + np.conj(phi[3]) * (phi[0]) * phi[3]  \
    #                        + phi[4] * (phi[0]) * np.conj(phi[4])) \
    #         + gammaM * (np.conj(phi[4]) * phi[4] * (phi[0])       \
    #                     - np.conj(phi[3]) * phi[3] * (phi[0]))

    # dphidt[1] = -1j * omega0 * phi[1] - 1j * chi * (            np.conj(phi[0]) * (phi[0]) * phi[1] + np.conj(phi[2]) * (                phi[0]) * (phi[0])) \
    #             + gamma * (np.conj(phi[0]) * (phi[0]) * phi[1] + np.conj(    phi[2]) * (phi[0]) * (phi[0]))

    # dphidt[2] = -1j * omega0 * phi[2] - 1j * chi * (            np.conj(phi[0]) * (phi[0]) * phi[2] + np.conj(phi[1]) * (                phi[0]) * (phi[0])) \
    #             - gamma * (np.conj(phi[0]) * (phi[0]) * phi[2] + np.conj(    phi[1]) * (phi[0]) * (phi[0]))

    # dphidt[3] = -1j * omega0 * phi[3] - 1j * chiM * (            np.conj((phi[0])) * (phi[0]) * phi[3] + np.conj(phi[4]) * (                phi[0]) * (phi[0])) \
    #             + gammaM * (np.conj((phi[0])) * (phi[0]) * phi[3] + np.conj(    phi[4]) * (phi[0]) * (phi[0]))
    # dphidt[4] = -1j * omega0 * phi[4] - 1j * chiM * (            np.conj((phi[0])) * (phi[0]) * phi[4] + np.conj(phi[3]) * (                phi[0]) * (phi[0])) \
    #             - gammaM * (np.conj((phi[0])) * (phi[0]) * phi[4] + np.conj(    phi[3]) * (phi[0]) * (phi[0]))

    return dphidt


#initialize different matrices to carry all information about different simulations and different times
rho0_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(0)
rho1_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(1) +1,+k
rho2_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(2)
rho3_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(3)
rho4_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(4)

#testing
rho5_vec = np.zeros((Npoints,Nrealiz,len(n_list)),dtype=np.csingle) #phiTW(5)

###### #initialize phi (which wont be squared later)
t_tot_1 = tm.time()
# for each classical seed
for seed_index in range(0, len(n_list)):
    # Choose average pair occupation, what is this?
    nSeed = n_list[seed_index]
    for realiz_index in range(0, Nrealiz):
        N_temp = 7900#np.random.normal(loc=N, scale=DeltaN, size=1)[0]  # N mean, DeltaN standard deviation
        # print('hi',repr(N_temp))

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
        # phi_initial[5] = 0 # Atoms in  mF = 0, +-2k_x
        # print(phi_initial)
        # Sample Quantum 1/2 noise
        #constant values
        # phi_initial[0] = phi_initial[0] + 0.15 + 0.1j#0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                  #np.random.normal(loc=0, scale=1, size=1)[0]
        # phi_initial[1] = phi_initial[1] -0.5 -0.9j#0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                  #np.random.normal(loc=0, scale=1, size=1)[0]
        # phi_initial[2] = phi_initial[2] + 0.2 + 0.4j# 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                 # np.random.normal(loc=0, scale=1, size=1)[0]
        # phi_initial[3] = phi_initial[3] -0.03 + 0.4j# 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                  #np.random.normal(loc=0, scale=1, size=1)[0]
        # phi_initial[4] = phi_initial[4] + 0.5 - 0.8j#0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                  #np.random.normal(loc=0, scale=1, size=1)[0]
        # #phi_initial[5] = phi_initial[5] +0.02 + 0.05j# 0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * \
        #                  #np.random.normal(loc=0, scale=1, size=1)[0]
        # phi_initial[5] = 0.3 + 1j * 0.2

        #random values
        phi_initial[0] = phi_initial[0]+  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]
        phi_initial[1] = phi_initial[1]+  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]
        phi_initial[2] = phi_initial[2] +  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]
        phi_initial[3] = phi_initial[3] +  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]
        phi_initial[4] = phi_initial[4] +  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]
        phi_initial[5] = phi_initial[5] +  0.5 * np.random.normal(loc=0, scale=1, size=1)[0] + 1j * 0.5 * np.random.normal(loc=0, scale=1, size=1)[0]

        # test

        # still have to look up this solver and what atol, rtol is
        t_1 = tm.time()
        # print('start solve')


        #scipy solver
        sol = solve_ivp(eoms, [time[0],time[-1]], y0=phi_initial, method='DOP853',atol=1e-8,rtol=1e-8, t_eval=time)#atol=1e-6,rtol=1e-4, t_eval=tbounds)
        t_2 = tm.time()
        # print('stop solve',t_2-t_1)
        #
        rho0_vec[:, realiz_index, seed_index] = sol.y[0]
        rho1_vec[:, realiz_index, seed_index] = sol.y[1]
        rho2_vec[:, realiz_index, seed_index] = sol.y[2]
        rho3_vec[:, realiz_index, seed_index] = sol.y[3]
        rho4_vec[:, realiz_index, seed_index] = sol.y[4]
        rho5_vec[:, realiz_index, seed_index] = sol.y[5]

        # testing
        # plt.plot(time, np.real(sol.y[5]))
        # plt.plot(time, np.abs(sol.y[5]) ** 2)
        # plt.show()

        #tf solver
        # solver = tfp.math.ode.DormandPrince(rtol = 1e-4, atol= 1e-6)
        # sol = solver.solve(ode_fn=eoms, initial_time=time[0], initial_state=phi_initial, solution_times=time)

        #diffrax solver (just for real)
        # term = ODETerm(eoms)
        # solver = Dopri5()
        # # y0 = jnp.array([2., 3.])
        # solution = diffeqsolve(term, solver, t0=time[0], t1=time[-1], dt0=time[1]-time[0], y0=jnp.array(phi_initial))

        #ode scipy solver
        # sol = None
        # r = ode(eoms).set_integrator('zvode', method='bdf')
        # r.set_initial_value(phi_initial, time[0])#.set_f_params(2.0)#.set_jac_params(2.0)
        #
        # ys5 = [phi_initial[5]]
        # ys0 = [phi_initial[0]]
        # for t in time[1:]:
        #     y = r.integrate(t)
        #     #why always showing not successful?
        #     #
        #     if not r.successful():
        #         print('not successful')
        #     ys5.append(y[5])
        #     ys0.append(y[0])
        # t_2 = tm.time()
        # print('stop solve',t_2-t_1)
        # rho5_vec[:, realiz_index, seed_index] = ys5
        # # y_values = sol.states.numpy()
        # # a = np.abs(sol.y[5] ** 2)
        # # plt.plot(time, np.real(rho5_vec[:, 0, 0]))
        # # plt.plot(time, np.imag(rho5_vec[:, 0, 0]))
        # plt.plot(time, np.abs(rho5_vec[:, 0, 0]) ** 2)
        # plt.show()

        #odeint solver
        # sol = odeint(eoms,phi_initial,time)

        # sol = [[] for i in range(6)]
        # for t in time:
        #     for i in range(6):
        #         #still very scetchy how this integrate thing works
        #         sol[i].append(r.integrate(t)[i])
        # print('calculation time', t_2 - t_1)
        # print(np.shape(time))
        # print(np.shape(sol.y))

# print(a[0:10], a[-1])
#N_ges = np.abs(rho0_vec[:,0,0])**2 + np.abs(rho1_vec[:,0,0])**2 +np.abs(rho2_vec[:,0,0])**2 +np.abs(rho3_vec[:,0,0])**2 +np.abs(rho4_vec[:,0,0])**2 +np.abs(rho5_vec[:,0,0])**2
# plt.plot(time,np.real(rho0_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.imag(rho0_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.real(rho1_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.imag(rho1_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.real(rho2_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.imag(rho2_vec[:,0,0]))
# plt.show()
plt.plot(time,np.real(rho3_vec[:,0,0]))
plt.show()
plt.plot(time,np.imag(rho3_vec[:,0,0]))
plt.show()
plt.plot(time,np.real(rho4_vec[:,0,0]))
plt.show()
plt.plot(time,np.imag(rho4_vec[:,0,0]))
plt.show()
# plt.plot(time,np.real(rho5_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.imag(rho5_vec[:,0,0]))
# plt.show()
# plt.plot(time,np.real(rho3_vec[:,0,0]))
# plt.plot(time,np.imag(rho3_vec[:,0,0]))
# plt.plot(time,np.real(rho4_vec[:,0,0]))
# plt.plot(time,np.imag(rho4_vec[:,0,0]))
# plt.plot(time,np.real(rho5_vec[:,0,0]))
# plt.plot(time,np.imag(rho5_vec[:,0,0]))
# plt.plot(time,np.abs(rho1_vec[:,0,0])**2)
# plt.plot(time,np.abs(rho2_vec[:,0,0])**2)
# plt.plot(time,np.abs(rho3_vec[:,0,0])**2)
# plt.plot(time,np.abs(rho4_vec[:,0,0])**2)
# plt.plot(time,np.abs(rho5_vec[:,0,0])**2)
#plt.plot(time,N_ges)
#plt.plot(time,np.abs(rho0_vec[:,1,0])**2)
# plt.plot(time,np.abs(rho0_vec[:,2,0])**2)
plt.show()
t_tot_2 = tm.time()
print('done',t_tot_2 - t_tot_1)