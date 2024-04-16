import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, ode, odeint

#import tensorflow
#import tensorflow_probability
#import keras



def eoms(t,y):
    dphidt = np.zeros(2,dtype=np.csingle)
    dphidt[0] = -5 * y[1]
    dphidt[1] = 30j* y[0]

    return dphidt



phi_initial= [1,1]
time = np.linspace(0,10,1000)

r = ode(eoms).set_integrator('zvode', method='BDF')
r.set_initial_value(phi_initial, time[0])


ys0 = [phi_initial[0]]
ys1 = [phi_initial[1]]
for t in time[1:]:
    y = r.integrate(t)
    ys0.append(y[0])
    ys1.append(y[1])
plt.plot(time,np.array(ys0))
plt.plot(time,np.array(ys1))
plt.plot(time,np.abs(np.array(ys0))**2)
plt.plot(time,np.abs(np.array(ys1))**2)
plt.show()


# dphidt[0] = - 1j * chi * (2 * np.conj(phi[0]) * phi[1]+       \
#                              phi[2] * np.conj(phi[2]) * (phi[0])  \
#             + np.conj(phi[1]) * (phi[0]) * phi[1])                \
#             + gamma * (np.conj(phi[2]) * phi[2] * (phi[0])        \
#                        - np.conj(phi[1]) * phi[1] * (phi[0]))     \
#             - 1j * chiM * (2 * np.conj(phi[0]) * phi[4] * phi[3]  \
#                            + np.conj(phi[3]) * (phi[0]) * phi[3]  \
#                            + phi[4] * (phi[0]) * np.conj(phi[4])) \
#             + gammaM * (np.conj(phi[4]) * phi[4] * (phi[0])       \
#                         - np.conj(phi[3]) * phi[3] * (phi[0]))
#
# dphidt[1] = -1j * omega0 * phi[1] - 1j * chi * (
#             np.conj(phi[0]) * (phi[0]) * phi[1] + np.conj(phi[2]) * (
#                 phi[0]) * (phi[0])) \
#             + gamma * (np.conj(phi[0]) * (phi[0]) * phi[1] + np.conj(
#     phi[2]) * (phi[0]) * (phi[0]))
#
# dphidt[2] = -1j * omega0 * phi[2] - 1j * chi * (
#             np.conj(phi[0]) * (phi[0]) * phi[2] + np.conj(phi[1]) * (
#                 phi[0]) * (phi[0])) \
#             - gamma * (np.conj(phi[0]) * (phi[0]) * phi[2] + np.conj(
#     phi[1]) * (phi[0]) * (phi[0]))
#
# dphidt[3] = -1j * omega0 * phi[3] - 1j * chiM * (
#             np.conj((phi[0])) * (phi[0]) * phi[3] + np.conj(phi[4]) * (
#                 phi[0]) * (phi[0])) \
#             + gammaM * (np.conj((phi[0])) * (phi[0]) * phi[3] + np.conj(
#     phi[4]) * (phi[0]) * (phi[0]))
#
# dphidt[4] = -1j * omega0 * phi[4] - 1j * chiM * (
#             np.conj((phi[0])) * (phi[0]) * phi[4] + np.conj(phi[3]) * (
#                 phi[0]) * (phi[0])) \
#             - gammaM * (np.conj((phi[0])) * (phi[0]) * phi[4] + np.conj(
#     phi[3]) * (phi[0]) * (phi[0]))
