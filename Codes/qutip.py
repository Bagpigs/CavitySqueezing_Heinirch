import numpy as np
from qutip import *

N = 101

psi_p = fock(N, 50)#int(N/2))
psi_0 = fock(N, 0)
psi_m = fock(N, 50)#int(N/2))
initial_psi1 = tensor(psi_p,psi_0,psi_m)


psi_p = fock(N, 49)#int(N/2))
psi_0 = fock(N, 0)
psi_m = fock(N, 51)#int(N/2))
initial_psi2 =  tensor(psi_p,psi_0,psi_m)

psi_p = fock(N, 51)#int(N/2))
psi_0 = fock(N, 0)
psi_m = fock(N, 49)#int(N/2))
initial_psi3 = -tensor(psi_p,psi_0,psi_m)

initial_psi = initial_psi2+initial_psi3#initial_psi1 +initial_psi2 + initial_psi3



a_p = tensor(destroy(N), identity(N), identity(N))
a_0 = tensor(identity(N), destroy(N), identity(N))
a_m = tensor(identity(N), identity(N), destroy(N))



S_x = 1 / np.sqrt(2) * (a_p.dag() * a_0 + a_0.dag() * a_m + a_0.dag() * a_p + a_m.dag() * a_0)
S_z = a_p.dag()*a_p - a_m.dag() * a_m
Q_yz = 1 / np.sqrt(2) * (- 1j * a_p.dag() * a_0 + 1j * a_0.dag() * (a_p + a_m) - 1j * a_m.dag() * a_0 )

CommSQ = commutator(S_x, Q_yz)

print(expect(CommSQ, initial_psi))
print(expect(S_x, initial_psi))
print(expect(S_z, initial_psi))

#heisenberg
print('var s_x',variance(S_x, initial_psi))
print('var mix',variance(S_x/np.sqrt(2) + Q_yz/np.sqrt(2), initial_psi))
print('var q_yz',variance(Q_yz, initial_psi))
print('limit to be squeezed', 1/2 * np.abs(expect(CommSQ,initial_psi)))
