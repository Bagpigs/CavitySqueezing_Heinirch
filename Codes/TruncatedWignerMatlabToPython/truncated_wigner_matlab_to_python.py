import numpy as np

dphidt = np.zeros(6, dtype=np.csingle)
phi = np.zeros(6, dtype=np.csingle) # maybe wrong
print(repr(dphidt))
gamma = 1
gammaM = 1
chi = 1
chiM = 1
omega0 = 1
omegaRec = 1

#things still unclear
scalecoupling_k2 = 1
a = 2/3

#why rescaling?
a = (scalecoupling_k2)**2 * a

#phi convention matlab:
#phi 1 % All atoms in mF=0
#phi 2 % Atoms in mF=1,+k
#phi 3 % Atoms in mF=-1,-k
#phi 4 % Atoms in mF=-1,+k
#phi 5 % Atoms in mF=1,-k
#phi 6 % Atoms in mF=0,+-2k_x

#convention python
#phi 0 % All atoms in mF=0
#phi 1 % Atoms in mF=1,+k
#phi 2 % Atoms in mF=-1,-k
#phi 3 % Atoms in mF=-1,+k
#phi 4 % Atoms in mF=1,-k
#phi 5 % Atoms in mF=0,+-2k_x


dphidt[0] = - 1j * chi * (2 * np.conj(phi[0]+np.sqrt(a)*phi[5]) * phi[1] + phi[2]       \
                          + np.conj(phi[2])* (phi[0]+np.sqrt(a)*phi[5])                 \
                          + np.conj(phi[1]) * (phi[0] + np.sqrt(a)*phi[5]) * phi[1])    \
            + gamma * (np.conj(phi[2])* phi[2] * (phi[0] + np.sqrt(a) * phi[5])         \
                       - np.conj(phi[1]) * phi[1] * np.sqrt(a) * phi[5])                \
            - 1j * chiM * (2 * np.conj(phi[0]+ np.sqrt(a)* phi[5])* phi[4]*phi[3]       \
                           +np.conj(phi[3]) * (phi[0] + np.sqrt(a)*phi[5]) * phi[3]     \
                           + phi[4]* (phi[0]* np.sqrt(a)* phi[5]) * np.conj(phi[4]))    \
            + gammaM * (np.conj(phi[4])*phi[4]*(phi[0]+ np.sqrt(a)*phi[5])              \
                        - np.conj(phi[3]) * phi[3] * (phi[0]+np.sqrt(a) * phi[5]))

dphidt[1] = -1j*omega0*phi[1] -1j*chi*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[1]+np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
                              + gamma*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[1]+np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

dphidt[2] = -1j*omega0*phi[2] -1j*chi*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]+np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
                              - gamma*(np.conj(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]+np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

dphidt[3] = -1j*omega0*phi[3] -1j*chiM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[3]+np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
                              + gammaM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[3]+np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

dphidt[4] = -1j*omega0*phi[4] -1j*chiM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[4]+np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5])) \
                              - gammaM*(np.conj((phi[0]+np.sqrt(a)*phi[5]))*(phi[0]+np.sqrt(a)*phi[5])*phi[4]+np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*(phi[0]+np.sqrt(a)*phi[5]))

dphidt[5] = -1j*4*omegaRec*phi[5] \
                    -1j*np.sqrt(a)*chi*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[1]*phi[2]+ np.conj(phi[1])*(phi[0]+np.sqrt(a)*phi[5])*phi[1] + np.conj(phi[2])*(phi[0]+np.sqrt(a)*phi[5])*phi[2]) \
                    + gamma*np.sqrt(a)*(np.conj(phi[2])*phi[2]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[1])*phi[1]*(phi[0]+np.sqrt(a)*phi[5])) \
                    -1j*np.sqrt(a)*chiM*(2*np.conj((phi[0]+np.sqrt(a)*phi[5]))*phi[3]*phi[4]+ np.conj(phi[3])*(phi[0]+np.sqrt(a)*phi[5])*phi[3] + np.conj(phi[4])*(phi[0]+np.sqrt(a)*phi[5])*phi[4]) \
                    + gammaM*np.sqrt(a)*(np.conj(phi[4])*phi[4]*(phi[0]+np.sqrt(a)*phi[5])-np.conj(phi[3])*phi[3]*(phi[0]+np.sqrt(a)*phi[5])) 

print(repr(dphidt))