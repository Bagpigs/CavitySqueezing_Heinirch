import numpy as np


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
