import numpy as np
hbar = 1.054571628*(10**(-34))
mRb87 = 1.443160648*(10**(-25))
epsilon0 = 8.8541878128e-12
#?
lam_M=790.02e-9 #doesnt matter whether it is k in x or z direction, see notes.
E0=403 #V/m electric field per cavity photon;
c=2.9979e8
# why points in original version?omegaRec = hbar*2*pi*pi./(mRb87.*lam_M.*lam_M) where does formula come from?
omegaRec = hbar * 2 * np.pi * np.pi / (mRb87 * lam_M * lam_M)
omegaR = omegaRec / 1000 #everythig in kHz and ms
p = 0.7024*1e6 # First order splitting of Rb87 in Hz
q=144  #2md second order splitting of Rb87 in Hz

N = 80000 # atom number
DeltaN = 0.03*N # Fluctuations of Atom Number
tbounds = np.array([0,0.2]) # Evoution time in ms
eta = 2*np.pi*1.7e3 # Raman coupling

Npoints = 2000
time = np.linspace(tbounds[0],tbounds[1], Npoints)
Kappa = 2*np.pi*1.25e6 #Cavity losses in Hz
omegaZ = 2*np.pi * 1.09e6 #Zeemansplitting in Hz
deltaC = -2*np.pi *25.8e6 # Cavity detuning in Hz

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

#?
n_list = np.array([0]) # List of average initial pair number from classical seeds
Nrealiz = 5 #Number of TW simulations pro setting (used for averaging)
#?
scalecoupling_k2 = 0 #Scale Coupling to m=0, k_x=+-2k mode, set to 0 to supress coupling to this mode and 1 to consider it correctily

#initialize different matrices to carry all information about different simulations and different times
#notation here is not clear/ not consistent with rest of script
rho0_vec = np.zeros((Npoints,Nrealiz,len(n_list))) #phiTW(0)
rho1_vec=np.zeros((Npoints,Nrealiz,len(n_list))) #phiTW(1) +1,+k
rhoM1_vec=np.zeros((Npoints,Nrealiz,len(n_list))) #phiTW
rho1_M_vec=np.zeros((Npoints,Nrealiz,len(n_list))) #phiTW
rhoM1_M_vec=np.zeros((Npoints,Nrealiz,len(n_list))) #phiTW

#initialize phi (which wont be squared later)

for n in range(0, len(n_list)):
    # Choose average pair occupation, what is this?
    nSeed = n_list[n]
    for j in range(0,Nrealiz):
        N_temp = np.random.normal(loc=N, scale=DeltaN, size=1) #N mean, DeltaN standard deviation
        # switch seed type

        #initialize Phi

        #why in matlab script 5?
        phi_initial = np.zeros(6)
        phi_initial[0] = np.sqrt(N_temp) #All atoms in mF = 0



