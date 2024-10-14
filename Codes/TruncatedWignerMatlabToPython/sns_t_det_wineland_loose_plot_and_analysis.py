import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#     filename = 'tw_30_1000_045.bin'

def power_func(y, a, r): #this one worked for the analogy in rns, but here it simply doesnt.
    return a * y ** r

def laurent_polynomial(y, *coefficients):
    return sum(c * y**(-i) for i, c in enumerate(coefficients))

def power_func_with_const(y, a, r,c):
    return a * y ** r + c

def log_decay(y, a, c): # if I use log_decay + "easy data", i have to iteratively apply five logs to get such a slow grow
    return  a* np.log(y) + c

def random_guess(y,a,b,c): # for "easy data" I get a good approx with a = -74, b=-1 , c = 13
    return a * y ** b + c

delta_p = np.genfromtxt("sns_detuning_vec.csv", delimiter=",")
time_wineland_loose = np.genfromtxt("sns_time_loose_wineland_vec.csv", delimiter=",")



## fit data

# Power func
# initial_guess = [0,0]
# params, params_covariance = curve_fit(power_func, -delta_p, time_wineland_loose, p0=initial_guess)
# print(params)
#
# power_fit_1 = power_func(-delta_p, *params)
#
# mse1 = np.mean((time_wineland_loose- power_fit_1) **2)
# print('mse power', mse1)
#
# plt.plot(delta_p/2/np.pi/10**6,power_fit_1*1000, label='power fit')




font = {'size': 16}
plt.rc('font', **font)

plt.plot(delta_p/2/np.pi/10**6,time_wineland_loose*1000)
# plt.xscale('log')
# plt.yscale('log')
plt.ylabel('t$_L^W(\\mu s)$')
plt.xlabel('$\\delta_+/2\\pi \\,\\,$(MHz)')
plt.xlim((delta_p/2/np.pi/10**6)[0],(delta_p/2/np.pi/10**6)[-1])
plt.tight_layout()

plt.legend()
plt.savefig('plots/sns_t_det_wineland_loose.svg')
plt.show()
