from cProfile import label

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def power_func(y, a, r):
    return a * y ** r

def laurent_polynomial(y, *coefficients):
    return sum(c * y**(-i) for i, c in enumerate(coefficients))

def power_func_with_const(y, a, r,c):
    return a * y ** r + c

def log_decay(y, a, c): # if I use log_decay + "easy data", i have to iteratively apply five logs to get such a slow grow
    return  a* np.log(y) + c

def random_guess(y,a,b,c): # for "easy data" I get a good approx with a = -74, b=-1 , c = 13
    return a * y ** b + c

x = np.genfromtxt("sns_detuning_vec.csv", delimiter=",")
y = np.genfromtxt("sns_squeezing_vec.csv", delimiter=",")


# Fit the model to the data

## Power func
# initial_guess = [160,-0.4]
# params, params_covariance = curve_fit(power_func, -x, y, p0=initial_guess)
# print(params)
#
# power_fit_1 = power_func(-x, *params)
#
# mse1 = np.mean((y- power_fit_1) **2)
# print('mse power', mse1)
#
#

## Laurent polynomial
# initial_guess = [0,0] # I tried to use up to [0,0,0,0] but it the fit of "min winelande over det" didnt get better
#
# # Fit the model to the data
# params, params_covariance = curve_fit(laurent_polynomial, x, y, p0=initial_guess)
# print(params)
#
# laurent_fit_2 = laurent_polynomial(x, *params)
#
# mse2 = np.mean((y- laurent_fit_2) **2)
# print('mse laurent2',mse2)


## Power func with const
# Fit the model to the data
# initial_guess = [5000,0,0]
#
# params, params_covariance = curve_fit(power_func_with_const, -x, y, p0=initial_guess)
# print(params)
#
#
# power_func_with_const_fit = power_func_with_const(-x, *params)
#
# mse3 = np.mean((y- power_func_with_const_fit) **2)
# print('mse power_func_with_const_fit',mse3)

#the following are the explicit values for two different "power_func_with_const" fit. note, that I gave them slightly different inital values
#result, both functions are super close to each other even though their parameters are far aways from each other
# also true: the fit sucks!!
# plt.plot(-x/2/np.pi/10**6, -5.16042557*10**3 * (-x) ** (5.38502869 *10**(-6)) + 5.161029*10**(3))
# plt.plot(-x/2/np.pi/10**6, -3.44435567*10**4 * (-x) ** (8.06873744 *10**(-7)) + 3.44441602*10**(4))

## end of power func with const

## Log decay also didnt work so good
# initial_guess = [-0.02,-0.24, 0.5]
# params, params_covariance = curve_fit(log_decay, x, y, p0=initial_guess)
# print('params log fit: ',params)
#
# log_fit = laurent_polynomial(x, *params)
#
# mse4 = np.mean((y- log_fit) **2)
# print('mse log fit: ',mse4)
# plt.plot(-x/2/np.pi/10**6,log_fit, label='log fit')
#

# Plot the approximations

# plt.plot(-x/2/np.pi/10**6,power_fit_1, label='power fit')
# plt.plot(-x/2/np.pi/10**6,power_func_with_const_fit,label='power_func_with_const_fit')
# plt.plot(-x/2/np.pi/10**6,laurent_fit_2, label='laurent polynomial')


# plt.plot(-x/2/np.pi/10**6,y, label='data')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()


        ## modify data first:
#
# y_easy = - 10 * np.log10(y)
# x_easy = -x/2/np.pi/10**6
#
# ## log fit AGAIN
# initial_guess = [1, 0.5]
# params, params_covariance = curve_fit(log_decay, x_easy, y_easy, p0=initial_guess)
# print('params log fit: ',params)
#
# log_fit = log_decay(x_easy, *params)
#
# mse4 = np.mean((y- log_fit) **2)
# print('mse log fit: ',mse4)
# plt.plot(x_easy,log_fit, label='log fit')
#
#
#
#
# plt.plot(x_easy, y_easy, label='data')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.legend()
# plt.show()

        ## end modify data first


