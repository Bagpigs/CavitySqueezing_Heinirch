
import numpy as np
from scipy.integrate import ode

def eoms(t, y):
    return -5 * y

phi_initial = 1
time = np.linspace(0, 10, 1000)
r = ode(eoms).set_integrator('dopri5')
r.set_initial_value(phi_initial, time[0])

# Integrate over the entire time span
ys = []#[phi_initial]
for t in time[1:]:
    y = r.integrate(t)
    print(r.successful())
    ys.append(y[0])
    # else:
    #     print('nope')

ys = np.array(ys)
# Plotting
import matplotlib.pyplot as plt
plt.plot(time, ys)
plt.xlabel('Time')
plt.ylabel('Solution')
#plt.title('Solution of the ODE')
plt.show()