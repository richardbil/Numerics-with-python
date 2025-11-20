import math
import numpy as np 
import matplotlib.pyplot as plt 

#def ivp

def func_f(t,y):
    return -200*t*y**2

#def exakt solution

def analytical(t):
    return 1/(100 * t**2+1)  

Ns = [25, 50, 100, 200, 400, 800, 1600]
errors = []

for N in Ns:
    # interval and step size
    t0, tN = -1, 0
    h = (tN - t0) / N
    t = np.linspace(t0, tN, N+1)

    # solution array
    Y = np.zeros(N+1)
    Y[0] = 1/101  # initial condition

    # explicit Euler for Y1
    Y[1] = Y[0] + h * func_f(t[0], Y[0])

    # Adams Bashforth m2
    for k in range(1, N):
        Y[k+1] = Y[k] + h * (3/2*func_f(t[k], Y[k]) - 1/2*func_f(t[k-1], Y[k-1]))

    # compute max error
    exact = analytical(t)
    EN = np.max(np.abs(Y - exact))
    errors.append(EN)

    print("N =", N, " max error =", EN)

print("\nError ratios (EN / E_prev):")
for i in range(1, len(errors)):
    print(Ns[i], "ratio =", errors[i] / errors[i-1])

#this shows it exhibits second order convergence since, En/e_prev /rightarrow 1/4