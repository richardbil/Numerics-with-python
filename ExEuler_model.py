#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:15:30 2025

@author: adm_andrealez95
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def f(t, y):
    return -y/(1+t) + 1/((1+t)**2)

def euler_explicit(f, t0, T, y0, h):
    N = int(round((T - t0)/h))
    t = np.linspace(t0, T, N+1)
    y = np.empty(N+1); y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(t[n], y[n])
    return t, y

def y_exact(t):
    return (np.log(1+t) + 1) / (1+t)

# (a) Numerical solutions for both step sizes
t1, y1 = euler_explicit(f, 0.0, 1.0, 1.0, 0.2)
t2, y2 = euler_explicit(f, 0.0, 1.0, 1.0, 0.1)

# (b) Computation of the error
y_ex_1 = y_exact(1.0)                # ≈ 0.84657359028
err_h1 = abs(y_ex_1 - y1[-1])        # ≈ 0.0121124767
err_h2 = abs(y_ex_1 - y2[-1])        # ≈ 0.0054113588
print("y_exact(1) =", y_ex_1)
print("Euler h=0.2 @1 =", y1[-1], " | Fehler:", err_h1)
print("Euler h=0.1 @1 =", y2[-1], " | Fehler:", err_h2)


t_fine = np.linspace(0,1,400)
y_fine = y_exact(t_fine)

plt.plot(t1, y1, 'o-', label='Euler h=0.2')
plt.plot(t2, y2, 'o-', label='Euler h=0.1')
plt.plot(t_fine, y_fine, label='analytic')
plt.legend(); plt.xlabel('t'); plt.ylabel('y(t)'); plt.tight_layout()
plt.show()
