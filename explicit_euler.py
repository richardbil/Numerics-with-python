import numpy as np
import matplotlib.pyplot as plt
import math

#defining stepsize, initial value and the area in which we want to approx

h1 = 0.2
h2 = 0.1
ybla = 1
t0 = 0
t_end = 1

#creating a list of nodes which we could do more elegant but are doing as specified 

t = np.arange(0,1, h1)

#defining the function (rearranged)

def func_f(t,y):
    return 1/(1+t)**2 - y/(1+t)

#defining explicit euler by building the tvalues and and empty array for the y values, then 
#ranging over the tvalues and defining the one used in explizit euler as the one before our index right now, so for i =1 t = 0, and so on 
#and then always approximating the next yvalue with explizit euler

def explicit_euler(f, h1):
    t_values = np.arange(t0, t_end+h1, h1)
    y_values = np.zeros(len(t_values))
    y_values[0] = ybla
    
    for i in range(1, len(t_values)):
        t_before = t_values[i-1]
        y_before = y_values[i-1]

        y_values[i] = y_before + h1*f(t_before, y_before)
    return t_values, y_values 

t_values, y_approx = explicit_euler(func_f, h1)
print("t =", t_values)
print("y =", y_approx)

def explicit_euler2(f, h2):
    t_values = np.arange(t0, t_end + h2, h2)  
    y_values = np.zeros(len(t_values))
    y_values[0] = ybla
    
    for i in range(1, len(t_values)):
        t_before = t_values[i-1]
        y_before = y_values[i-1]

        y_values[i] = y_before + h2 * f(t_before, y_before)
    return t_values, y_values

t_values2, y_approx2 = explicit_euler2(func_f, h2)
print("t =", t_values2)
print("y =", y_approx2)

#analytical solution

def analytical(h2):
    t_values = np.arange(t0, t_end + h2, h2)  
    y_values = (np.log(t_values + 1) + 1) / (t_values + 1)
    return t_values, y_values 

t_values_analytical, y_analytical = analytical(h2)
print("t =", t_values_analytical)
print("y =", y_analytical)


#plotting everything 

plt.plot(t_values_analytical, y_analytical, label="Analytical")
plt.plot(t_values, y_approx, label="Euler h={h1}")
plt.plot(t_values2, y_approx2, label="Euler h={h2}")

plt.plot(t_values, abs(y_approx - np.interp(t_values, t_values_analytical, y_analytical)), label="Error h1")
plt.plot(t_values2, abs(y_approx2 - np.interp(t_values2, t_values_analytical, y_analytical)), label="Error h2")

plt.legend()
plt.show()

#obviously the error gets smaller with better stepsize, the results are as expected, a smaller stepsize gives us 
# a better approximation and the error is also smaller at
#t=1
