import numpy as np
import matplotlib.pyplot as plt

# defining the N, beginn and ending and value at -1 
t0, t_end, y0 = -1, 0, 1/101
N = np.array([25, 50, 100, 200, 400, 800, 1600])

# definiing the ivp
def func_f(t, y):
    return -200 * t * y**2

# building explicit euler new in the way  of the model code 
def explicit_euler(f, t0, T, y0, h, dtype):
    N_steps = int(round(abs(T - t0) / h))
    t_values = np.linspace(t0, T, N_steps + 1, dtype=dtype)
    y_values = np.zeros(N_steps + 1, dtype=dtype)
    y_values[0] = dtype(y0)
    h = dtype(h)
    for n in range(N_steps):
        y_values[n + 1] = y_values[n] + h * f(t_values[n], y_values[n])
    return t_values, y_values

# improved eulers method 
def improved_euler(f, t0, T, y0, h, dtype):
    N_steps = int(round(abs(T - t0) / h))
    t_values = np.linspace(t0, T, N_steps + 1, dtype=dtype)
    y_values = np.zeros(N_steps + 1, dtype=dtype)
    y_values[0] = dtype(y0)
    h = dtype(h)
    for n in range(N_steps):
        k1 = f(t_values[n], y_values[n])
        k2 = f(t_values[n] + 0.5 * h, y_values[n] + 0.5 * h * k1)
        y_values[n + 1] = y_values[n] + h * k2
    return t_values, y_values

# Analytical solution which was given
def y_true(t):
    return 1 / (100 * t**2 + 1)  # remember y(-1)=1/101 defined in the beginning 

# defining the methods to call them in a for loop
methods = {'Explizit Euler': explicit_euler, 'Improved Euler': improved_euler}

for method_name, method in methods.items():
    print(f"\n~~~ {method_name} ~~~")
    errors32, errors64, hs = [], [], []

    for n in N:
        h = 1 / n
        hs.append(h)

        # float32
        t32, y32 = method(func_f, t0, t_end, y0, h, np.float32)
        err32 = abs(float(y32[-1]) - y_true(t_end))
        errors32.append(err32)

        # float64
        t64, y64 = method(func_f, t0, t_end, y0, h, np.float64)
        err64 = abs(float(y64[-1]) - y_true(t_end))
        errors64.append(err64)

        #getting a table that is formatted nicely 

        print(f"N={n:5d} | h={h:.5e} | Y_N32={float(y32[-1]):.8f} | err32={err32:.3e} | "
              f"Y_N64={float(y64[-1]):.8f} | err64={err64:.3e}")

    # plotting log-log plot
    plt.figure()
    plt.loglog(hs, errors32, 'o-', label='float32')
    plt.loglog(hs, errors64, 's-', label='float64')
    plt.xlabel("h")
    plt.ylabel("Absolute Error |Y_N - y(T)|")
    plt.title(f"Error vs. Step Size (log–log) – {method_name}")
    plt.legend()
    plt.grid(True)
    plt.show()


#results show that improved euler is more precise that explizit euler, specially for smaller stepsize 
#float32 has larger errors that float64 but that gets less important with bigger h, but you can see a difference in 
#the plot of improve euler 