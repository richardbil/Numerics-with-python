import numpy as np
import matplotlib.pyplot as plt

#define function 
def func1(t,y):
    return -1*(y-np.exp(-t))-np.exp(-t)

def func1000(t,y):
    return -1000*(y-np.exp(-t))-np.exp(-t)

#define exakt solution

def anylytical(t):
    return np.exp(-t)

#define grid 

h = 0.01
N = int(1 / h)

t = np.linspace(0, N*h, N+1)

y0 = 1;

#we dont use the methods how we defined them before, i will make 
#all of them new, in order to be able to use them from now on all in the same fashion:
# variables, function, etc defined outside, with the same variable names and same looking output


def explicit_euler():

    # nutzt: f, t, y0, h (alles muss davor definiert sein)
    y_values_ex_euler = np.zeros(len(t))
    y_values_ex_euler[0] = y0

    for i in range(1, len(t)):
        t_prev = t[i - 1]
        y_prev = y_values_ex_euler[i - 1]

        y_values_ex_euler[i] = y_prev + h * f(t_prev, y_prev)

    return y_values_ex_euler

def improved_euler(): #heun

    #nutzt f, t, y0, h (alles muss davor definitert werden)

    y_values_imp_euler = np.zeros(len(t))
    y_values_imp_euler[0] = y0

    for n in range(1, len(t)):
        t_prev = t[n - 1]
        y_prev = y_values_imp_euler[n - 1]

        k1 = f(t_prev, y_prev)
        k2 = f(t_prev + h, y_prev + h * k1)
        y_values_imp_euler[n] = y_prev + 0.5 * h * (k1 + k2)

    return y_values_imp_euler

def runge_kutta_4():

    # nutzt: f, t, y0, h (alles muss davor definiert sein)

    y_values_rk4 = np.zeros(len(t))
    y_values_rk4[0] = y0

    for n in range(1, len(t)):
        t_prev = t[n - 1]
        y_prev = y_values_rk4[n - 1]

        k1 = f(t_prev, y_prev)
        k2 = f(t_prev + 0.5*h, y_prev + 0.5*h*k1)
        k3 = f(t_prev + 0.5*h, y_prev + 0.5*h*k2)
        k4 = f(t_prev + h, y_prev + h*k3)

        y_values_rk4[n] = y_prev + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return y_values_rk4

def adam_bashford_2():
    y_values_ab2 = np.zeros(len(t))
    y_values_ab2[0] = y0

    # Schritt 1: y1 mit explizit Euler
    y_values_ab2[1] = y_values_ab2[0] + h * f(t[0], y_values_ab2[0])

    # ab Schritt 2: Adams-Bashforth 2
    for k in range(1, len(t)-1):
        y_values_ab2[k+1] = y_values_ab2[k] + h * (
            3/2*f(t[k], y_values_ab2[k]) - 1/2*f(t[k-1], y_values_ab2[k-1])
        )

    return y_values_ab2



functions = [
    ("lambda = 1", func1),
    ("lambda = 1000", func1000)
]
print(f"{'Function':<15}{'Explicit':>12}{'Improved':>12}{'RK4':>12}{'AB2':>12}")

for label, func in functions:
    f = func  # setzt die globale f, wie von deinen Methoden erwartet
    
    # Berechne Lösungen
    y_explicit = explicit_euler()
    y_improved = improved_euler()
    y_rk4 = runge_kutta_4()
    y_ab2 = adam_bashford_2()
    
    # Analytische Lösung
    y_exact = anylytical(t)
    
    # Maximaler Fehler
    Emax_explicit = np.max(np.abs(y_explicit - y_exact))
    Emax_improved = np.max(np.abs(y_improved - y_exact))
    Emax_rk4 = np.max(np.abs(y_rk4 - y_exact))
    Emax_ab2 = np.max(np.abs(y_ab2 - y_exact))
    
    # Tabelle ausgeben
    print(f"{label:<15}{Emax_explicit:12.3e}{Emax_improved:12.3e}{Emax_rk4:12.3e}{Emax_ab2:12.3e}")


    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(t, y_exact, 'k-', label="Exact solution")
    plt.plot(t, y_explicit, 'r--', label="Explicit Euler")
    plt.plot(t, y_improved, 'b-.', label="Improved Euler")
    plt.plot(t, y_rk4, 'g:', label="RK4")
    plt.plot(t, y_ab2, 'm--', label="Adams-Bashforth 2")
    
    plt.title(f"Numerical solutions for {label}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)
    
    # Optional: adjust scaling for stiff problem
    if label == "lambda = 1000":
        plt.ylim([-0.5, 1.5])  # choose appropriate vertical limits for stability
    
    plt.show()

# all the methods seem to work okay for lampda=1, none of them seem to work in a great way for 
#lambda=1000

