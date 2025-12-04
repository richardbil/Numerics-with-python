import numpy as np 
import matplotlib.pyplot as plt 


def fixed_rk4(f, y0, I, h):
    
    # number of evaluations of f
    eval = 0

    # implement Butcher-Table
    A = np.zeros([4, 4])
    A[1][0] = 1/2
    A[2][1] = 1/2
    A[3][2] = 1

    c = np.array([0, 1/2, 1/2, 1])
    weights = np.array([(1/6), (1/3), (1/3), (1/6)])

    # prepare RK_K (K_i for Runge Kutta)
    if(y0 is float):
        RK_K = np.zeros(4)
    else:
        RK_K = np.zeros((4, np.size(y0)))
    
    # prepare outputs
    if(y0 is float):  
        Y = np.zeros(int((I[1]-I[0])/h))
    else:
        Y = np.zeros([int((I[1]-I[0])/h),np.size(y0)])

    t = np.arange(I[0], I[1], h)

    # Set initial value
    Y[0] = y0

    # Calculating the Y_k
    for k in range(int((I[1]-I[0])/h) - 1):

        phi = 0
        for i in range(4):

            temp = Y[k]
            for j in range(i):
                temp = temp + h * A[i][j] * RK_K[j]
            RK_K[i] = f(t[k] + h * c[i], temp)
            phi += weights[i] * RK_K[i]

        Y[k+1] = Y[k] + h * phi

        eval += 25

    return t, Y, eval



def solve_adaptive_rk43_procedure(f, y0, I, h0, epsilon, q):

    # number of evaluations of f
    eval = 0

    # implement Butcher-Table

    # normal A
    A = np.zeros([5, 5])
    A[1][0] = 1/2
    A[2][1] = 1/2
    A[3][2] = 1

    # embedded A
    A[4][0] = 1/6
    A[4][1] = 1/3
    A[4][2] = 1/3
    A[4][3] = 1/6

    # c and weights
    c = np.array([0, 1/2, 1/2, 1, 1])
    weights = np.array([(1/6), (1/3), (1/3), (1/6)])
    weights_embedded = np.array([(1/6), (1/3), (1/3), 0, (1/6)])

    # implement RK_K (K_i for Runge Kutta)
    if(y0 is float):
        RK_K = np.zeros(5)
    else:
        RK_K = np.zeros((5, np.size(y0)))

    # prepare t_k, Y_k

    times = [I[0]]
    values = [y0]
    h = h0

    # values for algorithm
    y = y0
    t = I[0]

    # algorithm of Ex5
    while True:

        # apply Runge-Kutta with stepsize h
        phi = 0
        phi_tilde = 0

        y_prev = y

        for i in range(5):
            temp = y
            for j in range(i):
                temp = temp + h * A[i][j] * RK_K[j]
            RK_K[i] = f(t + h * c[i], temp)
            if (i < 4):
                phi += weights[i] * RK_K[i]
            phi_tilde += weights_embedded[i] * RK_K[i]

        eval += 25

        y = y + h * phi
        y_tilde = y + h * phi_tilde

        # calculate scaling factor
        max_norm_delta_y = h * np.max(np.abs(phi - phi_tilde))

        if (max_norm_delta_y > 0):
            s = (h * epsilon / max_norm_delta_y)**(1/q)
        else:
            s = 2

        # evaluating step_sizes
        if (s >= 1):
            t = t + h
            h = np.min([2,s]) * h

            # return t_k, Y_k and evaluations of f.
            if(t + h > 20):
                return times, values, eval

            times.append(t)
            values.append(y)

        else:
            y = y_prev
            h = np.max([1/2, s]) * h

        
        if(h == 0):
            print("Somhow h got rounded down to 0!")
            return



def main():

    # van der Pol system
    f = lambda t, y : np.array([y[1], 10*(1 - y[0]**2)*y[1] - y[0]])

    # inputs for algorithm above
    y0 = np.array([0,1])
    I = np.array([0,20])
    h0 = 10**(-4)
    epsilon = 10**(-4)

    if (h0 == 0): return

    # calculating van der Pol with adaptive step sizes
    times, values, eval = solve_adaptive_rk43_procedure(f, y0, I, h0, epsilon, 3)
    if(type(times) == None):
        print("stepsize got rounded to 0!")
        return

    # calculating van der Pol with fixed step sizes
    times_fix, values_fix, eval_fix = fixed_rk4(f, y0, I, 0.0625)
    times_ref, values_ref, eval_ref = fixed_rk4(f, y0, I, 2**(-14))

    # errors
    err = 0
    err_fix = 0

    # calculating error of evaluations
    for k in range(1, int(np.size(values_fix) / np.size(y0))):
        var = np.max(np.abs(values_fix[k] - values_ref[ k * int( np.round( ( 0.0625 / (2**(-14)) ) , decimals=0) ) ] )) 
        if(var > err_fix):
            err_fix = var
    
    for k in range(1, int(np.size(values) / np.size(y0))):

        # linear interpolation the dynamic step sizes, to compare with y_ref(t_k)
        k_1 = int(np.floor(2**(14) * (times[k] - I[0])))
        k_2 = int(np.ceil(2**(14) * (times[k] - I[0])))
        convex = k_2 - 2**(14) * (times[k] - I[0])

        var = np.max( np.abs( values[k] - ( convex * values_ref[k_1] + (1 - convex) * values_ref[k_2] ) ) ) 
        if(var > err):
            err = var

    print("Error of adaptive Method:", err, "| Numer of evaluations of f:", eval)
    print("Error of fixed Method:", err_fix, "| Numer of evaluations of f:", eval_fix)
    print("Number of evaluations of f for reference:", eval_ref)

    # prepare values for plots

    Y_1 = [y[0] for y in values]
    Y_2 = [y[1] for y in values]

    Y_fix_1 = [y[0] for y in values_fix]
    Y_fix_2 = [y[1] for y in values_fix]

    Y_ref_1 = [y[0] for y in values_ref]
    Y_ref_2 = [y[1] for y in values_ref]

    # plotting those values

    fig = plt.figure(figsize=(10, 6))

    plt.plot(times_ref, Y_ref_1)
    plt.plot(times, Y_1)
    plt.plot(times_fix, Y_fix_1)

    plt.ylim(-2.1, 2.1)
    plt.legend(["reference", "adaptive h", "fixed h = 0.0625"])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("y(t)", fontsize=16)

    plt.title("Van de Pol Evaluations for y1", fontsize=17)
    fig.show()

    fig2 = plt.figure(figsize=(10, 6))

    plt.plot(times_ref, Y_ref_2)
    plt.plot(times, Y_2)
    plt.plot(times_fix, Y_fix_2)

    plt.legend(["reference", "adaptive h", "fixed h = 0.0625"])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("y(t)", fontsize=16)

    plt.title("Van de Pol Evaluations for y2", fontsize=17)
    fig2.show()

    input()



# main
if __name__ == "__main__":
    main()
