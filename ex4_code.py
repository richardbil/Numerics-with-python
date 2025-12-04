
# Libaries
import numpy as np
import matplotlib.pyplot as plt

# Explicit Runge Kutta (input: see excercise, output: times and values)
def runge_kutta_ex(f, y0, I, h, weights, A):

    # A is a matrix of size s^2
    size_i, size_j = np.shape(A)

    # Prepare RK_K (K_i for Runge Kutta) and c
    if(y0 is float):
        RK_K = np.zeros(size_j)
    else:
        RK_K = np.zeros((size_j, np.size(y0)))
    c = np.zeros(size_i)

    # Prepare the Y_k and constuct the output times

    if(y0 is float):  
        Y = np.zeros(int((I[1]-I[0])/h))
    else:
        Y = np.zeros([int((I[1]-I[0])/h),np.size(y0)])

    t = np.arange(I[0], I[1], h)

    # Set initial values
    Y[0] = y0
    t[0] = I[0]

    # Calculate the c_i by using Theorem 1.40
    for i in range(size_i):
        for j in range(i):
            c[i] = A[i][j]

    # Calculating the Y_k by using Definition 1.34
    for k in range(int((I[1]-I[0])/h) - 1):

        phi = 0
        for i in range(size_i):

            temp = Y[k]
            for j in range(i):
                temp = temp + h * A[i][j] * RK_K[j]
            RK_K[i] = f(t[k] + h * c[i], temp)
            phi += weights[i] * RK_K[i]

        Y[k+1] = Y[k] + h * phi

    return t, Y


def main():

    # Step sizes and a place to store the errors
    h_list = np.array([0.5, 0.25, 0.125, 0.0625])
    err_list = np.zeros(4)
    y2_list = []

    # Test Parameters from the excercise to be used for: runge_kutta_ex
    y0 = np.array([0,1])
    I = np.array([0,20])

    weights = np.array([(1/6), (2/6), (2/6), (1/6)])

    A = np.zeros([4, 4])
    A[1][0] = 1/2
    A[2][1] = 1/2
    A[3][2] = 1

    f = lambda t, y : np.array([y[1], 10*(1 - y[0]**2)*y[1] - y[0]])

    # Calculating of "exact" values and checking error between references

    t_ref, Y_ref = runge_kutta_ex(f, y0, I, 2**(-14), weights, A)
    t_ref2, Y_ref2 = runge_kutta_ex(f, y0, I, 2**(-15), weights, A)

    err_ref = 0
    for k in range(int(np.size(Y_ref) / np.size(y0))):
        var = np.max(np.abs( Y_ref2[2 * k] - Y_ref[ k ] ))
        if(var > err_ref):
            err_ref = var

    if err_ref < 10**(-9):
        print("Error between references small enough!")
    else:
        print("Error between references are too big!")

    fig = plt.figure(figsize=(10, 6))

    # Calculating and plotting of numerical values
    for l in range(np.size(h_list)):
        t, Y = runge_kutta_ex(f, y0, I, h_list[l], weights, A)

        # Calculating of max errors
        for k in range(int(np.size(Y) / np.size(y0))):
            var = np.max(np.abs(Y[k] - Y_ref[ k * int( np.round( ( h_list[l] / (2**(-14)) ) , decimals=0) ) ] )) 
            if(var > err_list[l]):
                err_list[l] = var
        print("Errors:", err_list)

        Y_1 = [y[0] for y in Y]
        Y_2 = [y[1] for y in Y]
        y2_list.append([t, Y_2])
        plt.plot(t, Y_1, '.-')

    plt.ylim(0, 5)
    plt.legend(["h = 0.5, y1", "h = 0.25, y1", "h = 0.125, y1", "h = 0.0625, y1"])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("y_1", fontsize=16)

    plt.title("runge_kutta_ex", fontsize=17)
    fig.show()

    fig2 = plt.figure(figsize=(10, 6))
    for i in y2_list:
        plt.plot(i[0], i[1], '-')

    plt.ylim(0, 12)
    plt.legend(["h = 0.5, y2", "h = 0.25, y2", "h = 0.125, y2", "h = 0.0625, y2"])
    plt.xlabel("t", fontsize=16)
    plt.ylabel("y_2", fontsize=16)

    plt.title("runge_kutta_ex", fontsize=17)
    fig2.show()

    # Input at the end to keep the plots up
    input("Input:")

# Main
if __name__ == "__main__":
    main()


