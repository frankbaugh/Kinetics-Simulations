from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

# Define rate constants globally for speed
k1 = 1.34 * pow(10, 0)
k2 = 1.6 * pow(10, 9)
k3 = 8 * pow(10, 3)
k4 = 4 * pow(10, 7)
k5 = 1 * pow(10, 0)

def ODE_func(t, state):

    ## A, B, P, Q, X, Y, Z
    ## 0, 1, 2, 3, 4, 5, 6
    ## Sacrificing some readability for speed here
    r1 = -1 * k1 * state[0] * state[5]
    r2 = -1 * k2 * state[4] * state[5]
    r3 = -1 * k3 * state[1] * state[4]
    r4 = k4 * state[4] * state[4]
    r5 = k5 * state[6]

    return np.array([r1,r3, -1 * r1 -r2,r4,-1 * r1 + r2 -r3 - 2 * r4,
                    r1 + r2 + r5, -1 * r3 - r5])

def getjac(t, state):
    ## Jacobian = df_i / dy_j ; needed here to obtain reasonable results with
    ## the stiff solver method Radau

    jac = np.zeros((7,7))
    jac[0][0] = -1 * k1 * state[5]
    jac[0][5] = -1 * k1 * state[0]
    jac[1][1] = -1 * k3 * state[4]
    jac[1][4] = -1 * k3 * state[1]
    jac[2][0] = k1 * state[5]
    jac[2][4] = k2 * state[5]
    jac[2][5] = k1 * state[0] + k2 * state[1]
    jac[3][4] = 2 * k4 * state[4]
    jac[4][0] = k1 * state[5]
    jac[4][1] = k3 * state[4]
    jac[4][4] = -1 * k2 * state[5] + k3 * state[1] - 4 * k4 * state[4]
    jac[4][5] = k1 * state[0] - k2 * state[4]
    jac[5][0] = -1 * k1 * state[5]
    jac[5][4] = -1 * k2 * state[5]
    jac[5][5] = -1 * k1 * state[0] - k2 * state[4]
    jac[5][6] = k5
    jac[6][1] = k3 * state[4]
    jac[6][4] = k3 * state[1]
    jac[6][6] = -1 * k5

    return jac

## Define initial conditions, again global for speed
initial = np.array([0.06, 0.06, 0, 0, 1 * pow(10, -9.8), 1 * pow(10, -6.52), 1 * pow(10, -7.32)])

def sampler(T, X, Y, Z):

    ## Take only some of the data (there is far more than required for the plot)
    indices = np.arange(0, len(T), 100)
    T = np.take(T, indices)
    X = np.take(X, indices)
    Y = np.take(Y, indices)
    Z = np.take(Z, indices)

    return T, X, Y, Z

def run_reac():

    ## Define time interval and time step
    tmin, tmax, dt = 0, 90, 5 * pow(10,-5)
    t_eval = np.arange(tmin,tmax,dt)

    ## Scipy uses c / fortran and is much quicker than anything I could write
    sol = solve_ivp(ODE_func, [tmin, tmax], initial, t_eval=t_eval, dense_output=False, vectorized=True, method = "Radau", jac = getjac)

    ## Extract relevant data from solve_ivp output
    T, X, Y, Z = sampler(sol.t, sol.y[4], sol.y[5], sol.y[6])

    ## Dataframes and pickle much quicker than saving as raw text
    BZ = pd.DataFrame(np.column_stack([T, X, Y, Z]), 
        columns=['Time', 'X', 'Y', 'Z'])

    BZ.to_pickle('BZdf_test.pkl')


start_time = time.time()

run_reac()

print("--- %s seconds ---" % (time.time() - start_time))




 









