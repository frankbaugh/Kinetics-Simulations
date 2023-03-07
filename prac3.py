import pandas as pd
import pickle
from scipy.integrate import solve_ivp
import numpy as np

def fold(urea):
    "This does the folding simulation"
    ## Define Rates as they vary with urea concentration
    k= np.zeros(4)
    k[0] = 26000 * pow(np.e, -1.68 * urea)
    k[1] = 0.06 * pow(np.e, 0.95 * urea)
    k[2] = 730 * pow(np.e, -1.72 * urea)
    k[3] = 0.00075 * pow(np.e, 1.2 * urea)

    ## Taking the state as a vector of D,I,N, the RHS ODE function is a matrix multiplied by the State
    f_mat = np.array([[-k[0], k[1], 0],[k[0], -k[1]-k[2], k[3]], [0, k[2], -k[3]]])

    ## Parameters for the integration
    tmin, tmax, dt = 0, 5, pow(10, -6)
    delta_t, length = 1, 1

    ## Threshold for equilibrium
    epsilon = 1 * pow(10, -3)

    State = [1,0,0]

    ## Include overall history so that the time evolution can be plotted
    sol_tot = np.empty((3,1))
    t_tot = np.empty((1))

    ## Defines the ODE function in one line
    F = lambda t, st: np.dot(f_mat, st)

    ## keep integrating over the next time interval until equilibrium is reached
    while length > epsilon:

        tmax += delta_t
        t_eval = np.arange(tmin,tmax,dt)

        ## Scipy solve_ivp function to give solution
        sol = solve_ivp(F, [tmin, tmax], State, t_eval=t_eval, method = "LSODA")

        end_pos = len(sol.y[0]) - 1
        S_new = np.array([sol.y[0][end_pos], sol.y[1][end_pos], sol.y[2][end_pos]])
        length = abs(S_new[0] - State[0]) + abs(S_new[1] - State[1]) + abs(S_new[2] - State[2])

        State = S_new

        sol_tot = np.concatenate((sol_tot, sol.y), axis = 1)
        t_tot = np.concatenate((t_tot, np.array(sol.t)))
        tmin = tmax

    ## concantenating arrays is weird; first row needs to be deleted
    sol_tot = np.delete(sol_tot, 0, axis = 1)
    t_tot = np.delete(t_tot, 0, axis = 0)

    ## Prints the amount of time needed before equilibrium reached for each run
    print(tmax)
    return State, sol_tot, t_tot

def save_fold():

    D, I, N = [], [], []
    urea = np.arange(0,9,0.2)

    ## Iterate over urea concentrations
    for u in urea:
        St, sol_tot, t_tot = fold(u)
        D.append(St[0])
        I.append(St[1])
        N.append(St[2])

    with open("folding.dat", "w") as f:
        f.write("# U \t D \t I \t N \n") 
        for udin in zip(urea, D, I, N): 
            f.write("\t".join([str(n) for n in udin])+ "\n")

save_fold()