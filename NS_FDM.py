# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
from LBM_variables import *

# FUNCTIONS
def velocityBoundary(U_slice,V_slice):
    V_slice[:,0] = 2*v_left - V_slice[:,1]
    V_slice[:,-1] = 2*v_right - V_slice[:,-2]
    U_slice[0,:] = 2*u_top - U_slice[1,:]
    U_slice[-1,:] = 2*u_bottom - U_slice[-2,:]

    U_slice[:,0] = u_left
    U_slice[:,-1] = u_right
    V_slice[0,:] = v_top
    V_slice[-1,:] = v_bottom

    return U_slice,V_slice

def pressureBoundary(P_slice):
    P_slice[:,0] = 2*p_left - P_slice[:,1]
    P_slice[:,-1] = 2*p_right - P_slice[:,-2]
    P_slice[0,:] = 2*p_top - P_slice[1,:]
    P_slice[-1,:] = 2*p_bottom - P_slice[-2,:]

    return P_slice

def modifiedBoundary(k,mode = "VP"):
    if mode == "P":
        P[0,:,k] = P[1,:,k]
        P[-1,:,k] = P[-2,:,k]
        P[0,:,k] = P[1,:,k]
        P[-1,:,k] = P[-2,:,k]
    else:
        V[:,0,k] = V[:,1,k]
        V[:,-1,k] = V[:,-2,k]
        V[0,:,k] = V[1,:,k]
        V[-1,:,k] = V[-2,:,k]
        U[0,:,k] = U[1,:,k]
        U[-1,:,k] = U[-2,:,k]
        U[0,:,k] = U[1,:,k]
        U[-1,:,k] = U[-2,:,k]
        P[0,:,k] = P[1,:,k]
        P[-1,:,k] = P[-2,:,k]
        P[0,:,k] = P[1,:,k]
        P[-1,:,k] = P[-2,:,k]

def timeStepping(k):
    if k != 0:
        return CFL_coefficient / (np.amax(U[:,:,k])/deltax + np.amax(V[:,:,k])/deltay)
    else:
        return CFL_coefficient * (deltax + deltay)

def simulationProgress(t_i): # t_i: time index
    percentage = int(t_i/n_ts * 100) # approximate when we consider the whole run
    if percentage != 100:
        os.system('cls')
        print("Finite Difference Run:")
        n_bars = percentage//5
        if percentage < 10:
            print("%s%% " % percentage, end = ' ')
        else:
            print("%s%%" % percentage, end = ' ')
        print("[",end = '')
        for i in range(1,21):
            if i <= n_bars:
                print("|",end = '')
            else:
                print(" ",end = '')
        print("]")

# # SPECIFICATIONS
#                             columns
#         _________________________________________________
#        |                                                |
#        |                                                |
#  rows  |                   ------->                     |
#        |                                                |
#        |________________________________________________|
#                              n_ts

# SEPARATE VARIABLES
rho = 1 # density of fluid
mu = 1E-2 # viscosity of fluid
nu = mu/rho # kinematic viscosity of fluid
deltat = 1E-3 # initial time increment (bo be changed)
deltax = 1E-2 # x increment
deltay = 1E-2 # y increment
eps = 1E-4 # tolerance
CFL_coefficient =  0.01 # coefficient for CFL-based timestepping

# STAGGERED GRID DEFINTION
grid_size = (rows+2, columns+2, n_ts+1)
grid_size_modified = (rows, columns, n_ts+1)
U = np.zeros(grid_size); V = np.zeros(grid_size)
U_star = np.zeros(grid_size); V_star = np.zeros(grid_size)
P = np.zeros(grid_size)

# INITIAL CONDITIONS
#P[1:rows+1,1:columns+1,0] = (np.ones((rows,columns)) + 0.05*np.random.uniform(rows,columns))*0.1
U[1:rows+1,1:columns+1,0] = np.load("NS_initial_conditions_Ux.npy") # swapped Ux and Uy because of difference in axes definition
V[1:rows+1,1:columns+1,0] = np.load("NS_initial_conditions_Uy.npy") # swapped Ux and Uy because of difference in axes definition
v_top = np.amax(V[1,:,0])*0
v_bottom = np.amax(V[-2,:,0])*0
v_right = np.amax(V[:,-2,0])
v_left = np.amax(V[:,1,0])
u_top = np.amax(U[1,:,0])*0
u_bottom = np.amax(U[-2,:,0])*0
u_right = np.amax(U[:,-2,0])
u_left = np.amax(U[:,1,0])
p_top = np.amax(P[1,:,0])
p_bottom = np.amax(P[-2,:,0])
p_right = np.amax(P[:,-2,0])
p_left = np.amax(P[:,1,0])
#U[1:rows+1,1:columns+1,0] = 0; V[1:rows+1,1:columns+1,0] = 0

# DRIVING LOOP
for k in range(1,n_ts+1): # till n_ts
    # Boundary Conditions
    U[:,:,k] = U[:,:,k-1].copy()
    V[:,:,k] = V[:,:,k-1].copy()
    modifiedBoundary(k)
    # U[:,:,k-1],V[:,:,k-1] = velocityBoundary(U[:,:,k-1],V[:,:,k-1])
    # P[:,:,k-1] = pressureBoundary(P[:,:,k-1])

    # Predictor Step
    # u||v-momentum (|| in C++ terminology not Python)
    deltat = timeStepping(k-1)
    U_star[:,:,k-1] = U[:,:,k-1].copy()
    dUdx_1 = (U[1:rows+1,2:,k-1] - U[1:rows+1,0:columns,k-1]) / (2 * deltax)
    dUdy_1 = (U[2:,1:columns+1,k-1] - U[0:rows,1:columns+1,k-1]) / (2 * deltay)
    dUdx_2 = (U[1:rows+1,2:,k-1] - 2*U[1:rows+1,1:columns+1,k-1] + U[1:rows+1,0:columns,k-1]) / (deltax ** 2)
    dUdy_2 = (U[2:,1:columns+1,k-1] - 2*U[1:rows+1,1:columns+1,k-1] + U[0:rows,1:columns+1,k-1]) / (deltay ** 2)
    V_face = (V[1:rows+1,1:columns+1,k-1] + V[1:rows+1,0:columns,k-1] + \
        V[2:,0:columns,k-1] + V[2:,1:columns+1,k-1])/4
    U_star[1:rows+1,1:columns+1,k-1] = U[1:rows+1,1:columns+1,k-1] - deltat * \
        (U[1:rows+1,1:columns+1,k-1] * dUdx_1 + V_face * dUdy_1 - \
            nu * (dUdx_2 + dUdy_2))
    
    V_star[:,:,k-1] = V[:,:,k-1].copy()
    dVdx_1 = (V[1:rows+1,2:,k-1] - V[1:rows+1,0:columns,k-1]) / (2 * deltax)
    dVdy_1 = (V[2:,1:columns+1,k-1] - V[0:rows,1:columns+1,k-1]) / (2 * deltay)
    dVdx_2 = (V[1:rows+1,2:,k-1] - 2*V[1:rows+1,1:columns+1,k-1] + V[1:rows+1,0:columns,k-1]) / (deltax ** 2)
    dVdy_2 = (V[2:,1:columns+1,k-1] - 2*V[1:rows+1,1:columns+1,k-1] + V[0:rows,1:columns+1,k-1]) / (deltay ** 2)
    U_face = (U[1:rows+1,1:columns+1,k-1] + U[0:rows,1:columns+1,k-1] + \
        U[0:rows,2:,k-1] + U[1:rows+1,2:,k-1])/4
    V_star[1:rows+1,1:columns+1,k-1] = V[1:rows+1,1:columns+1,k-1] - deltat * \
        (U_face * dVdx_1 + V[1:rows+1,1:columns+1,k-1] * dVdy_1 - \
            nu * (dVdx_2 + dVdy_2))

    # poisson equation
    dUsdx_1 = (U_star[1:rows+1,2:,k-1] - U_star[1:rows+1,0:columns,k-1]) / (2 * deltax)
    dVsdy_1 = (V_star[2:,1:columns+1,k-1] - V_star[0:rows,1:columns+1,k-1]) / (2 * deltay)
    P[:,:,k] = P[:,:,k-1].copy()
    factor = 1/(2*(1/deltax**2 + 1/deltay**2))
    error = 1
    m = 0
    while eps < error:
        P_previous = P[:,:,k].copy() # note taht P_previous is 2-dimensional NOT 3-dimensional
        # using k-1 instead might have been more optimal
        #dPdx_2 = (P_previous[1:rows+1,2:] + P_previous[1:rows+1,0:columns]) / (deltax ** 2)
        #dPdy_2 = (P_previous[2:,1:columns+1] + P_previous[0:rows,1:columns+1]) / (deltay ** 2)
        P[1:rows+1,1:columns+1,k] = (((P_previous[1:rows+1,2:]+P_previous[1:rows+1,0:columns]) * deltay**2) + \
            (P_previous[2:,1:columns+1]+P_previous[0:rows,1:columns+1]) * deltax**2) * factor - \
                (rho * deltax**2 * deltay**2) * factor * \
                    ((1/deltat) * (dUdx_1+dVdy_1) - \
                        dUdx_1**2 - 2*dUdy_1*dVdx_1 - 2*dVdy_1**2)

        # P[:,:,k] = pressureBoundary(P[:,:,k]) # setting boundaries of P[:,:,k]
        modifiedBoundary(k,mode = "P")
        m += 1
        if m > 1000:
            eps *= 10

    # Corrector Step
    dPdx_1 = (P[1:rows+1,2:,k] - P[1:rows+1,0:columns,k]) / (2 * deltax)
    dPdy_1 = (P[2:,1:columns+1,k] - P[0:rows,1:columns+1,k]) / (2 * deltay)
    U[1:rows+1,1:columns+1,k] = U_star[1:rows+1,1:columns+1,k-1] - (deltat/rho) * dPdx_1
    V[1:rows+1,1:columns+1,k] = V_star[1:rows+1,1:columns+1,k-1] - (deltat/rho) * dPdy_1

    # Visualization
    vorticity = (np.roll(U[1:rows+1,1:columns+1,k], -1, axis=0) - np.roll(U[1:rows+1,1:columns+1,k], 1, axis=0)) -\
        (np.roll(V[1:rows+1,1:columns+1,k], -1, axis=1) - np.roll(V[1:rows+1,1:columns+1,k], 1, axis=1))
    plt.imshow(vorticity, cmap = 'bwr')
    plt.title("%s" % k)
    plt.savefig("NS_FDM_%s" % k)

    # Checking Progress
    simulationProgress(k)

# VIDEO EXPORTATION
test_img = cv2.imread("NS_FDM_1.png")
height, width, layers = test_img.shape
framesize = (width, height)
output = cv2.VideoWriter("NS_FDM_video.avi",cv2.VideoWriter_fourcc(*'DIVX'),10,framesize)
for fname in sorted(glob.glob("*.png"), key = os.path.getmtime):
    img = cv2.imread(fname)
    output.write(img)
    os.remove(fname)

# EXPORTING DATA
np.save("NS_FDM_U.npy",U[1:rows+1,1:columns+1,1:n_ts+1])
np.save("NS_FDM_V.npy",V[1:rows+1,1:columns+1,1:n_ts+1])

# NOTIFYING THE USER THAT THE SIMULATION IS DONE
os.system('cls')
print("Finite Difference Run:")
print("100% [||||||||||||||||||||]")