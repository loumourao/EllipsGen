import numpy as np
import pandas as pd
from configurations import configurations
from matrices import matrices
from ellipsegraph import ellipsegraph
from membership_test import membership_test

# main.m - Ellipsoid Pair Data Generation and Export
# 
# This script serves as the main code to run in order to generate and store
# geometric and transformation data for pairs of ellipsoidal particles. 
# The generated dataset is saved to a CSV file for further analysis. 
# The script can also visualizes a selected pair of ellipses.
#
# Author: Gaston Banna
# Date: 28 Septembre 2023
#
# -------------------------------------------------------------------------
# Functionality:
# - Generates N random pairs of ellipsoidal particles.
# - Computes their geometric parameters and transformations.
# - Stores the results in a structured data matrix.
# - Saves the data as 'data.csv'.
# - Visualizes a randomly selected pair of ellipses.
#
# Dependencies:
# - Configurations.m: Computes ellipsoid configurations.
# - Matrices.m: Computes transformation matrices.
# - ellipsegraph.m: Plots ellipses.
# - test_appartenance.m: Tests point membership in ellipses.
#
# Output:
# - 'data.csv' : Table of generated ellipse parameters.
#
# -------------------------------------------------------------------------

# Random seeding that match MATLAB environment
# for debugging purposes
rng = np.random.RandomState(42)

# Initialization
N = 2 # Number of ellipse pairs

# Preallocate arrays for efficiency
A = np.zeros((2, 2))
B = np.zeros(10)
x = np.zeros((N, 2))
y = np.zeros((N, 2))
param = np.zeros((N, 10))
data = np.zeros((N, 38))

# Generate Random Ellipse Pairs
for i in range(0, N):
    # Randomly generate ellipse parameters
    gamma_i = 1 + 19*rng.rand()
    omega_i = 1 + 19*rng.rand()
    theta_i = np.pi*rng.rand()
    gamma_j = 1 + 19*rng.rand()
    theta_j = np.pi*rng.rand()
    c_j = rng.rand(2, 1) # Center of the second ellipse
    phi = -np.pi + 2*np.pi*rng.rand() # Random rotation angle
    epsilon_bar = np.sign(-1 + 2*rng.rand())*10**(-(1 + 1*rng.rand())) # "Distance"
    
    # Compute ellipse configurations
    E_i, E_j, x_i, x_j, epsilon = configurations(gamma_i, omega_i, theta_i, gamma_j, theta_j, c_j, phi, epsilon_bar)
    
    # Store position and parameters
    A = np.hstack((x_i, x_j))
    B = np.hstack((E_i, E_j))
    x[i, :] = A[0, :]
    y[i, :] = A[1, :]
    param[i, :] = B
    
    # Store ellipse parameters and computed properties in the dataset
    data[i, 0:5] = E_i
    data[i, 5:7] = np.array([gamma_i, omega_i])
    data[i, 7:9] = np.array([np.cos(theta_i), np.sin(theta_i)])
    Q_i, o_i, *_ = matrices(E_i) # Compute transformation matrix
    data[i, 9:12] = np.array([Q_i[0,0], Q_i[0,1], Q_i[1,1]])
    data[i, 12:16] = np.array([np.linalg.norm(o_i), np.atan2(o_i[1,0], o_i[0,0]), np.cos(np.atan2(o_i[1,0], o_i[0,0])), np.sin(np.atan2(o_i[1,0], o_i[0,0]))])
    
    data[i, 16:21] = E_j
    data[i, 21] = gamma_j
    data[i, 22:25] = np.array([1, np.cos(theta_j), np.sin(theta_j)])
    Q_j, o_j, *_ = matrices(E_j)
    data[i, 25:28] = np.array([Q_j[0,0], Q_j[0,1], Q_j[1,1]])
    data[i, 28:32] = [np.linalg.norm(o_j), np.atan2(o_j[1,0], o_j[0,0]), np.cos(np.atan2(o_j[1,0], o_j[0,0])), np.cos(np.atan2(o_j[1,0], o_j[0,0]))]
    
    # Store contact point and perturbation information
    data[i, 32:38] = np.array([x_i[0,0], x_i[1,0], x_j[0,0], x_j[1,0], np.sign(epsilon_bar)*np.linalg.norm(x_j - x_i), epsilon_bar])


# Export Data to CSV File
parameters = ('a_i', 'b_i', 'theta_i', 'o_i_x', 'o_i_y', 'gamma_i',
              'omega_i', 'cos_theta_i', 'sin_theta_i', 'Q_i_11', 'Q_i_12',
              'Q_i_22', 'r_i', 'phi_i', 'cos_phi_i', 'sin_phi_i', 'a_j', 'b_j',
              'theta_j', 'o_j_x', 'o_j_y', 'gamma_j', 'omega_j', 'cos_theta_j',
              'sin_theta_j', 'Q_j_11', 'Q_j_12', 'Q_j_22', 'r_j', 'phi_j',
              'cos_phi_j', 'sin_phi_j', 'x_i_x', 'x_i_y', 'x_j_x', 'x_j_y',
              'epsilon','epsilon_bar')
T = pd.DataFrame(data)
T.columns = parameters
T.to_csv('data.csv', index=False)

# Visualization: Display a Random Pair of Ellipses
pair = 1
E_i = param[pair, 0:5]
E_j = param[pair, 5:10]
x_i = np.array([[x[pair, 0]],
                [y[pair, 0]]])
x_j = np.array([[x[pair, 1]],
                [y[pair, 1]]])
ellipsegraph(1, E_i, E_j, 3, x_i, x_j)

# Test point membership for the displayed pair
p1inE, p1inH, p1ok = membership_test(x_i, E_i, E_j)
p2inE, p2inH, p2ok = membership_test(x_j, E_j, E_i)
