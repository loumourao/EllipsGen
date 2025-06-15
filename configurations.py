import numpy as np
from parametrize import parametrize
from contact_solve import contact_solve

## Configurations.m is a function that generates a pair of ellipses and 
## computes their contact points
#
# -------------------------------------------------------------------------
# Inputs:
#   gamma_i, omega_i - Shape parameters of the first ellipse
#   theta_i - Orientation of the first ellipse
#   gamma_j - Shape parameter of the second ellipse
#   theta_j - Orientation of the second ellipse
#   o_j - Center of the second ellipse
#   phi - Contact angle parameter
#   epsilon_bar - Distance parameter, using the contact point on (i)
#
# Outputs:
#   E_i - Parameters of the first ellipse [a, b, theta, o_x, o_y]
#   E_j - Parameters of the second ellipse [a, b, theta, o_x, o_y]
#   x_i - Contact point on the first ellipse
#   x_j - Contact point on the second ellipse
#
# -------------------------------------------------------------------------

def configurations(gamma_i, omega_i, theta_i, gamma_j, theta_j, o_j, phi, epsilon_bar):
    # Rotation matrix function
    R = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    
    # Diagonal matrix function for ellipse shape
    D = lambda a, b : np.array([[1/a**2, 0],
                                [0, 1/b**2]])

    # Compute semi-axes of the ellipses
    a_i = np.sqrt(omega_i * gamma_i)
    b_i = np.sqrt(omega_i / gamma_i)
    a_j = np.sqrt(gamma_j)
    b_j = 1/a_j

    # Compute the transformed shape matrix of the first ellipse (i) in the 
    # second ellipse (j)'s frame
    Q_i_hat = D(1/np.sqrt(a_j), 1/np.sqrt(b_j)) @ R(-theta_j) @ R(theta_i) @ D(a_i, b_i) @ R(-theta_i) @ R(theta_j) @ D(1/np.sqrt(a_j), 1/np.sqrt(b_j))
    a_i_hat, b_i_hat, theta_i_hat = parametrize(Q_i_hat)

    # Compute the initial contact point in the local frame of (i)
    x_i_bar = np.array([[a_i_hat * np.cos(phi)],
                        [b_i_hat * np.sin(phi)]])
    g_x_i_b = np.array([[b_i_hat * np.cos(phi)],
                        [a_i_hat * np.sin(phi)]])
    n_i_bar = g_x_i_b / np.linalg.norm(g_x_i_b) # Normal vector at the contact point
    o_j_bar = x_i_bar + (1 + epsilon_bar) * n_i_bar # Second ellipse pos

    # Transformations to global coordinates
    T1 = lambda x_bar : R(theta_i_hat) @ (x_bar - o_j_bar)
    T2 = lambda x_hat : R(theta_j) @ D(1/np.sqrt(a_j), 1/np.sqrt(b_j)) @ x_hat + o_j

    # Compute center of the first ellipse in the transformed frame
    o_i_bar = np.zeros((2, 1))
    o_i_hat = T1(o_i_bar)
    o_i = T2(o_i_hat)

    # Define ellipse parameters in global coordinates
    E_i = np.array([a_i, b_i, theta_i, o_i[0][0], o_i[1][0]])
    E_j = np.array([a_j, b_j, theta_j, o_j[0][0], o_j[1][0]])

    # Solve for contact points on the ellipses
    x_i, x_j, *_ = contact_solve(E_i, E_j)

    # Compute the separation distance
    epsilon = np.linalg.norm(x_j - x_i)

    return E_i, E_j, x_i, x_j, epsilon
