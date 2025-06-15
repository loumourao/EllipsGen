import numpy as np

## Matrices.m is a function that computes the quadratic matrix and the  
## center for an ellipse defined by a parameter set.
#
# Given the parameter set E of the ellipse, this function calculates the 
# quadratic matrix Q and the center o of the ellipse. It also provides 
# additional outputs for the rotation matrix R and the scaling matrix D.
#
# -------------------------------------------------------------------------
# Inputs:
#   E - Parameter set of the ellipse [a, b, theta, o_x, o_y]
#     where:
#       a     - Semi-major axis
#       b     - Semi-minor axis
#       theta - Rotation angle of the ellipse
#       o_x   - x-coordinate of the center
#       o_y   - y-coordinate of the center
#
# Outputs:
#   Q - Quadratic matrix representing the ellipse
#   o - Center of the ellipse [o_x, o_y]
#   varargout:
#     {1} - Rotation matrix R for the ellipse
#     {2} - Scaling matrix D for the ellipse
#
# -------------------------------------------------------------------------

def matrices(E):
    # Extract parameters from the input
    a = E[0]      # Semi-major axis
    b = E[1]      # Semi-minor axis
    theta = E[2]  # Rotation angle
    o_x = E[3]    # x-coordinate of the center
    o_y = E[4]    # y-coordinate of the center
    
    # Define the rotation matrix for the given angle
    R = lambda theta : np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    
    # Define the scaling matrix for the given semi-axes
    D = lambda a, b : np.array([[1/a**2, 0],
                                [0, 1/b**2]])
    
    # Compute the quadratic matrix Q
    Q = R(theta) @ D(a, b) @ R(-theta)
    
    # Define the center of the ellipse
    o = np.array([[o_x],
                  [o_y]])
    
    return Q, o, R(theta), D(a,b)
