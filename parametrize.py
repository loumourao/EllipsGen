import numpy as np

## parametrize.m is a function that extracts the parameters of an ellipse 
## from its quadratic form.
#
# Given a symmetric positive-definite matrix Q representing the implicit 
# equation of an ellipse, this function computes:
#   - a: the semi-major axis length
#   - b: the semi-minor axis length
#   - theta: the orientation angle of the ellipse
#
# -------------------------------------------------------------------------
# Inputs:
#   Q - 2x2 symmetric positive-definite matrix defining the ellipse
#
# Outputs:
#   a - Semi-major axis length
#   b - Semi-minor axis length
#   theta - Rotation angle (in radians) of the major axis from the x-axis
#
# -------------------------------------------------------------------------

def parametrize(Q):
    # Compute eigenvalues and eigenvectors of Q
    e, v = np.linalg.eig(Q)
    
    # Extract and sort eigenvalues in ascending order
    I = np.argsort(e)
    e = e[I]
    
    # Rearrange eigenvectors accordingly
    v_i = v[:, I]
    
    # Compute semi-major and semi-minor axes
    a = np.sqrt(1/e[0])
    b = np.sqrt(1/e[1])
    
    # Compute rotation angle of the major axis
    theta = np.atan2(v_i[1, 0], v_i[0, 0])
    
    return a, b, theta
