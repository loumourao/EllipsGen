import numpy as np

## init_ellipse.m is a function that computes the x and y coordinates 
## for an ellipse based on its parameters. The ellipse is parameterized 
## using a set of inputs, and the coordinates are returned for plotting.
#
# Given the semi-major axis 'a', semi-minor axis 'b', center coordinates 
# ('x0', 'y0'), and rotation angle 'theta', this function generates the 
# parametric coordinates of the ellipse.
#
# -------------------------------------------------------------------------
# Inputs:
#   a     - Semi-major axis of the ellipse
#   b     - Semi-minor axis of the ellipse
#   x0    - x-coordinate of the ellipse center
#   y0    - y-coordinate of the ellipse center
#   theta - Rotation angle of the ellipse (in radians)
#
# Outputs:
#   ellipsex - x-coordinates of the ellipse
#   ellipsey - y-coordinates of the ellipse
#
# -------------------------------------------------------------------------

def init_ellipse(a, b, x0, y0, theta):
    # Define a parameter t that ranges from -pi to pi for the parametric 
    # representation of the ellipse
    t = np.arange(-np.pi, np.pi, 0.0001)
    
    # Compute the parametric coordinates of the ellipse
    ellipsex = x0 + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
    ellipsey = y0 + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
    
    return ellipsex, ellipsey
