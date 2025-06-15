import numpy as np
from matrices import matrices

## membership_test.m is a function that tests whether a point lies on an 
## ellipse, the co-gradient locus of two ellipses and checks in general the
## point's membership in sets of interest.
#
# Given a point 'x' and the parameter sets E_i and E_j for two ellipses, 
# this function computes the distance of the point from the first ellipse 
# and checks its relationship to the second ellipse’s co-gradient locus. 
# It also checks a condition related to the inner product of the ellipses' 
# vectors.
#
# -------------------------------------------------------------------------
# Inputs:
#   x    - Point to test [x, y]
#   E_i  - Parameter set of the first ellipse [a, b, theta, o_x, o_y]
#   E_j  - Parameter set of the second ellipse [a, b, theta, o_x, o_y]
#
# Outputs:
#   ellipse  - Scalar value representing the distance of the point `x` from 
#              the first ellipse (based on the quadratic form), must equal
#              1 (or 1.0000).
#   locus    - Vector representing the cross product of the vectors from 
#              the point 'x' to the centers of the ellipses, indicating 
#              whether the point is on the co-gradient locus, must equal
#              [0; 0; 0]
#   bon_point - Boolean value indicating whether the dot product of the 
#               vectors from 'x' to the ellipses' centers is negative, 
#               suggesting a certain geometrical relationship, must be 1
#
# -------------------------------------------------------------------------

def membership_test(x, E_i, E_j):
    # Compute the quadratic matrices and centers of the ellipses
    Q_i, o_i = matrices(E_i)
    Q_j, o_j = matrices(E_j)
    
    # Compute the distance of the point from the first ellipse (equation)
    ellipse = (x - o_i).T @ Q_i @ (x - o_i)
    
    # Compute the cross product of the vectors from point to centers
    locus = np.cross(np.array([[Q_i @ (x - o_i)],
                               [0]]),
                     np.array([[Q_j @ (x - o_j)],
                               [0]]))
    
    # Check if the dot product of the vectors from the point to the ellipse
    # centers is negative
    bon_point = np.dot(Q_i @ (x - o_i), Q_j @ (x - o_j)) < 0

    return ellipse, locus, bon_point
