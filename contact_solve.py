import numpy as np
from scipy.optimize import root_scalar
from matrices import matrices

## Contact_Solve.m is a function that computes the contact points between 
## two ellipses.
#
# Given two ellipses defined by their parameter sets E_i and E_j, this 
# function finds the contact points x_i and x_j that satisfy the geometric 
# conditions.
#
# -------------------------------------------------------------------------
# Inputs:
#   E_i - Parameter set of the first ellipse [a, b, theta, o_x, o_y]
#   E_j - Parameter set of the second ellipse [a, b, theta, o_x, o_y]
#
# Outputs:
#   x_i - Contact point on the first ellipse
#   x_j - Contact point on the second ellipse
#   varargout:
#     {1} - Residual error for the first contact equation (fval1)
#     {2} - Residual error for the second contact equation (fval2)
#     {3} - Exit flag for the first solver (exitflag1)
#     {4} - Exit flag for the second solver (exitflag2)
#     {5} - Solver output details for the first contact point (output1)
#     {6} - Solver output details for the second contact point (output2)
#
# -------------------------------------------------------------------------

def contact_solve(E_i, E_j):
    # Compute quadratic matrices and centers of the ellipses
    Q_i, o_i, *_ = matrices(E_i)
    Q_j, o_j, *_ = matrices(E_j)
    
    # Define quadratic potential functions for each ellipse
    ei = lambda x : x.T @ Q_i @ x - 1;  # Ellipse i equation
    ej = lambda x : x.T @ Q_j @ x - 1;  # Ellipse j equation
    
    # Define the parametric form w(u) of co-gradient locus H_ij
    w = lambda u : np.linalg.solve((1-u)*Q_i + u*Q_j,
                                   (1-u)*(Q_i @ o_i) + u*(Q_j @ o_j))
    
    # Define contact conditions for each ellipse
    contact_point_1 = lambda u : ei(w(u) - o_i)
    contact_point_2 = lambda u : ej(w(u) - o_j)
    
    # Solve for the first contact point
    result1 = root_scalar(contact_point_1, bracket=[0, 1], method='brentq')
    parameter_1 = result1.root
    fval1 = contact_point_1(parameter_1)
    exitflag1 = result1.converged
    output1 = {'iterations': result1.iterations,
               'function_calls': result1.function_calls,
               'flag': 'converged' if exitflag1 else 'not converged'}
    x_i = w(parameter_1)
    
    # Solve for the second contact point
    result2 = root_scalar(contact_point_2, bracket=[0, 1], method='brentq')
    parameter_2 = result2.root
    fval2 = contact_point_2(parameter_2)
    exitflag2 = result2.converged
    output2 = {'iterations': result2.iterations,
               'function_calls': result2.function_calls,
               'flag': 'converged' if exitflag2 else 'not converged'}
    x_j = w(parameter_2)

    return x_i, x_j, fval1, fval2, exitflag1, exitflag2, output1, output2
