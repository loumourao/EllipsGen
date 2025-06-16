import matplotlib.pyplot as plt
import numpy as np
from init_ellipse import init_ellipse
from matrices import matrices

## ellipsegraph.m is a function that plots two ellipses and a co-gradient 
## locus for a given configuration. It also allows additional points to be 
## plotted on the graph based on input arguments.
#
# Given two ellipses defined by their parameter sets E_i and E_j, this 
# function visualizes both ellipses, the co-gradient locus between them, 
# and additional scatter points if provided.
#
# -------------------------------------------------------------------------
# Inputs:
#   fignumber - Number of the figure to plot
#   E_i - Parameter set of the first ellipse [a, b, theta, o_x, o_y]
#   E_j - Parameter set of the second ellipse [a, b, theta, o_x, o_y]
#   config_num - Index for selecting the label configuration for x & y axes
#   varargin - Additional scatter points to be plotted on the graph
#
# Outputs:
#   f - Handle to the created figure
#
# -------------------------------------------------------------------------

def ellipsegraph(fignumber, E_i, E_j, config_num, *args):
    # Create a figure and hold the plot for subsequent elements
    plt.figure(fignumber)
    
    # Generate and plot the first ellipse (red)
    x_ellipse, y_ellipse = init_ellipse(E_i[0], E_i[1], E_i[3], E_i[4], E_i[2])
    plt.plot(x_ellipse, y_ellipse, 'r-', linewidth=2)
    
    # Generate and plot the second ellipse (blue)
    x_ellipse ,y_ellipse = init_ellipse(E_j[0], E_j[1], E_j[3], E_j[4], E_j[2])
    plt.plot(x_ellipse, y_ellipse, 'b-', linewidth=2)
    
    # Set the axis to be equal to preserve ellipse shapes
    plt.axis('equal')
    
    # Set the ylabel based on the selected configuration
    config_name = ('$\bar y$', '$\hat y$', '$y$', '$\tildey$', '$\mathring y$')
    plt.ylabel(config_name[config_num], usetex=True)
    
    # Set the xlabel based on the selected configuration
    config_name = ('$\bar x$', '$\hat x$', '$x$', '$\tilde x$', '$\mathring x$')
    plt.xlabel(config_name[config_num], usetex=True)

    # Compute the quadratic matrices and centers of the ellipses
    Q_i, o_i, *_ = matrices(E_i)
    Q_j, o_j, *_ = matrices(E_j)
    
    # Calculate the distance and unit vector between the ellipse centers
    d = np.linalg.norm(o_j - o_i)
    n = (o_j - o_i)/d
    
    # Define the parametric co-gradient locus function
    w = lambda u : np.linalg.solve(((1 - u)*Q_i + u*Q_j), Q_j @ n)
    Choi = lambda u : u*d*w(u)
    
    # Number of points for the locus
    N_locus = 100
    x_locus = np.zeros(1 + N_locus)
    y_locus = np.zeros(1 + N_locus)
    
    # Calculate the points of the co-gradient locus
    point_i = 0
    for k in np.linspace(0, 1, N_locus+1):
        point = Choi(k)
        x_locus[point_i] = point[0]
        y_locus[point_i] = point[1]
        point_i += 1

    # Plot the co-gradient locus (cyan)
    plt.plot(x_locus + o_i[0], y_locus + o_i[1], 'c-', linewidth=2)
    
    # Plot any additional points provided as inputs
    for arg in args:
        plt.scatter(arg[0], arg[1])
    
    plt.savefig('ellipses_pair.png')
