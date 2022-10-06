# Author : Ali Snedden
# Date   : 10/05/22
# License: MIT
#
# Questions :
#
#
"""This code test's my understanding of k-means clustering
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

def make_data(MuL=None, CovL=None, N=None):
    """
    Takes list of group averages (MuL) and variances (VarL) and generates 
    N point points per group from a Gaussian distributions in 2D space

    Args:
        MuL = (list of list of floats), list of mean x,y posistions for group center
        CovL= (list of covariance matrices)
        N   = (int), number of points to use in each group

    Returns:
        Matrix of floating point x-y coordinates with the last column is the class

    Raises:
    """
    dataL = []
    cL    = []
    ngrp  = len(MuL)
    dataM = np.zeros([ngrp*N, len(MuL[0])+1])
    classV= np.zeros(ngrp*N)
    j     = 0
    # loop over groups
    for i in range(ngrp):
        v = np.random.multivariate_normal(MuL[i], CovL[i], N)
        c = np.zeros(N)
        c[:] = i
        print(v)
        print("")
        dataM[i*N : ((i+1)*N), 0:2] = v
        dataM[i*N : ((i+1)*N), 2] = c
    return dataM


def plot_data():
    """

    Args:

    Returns:

    Raises:
    """


def kmeans():
    """

    Args:

    Returns:

    Raises:
    """


def main():
    """

    Args:

    Returns:

    Raises:
    """
    n    = 10
    muL  = [[0,0], [10,10], [10,5]]
    covL = [ [[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]] ]
    dataM= make_data(MuL=muL, CovL=covL, N=n)
    ngrp = len(muL)
    # 
    markerL = ["o", "v", "8", "s", "p", "P", "*", "h", "D", "."]
    colorL = ["black", "red", "blue", "orange", "olivedrap", "pink",
              "slategrey", "yellow", "teal", "skyblue"]
    for i in range(ngrp):
        tmp = dataM[dataM[:,-1]==i]
        plt.scatter(tmp[:,0], tmp[:,1], c=colorL[i], marker=markerL[i])
    plt.show()
    sys.exit(0)

if __name__ == "__main__":
    main()
