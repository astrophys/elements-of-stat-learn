import numpy as np
from error import exit_w_error

def nearest_neighbor(DataL = None, N = None):
    """
    ARGS:
        Data   = Training Data
        N      = Number of nearest neighbors to vote.
    RETURN:
    DESCRIPTION:
        
    NOTES: 
    DEBUG:
    FUTURE:
        1. Debug...
    """



def get_N_nearest_neighbors_votes(DataL = None, N = None, Pos = None):
    """
    ARGS:
        Data   = Training Data
        N      = Number of nearest neighbors to vote.
        Pos    = position in x1,x2,etc
    RETURN:
    DESCRIPTION:
    NOTES: 
        As one takes the limit of N->1, you are effectively fitting
        N points with N parameters.  This isn't good. Kind of like
        fitting N-points with an N'th order polynomial. Called 
        'overfitting'.

        Most appropriate when (quoting Hastie):
            'The training data in each class came from a mixture of 10 low-variance
            Gaussian distributions, with individual means themselves distributed
            as Gaussian.' 
        defined as Scenerio 2 on p13. We should TEST!!

    DEBUG:
        1. Visually compared my plot with Hastie Fig 2.2. Very close
    FUTURE:
        1. Test Scenerio 2 as defined on ESL by Hastie p13
    """
    dist2L  = []
    valueL  = []

    # Loop over groups
    x1 = []
    x2 = []
    groupL = []
    for grpIdx in range(len(DataL)):
        if(len(DataL[grpIdx]) != 2):
            exit_w_error("ERROR!!! Expecting only 2 dimenional categoral data")
        x1.extend(DataL[grpIdx][0])
        x2.extend(DataL[grpIdx][1])
        groupL.extend( [grpIdx] * len(DataL[grpIdx][0]))
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    for i in range(len(x1)):
        dist = (Pos[0] - x1[i])**2 + (Pos[1] - x2[i])**2
        dist2L.append(dist)

    #(distSort2L, groupSortL) = sorted(zip(dist2L, groupL), key = lambda t: t[0])
    (distSort2L, groupSortL) = (list(x) for x in zip(*sorted(zip(dist2L, groupL))))
    score = np.mean(groupSortL[0:N])
    if(score >= 0.5):
        return 1
    else:
        return 0




