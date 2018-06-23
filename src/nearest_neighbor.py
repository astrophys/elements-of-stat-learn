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
    DEBUG:
    FUTURE:
        1. Debug...
        2. Make efficient
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




