# Author : Ali Snedden
# Date   : 6/17/18
# Purpose: 
#   To understand chapters 2,3,4 and 18 in "Elements of Statistical Learning"
#   by Hastie et al. I'm going to try to use several different techniques to 
#   fit his mixture simulation, see Figure 2.1
#
#   Data got from https://web.stanford.edu/~hastie/ElemStatLearn/
#   Had to load Rdata, and print it to a file e.g. 
#   write.table(x=ESL.mixture$px2, "px2.txt")
#
#   Conclusion : 
#       I have a pretty decent understanding of linear regression
#
#
import time
import pandas as pd
import sys
import numpy as np
from error import exit_w_error
from plot import plot_data
from lin_reg import linear_regression
from nearest_neighbor import get_N_nearest_neighbors_votes
from functions import order_points

def main():
    """
    ARGS:
    RETURN:
    DESCRIPTION:
    NOTES: 
    DEBUG:
    FUTURE:
        1. Create GROUP class to make this clean
    """
    startTime = time.time()
    # Check Python version
    if(sys.version_info[0] < 3):
        exit_w_error("ERROR!!! Runs with python3, NOT {}\n".format(sys.argv[0]))
    classification = pd.read_table("data/mixture_simulation/y.txt", sep=" ",
                                    header=0, names=['classification'])
    data = pd.read_table("data/mixture_simulation/x.txt", sep=" ", header=0,
                           names=["x", "y"])
    ### Get Groups ###
    groupA_x1 = data['x'][:100]
    groupA_x2 = data['y'][:100]
    groupB_x1 = data['x'][100:]
    groupB_x2 = data['y'][100:]
    dataL = [ [groupA_x1, groupA_x2], [groupB_x1, groupB_x2]]

    ### Train ML algorithm - Recall classification based on _2_ vector quantitites ###
    # Here I explicitly solve the equation 0.5 = x*beta, where x = [1, x1, x2]
    # beta = [beta_0, beta_1, beta_2] and 0.5 is the decision boundary. I solve
    # for x2 given x1
    beta = linear_regression(DataL = dataL, Method="Normal")
    beta = linear_regression(DataL = dataL, Method="QR")
    minVal = -3
    maxVal = 4
    iterations = 100
    x1V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # Vector
    x2V = (0.5 - beta[0] - x1V * beta[1])/beta[2]
    plot_data(ScatterDataL = dataL, LineDataL = [x1V, x2V] )
    
    ## Try nearest neighbor -- Super expensive ##
    x1V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # Vector
    x2V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # Vector
    x2Trans = []    # Get values of [x1, x2] 
    boundary = []
    matrix = np.zeros([x1V.shape[0], x1V.shape[0]], dtype=np.int32)
    for i in range(len(x1V)):
        for j in range(len(x2V)):
            x1 = x1V[i]
            x2 = x2V[j]
            matrix[i,j] = get_N_nearest_neighbors_votes(DataL=dataL, N=15, Pos=[x1,x2])
            curGroup = int(matrix[i,j])
            if(j!=0 and j!=matrix.shape[1]):
                if(int(prevGroup) != int(curGroup)):
                    boundary.append([x1V[i],x2V[j]])
                prevGroup = curGroup
            else:
                prevGroup = curGroup
                
            
            
    ### Get NN line - Find position where transition from one group to another occurs ###
    ### Convert boundary to format that can be used by plot_data ###
    boundary = order_points(boundary)
    boundary = np.asarray(boundary)
    boundary = np.swapaxes(boundary,1,0)
    plot_data(ScatterDataL = dataL, LineDataL = boundary)
    
    ### Output data for diagnostics ###
    fout = open("tmp2.txt", "w+")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            fout.write("{} {} {}\n".format(x1V[i], x2V[j], matrix[i,j]))
    fout.close()
    
    
    

    print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))
    return 0

if __name__ == "__main__":
    main()
