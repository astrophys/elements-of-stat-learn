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
from bayes import bayes_classifier
from bayes import plot_bivariate_gaussian

def main():
    """
    ARGS:
    RETURN:
    DESCRIPTION:
    NOTES: 
        1. Data used in the program was taken from Hastie's website for 
           Elements of Statistical Learning. Here is his description
           of how it was created (p 17): 
                'First we generated 10 means mk from a bivariate Gaussian distribution
                 N((1, 0)T , I) and labeled this class BLUE. Similarly, 10 more were
                 drawn from N((0,1)T,I) and labeled class ORANGE. Then for each class
                 we generated 100 observations as follows: for each observation, we
                 picked an mk at random with probability 1/10, and'then generated a
                 N(mk,I/5), thus leading to a mixture of Gaussian clusters for each
                 class.'
    DEBUG:
    FUTURE:
        1. Create GROUP class to make this clean
        2. Try generating random data the same way Hastie did.
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
    # Group A = Blue,   y <= 0.5
    # Group B = Orange, y >  0.5
    groupA_x1 = data['x'][:100]
    groupA_x2 = data['y'][:100]
    groupB_x1 = data['x'][100:]
    groupB_x2 = data['y'][100:]
    dataL = [ [groupA_x1, groupA_x2], [groupB_x1, groupB_x2]]


    ####################### Least Squares #########################
    beta = linear_regression(DataL = dataL, Method="Normal")
    beta = linear_regression(DataL = dataL, Method="QR")
    ###  Create line to plot
    minVal = -3
    maxVal = 4
    iterations = 100
    # Hastie has 3 dimensions :
    #   1. y    (group)
    #   2. x1V  (x-axis)
    #   3. x2V  (y-axis) 
    x1V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # x-axis,
    # Here I explicitly solve eqn 2.1 / 2.2 : 
    #           \hat{Y} = X^{T} \hat{\beta}
    #           \hat{Y} = [1, x1, x2] [\hat{\beta_{0}}, \hat{\beta}_{1}, \hat{\beta}_{2}]^T
    #       Decision boundary is \hat{Y} = 0.5              
    #           0.5 = \beta_{0} + x1V \beta_{1} + x2V \beta{2}
    #           x2V = (0.5 - \beta_{0} - x1V \beta_{1}) / \beta{2}
    x2V = (0.5 - beta[0] - x1V * beta[1])/beta[2]                    # y-axis
    ### Plot line predicted line (x1V vs. x2V) and scatter plot
    plot_data(ScatterDataL = dataL, LineDataL = [x1V, x2V], Comment = "Hastie Fig 2.1")
    

    ####################### Nearest Neighbors #########################
    # Strategy :
    #   1. Create vectors for graph's cartesian coordinates (i.e. x1V, x2V)
    #   2. Generate a uniform grid
    #   3. Classify each point's group / category on the grid with N nearest neighbors
    #   4. Find the transition points to generate the boundary
    #   5. Plot
    x1V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # x-axis
    x2V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # y-axis 
    boundary = []
    matrix = np.zeros([x1V.shape[0], x1V.shape[0]], dtype=np.int32)  # uniform grid
    for i in range(len(x1V)):
        for j in range(len(x2V)):
            x1 = x1V[i]
            x2 = x2V[j]
            # Computationally expensive and scales poorly
            matrix[i,j] = get_N_nearest_neighbors_votes(DataL=dataL, N=15, Pos=[x1,x2])
            curGroup = int(matrix[i,j])
            if(j!=0 and j!=matrix.shape[1]):
                # Find position where transition from one group to another occurs
                if(int(prevGroup) != int(curGroup)):
                    boundary.append([x1V[i],x2V[j]])
                prevGroup = curGroup
            else:
                prevGroup = curGroup
    # Sort boundary so itcan be used by plot_data 
    boundary = order_points(boundary)
    boundary = np.asarray(boundary)
    boundary = np.swapaxes(boundary,1,0)
    plot_data(ScatterDataL = dataL, LineDataL = boundary, Comment = "Hastie Fig 2.2")
    
    ### Debug bivariate_gaussian()
    #plot_bivariate_gaussian(Mu1=0, Mu2=1)
    #plot_bivariate_gaussian(Mu1=1, Mu2=0)


    ####################### Bayes Classifier #########################
    ### Do bayes classifier using our exact knowledge of how the distribution was drawn ###
    ### See ESL by Hastie for further details ###
    boundary = bayes_classifier([-4, 4])
    boundary = order_points(boundary)
    boundary = np.asarray(boundary)
    boundary = np.swapaxes(boundary,1,0)
    plot_data(ScatterDataL = dataL, LineDataL = boundary, Comment = "Hastie Fig 2.5")

    ### Output data for diagnostics ###
    #fout = open("tmp2.txt", "w+")
    #for i in range(matrix.shape[0]):
    #    for j in range(matrix.shape[1]):
    #        fout.write("{} {} {}\n".format(x1V[i], x2V[j], matrix[i,j]))
    #fout.close()
    
    
    

    print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))
    return 0

if __name__ == "__main__":
    main()
