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

    ### Train ML algorithm - Recall classification based on _2_ vector quantitites ###
    # Here I explicitly solve the equation 0.5 = x*beta, where x = [1, x1, x2]
    # beta = [beta_0, beta_1, beta_2] and 0.5 is the decision boundary. I solve
    # for x2 given x1

    ####################### Least Squares #########################
    beta = linear_regression(DataL = dataL, Method="Normal")
    beta = linear_regression(DataL = dataL, Method="QR")
    ###  Create line to plot
    minVal = -3
    maxVal = 4
    iterations = 100
    # Use x1V and x2V instead of x and y to follow Hastie's convention 
    x1V = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)  # x-axis,
    x2V = (0.5 - beta[0] - x1V * beta[1])/beta[2]                    # y-axis
    ### Plot line and scatter plot
    plot_data(ScatterDataL = dataL, LineDataL = [x1V, x2V], Comment = "Hastie Fig 2.1")
    
    ####################### Nearest Neighbors #########################
    # Super computationally expensive 
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
    plot_data(ScatterDataL = dataL, LineDataL = boundary, Comment = "Hastie Fig 2.2")
    
    ### Debug bivariate_gaussian()
    #plot_bivariate_gaussian(Mu1=0, Mu2=1)
    #plot_bivariate_gaussian(Mu1=1, Mu2=0)

    ### Do bayes classifier using our exact knowledge of how the distribution was drawn ###
    ### See ESL by Hastie for further details ###
    boundary = bayes_classifier([-4, 4])
    boundary = order_points(boundary)
    boundary = np.asarray(boundary)
    boundary = np.swapaxes(boundary,1,0)
    plot_data(ScatterDataL = dataL, LineDataL = boundary, Comment = "Hastie Fig 2.5")

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
