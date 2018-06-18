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
import pandas as pd
import sys
import numpy as np
from error import exit_w_error
from plot import plot_data
from lin_reg import linear_regression

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
    beta = linear_regression(DataL = dataL, Method="Normal")
    beta = linear_regression(DataL = dataL, Method="QR")
    minVal = -4
    maxVal = 4
    iterations = 300
    x1 = np.arange(minVal, maxVal, (maxVal - minVal) / iterations)
    x2 = (0.5 - beta[0] - x1 * beta[1])/beta[2]
    

    plot_data(ScatterDataL = dataL, LineDataL = [ [x1, x2] ])
    


if __name__ == "__main__":
    main()
