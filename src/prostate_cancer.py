# Author : Ali Snedden
# Date   : 3/27/22
# Purpose: 
#   Data got from https://web.stanford.edu/~hastie/ElemStatLearn/
#   Working out prostate cancer example from section 3.2.1
#
# Notes  : 
#   1. Fit Linear model : y=lpsa, x=everything-else
#   2. Standardize data : https://stackoverflow.com/a/8717248/4021436
#       a) Center : subtract mean
#       b) Find variance of each variable
#       c) Divide the shifted (by mean) by the variance
#      
# Questions :  
#   1. How to handle categorical vars
#   #. I understand computing correlatoin
#       a) np.dot(df["lcavol"] - np.mean(df["lcavol"]), df["lweight"] - np.mean(df["lweight"])) / (df.shape[0] * np.std(df["lcavol"]) * np.std(df["lweight"]))
#       #) np.dot(newDF["lcavol"] - np.mean(newDF["lcavol"]), newDF["lweight"] - np.mean(newDF["lweight"])) / newDF.shape[0]
#
#
#
import time
import pandas as pd
import sys
import numpy as np
from error import exit_w_error

def main():
    """
    ARGS:
    RETURN:
    DESCRIPTION:
    NOTES: 
    DEBUG:
    FUTURE:
    """
    startTime = time.time()
    # Check Python version
    if(sys.version_info[0] < 3):
        exit_w_error("ERROR!!! Runs with python3, NOT {}\n".format(sys.argv[0]))
    classification = pd.read_table("data/mixture_simulation/y.txt", sep=" ",
                                    header=0, names=['classification'])
    rawDF = pd.read_table("data/prostate_cancer/data.txt", sep="\t", header=0)
    skipColL= ["pgg45", "lpsa", "gleason", "train"]
    keepColL= [col for col in rawDF.columns if col not in skipColL]
    newDF   = pd.DataFrame(columns = keepColL)       # Empty DF, standardized trainDF here
    trainDF = rawDF[rawDF["train"] == "T"]
    testDF  = rawDF[rawDF["train"] == "F"]


    # Fit linear model : y=lpsa. x=all other vars
    for col1 in newDF.columns:
        # Skip categorical vars
        #   --> svi
        #   --> gleason
        if(col1 in skipColL):
            continue
        else: 
            newDF[col1] = trainDF[col1]


    # Compute covariance matrix
    corrM = np.zeros([newDF.shape[1], newDF.shape[1]])            # Yuck - hard coded
    i = 0
    j = 0
    colL = []
    for col1 in newDF.columns:
        colL.append(col1)
        mu1 = np.mean(newDF[col1])
        j = 0
        for col2 in newDF.columns:
            mu2 = np.mean(newDF[col2])
            # https://en.wikipedia.org/wiki/Correlation
            #. The newDF.shape comes from expectation value of the numerator
            corrM[i,j] = (np.dot((newDF[col1] - mu1),(newDF[col2] - mu2))) / (np.std(newDF[col1]) * np.std(newDF[col2]) * newDF.shape[0])
            j = j + 1
        i = i + 1


    ### Print out Table 3.1 ###
    # print header
    sys.stdout.write("{:<8}".format(" "))
    for c in colL :
        if(c in skipColL):
            continue
        else:
            sys.stdout.write("{:<8}".format(c))
    print("")
    for i in range(corrM.shape[0]):
        sys.stdout.write("{:<8}".format(colL[i]))
        for j in range(corrM.shape[1]):
            if(j < i):
                sys.stdout.write("{:<8.3f}".format(corrM[i,j]))
        print("")

    #
    # Compute correllations more obviously via 
    #       np.dot(trainDF["lcavol"] - np.mean(trainDF["lcavol"]), trainDF["lweight"] - np.mean(trainDF["lweight"])) / (trainDF.shape[0] * np.std(trainDF["lcavol"]) * np.std(trainDF["lweight"]))
    #
    #


    print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))
    return 0

if __name__ == "__main__":
    main()
