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
    newDF = pd.DataFrame(columns = rawDF.columns)
    trainDF = rawDF[rawDF["train"] == "T"]
    testDF  = rawDF[rawDF["train"] == "F"]


    # Fit linear model : y=lpsa. x=all other vars
    print("{:<10} : {:<10} {:<10}".format("col", "mean", "stdev"))
    for col in newDF.columns:
        # Center to zero, scale by stdev to make unit var
        # Skip categorical vars
        #   --> svi
        #   --> gleason
        if(col == "svi" or col == "gleason" or col == "train"):
            continue
        #newDF[col] = (df[col] - np.mean(df[col]))/np.std(df[col])
        newDF[col] = trainDF[col]/np.std(trainDF[col])
        print("{:<10} : {:<10.3f} {:<10.3f}".format(col, np.mean(trainDF[col]), np.std(trainDF[col])))

    # Compute covariance matrix
    covM = np.zeros([10,10])            # Yuck - hard coded
    i = 0
    j = 0
    for col1 in newDF.columns:
        mu1 = np.mean(newDF[col1])
        j = 0
        for col2 in newDF.columns:
            mu2 = np.mean(newDF[col2])
            covM[i,j] = (np.dot((newDF[col1] - mu1),(newDF[col2] - mu2))) / (newDF.shape[0])
            j = j + 1
        i = i + 1


    print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))
    return 0

if __name__ == "__main__":
    main()
