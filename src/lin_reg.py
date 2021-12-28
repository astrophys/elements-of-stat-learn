import numpy as np
from error import exit_w_error

def linear_regression(DataL = None, Method = None):
    """
    ARGS:
        Data   = Training Data
        Method = "QR",  "Normal"
    RETURN:
        beta
    DESCRIPTION:
        This function returns 'beta' as trained on _all_ of data.
        
    NOTES: 
        1. Per Hastie in Statistical Learning : 
            p104 : "However the predictions can be negative or greater than 1 and
            typically some are. This is a consequence of the rigid nature of
            linear regression, especially if we make predictions outside the hull
            of the training data. 
            --> This seems to be likely source of the awkwardness of picking the
                threshold for x * beta.
        2. Doing linear regression with more 3 or more classes is perilous.
           See Figure 4.2
            
    DEBUG:
    FUTURE:
        1. Debug...
        2. Offer switches between using two different xlsx spread sheets
        3. Figure out how to pick threshold... for x*beta
    """
    x1 = []
    x2 = []
    group = []
    for grpIdx in range(len(DataL)):
        if(len(DataL[grpIdx]) != 2):
            exit_w_error("ERROR!!! Expecting only 2 dimenional categoral data")
        x1.extend(DataL[grpIdx][0])
        x2.extend(DataL[grpIdx][1])
        group.extend( [grpIdx] * len(DataL[grpIdx][0]))
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    ### add ones to get intercept / bias ###
    matrix = np.column_stack((np.ones(len(x1)), x1,x2))## Matrix of vector quantities
    group = np.asarray(group)

    print("Training machine learning algorithm : linear regression : ...".format(Method))
    if(Method == "QR"):
        (Q,R) = np.linalg.qr(matrix)
        Rinv = np.linalg.inv(R)
        Qt   = np.transpose(Q)
        beta = np.dot(Rinv, np.dot(Qt, group))

    elif(Method == "Normal"):
        xTxInv =np.linalg.inv(np.dot(matrix.T, matrix))   # Fails b/c singular
        beta = np.dot(np.dot(xTxInv, matrix.T), group)

    else:
        exit_w_error("ERROR!!! Method == {} is invalid".format(Method))

    print("    beta : {}".format(beta))
    return beta

