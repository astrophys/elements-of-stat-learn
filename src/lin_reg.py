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

    ##### Note I compared my 'beta' to what R gets and I'm spot on #####
    ### This was with using surname.score = surname.frac, out of date now ###
    #x = read.table("R/xtrain.txt", sep=",", header=FALSE)
    #y = read.table("R/ytrain.txt", header=FALSE)
    #df = cbind(x,y)
    #colnames(df) = c("x1","x2","y")
    #model = lm(formula = df$y ~ df$x1 + df$x2)
    # Returns:
    # Coefficients:
    # (Intercept)        df$x1        df$x2  
    #      0.5540       0.3968       0.6398  
    # In Python, I get beta = 0.55395396 0.39679332 0.63978026


    # yPred = np.dot(np.dot(Q, Qt), yTrain) is equivalent to np.dot(xTrain,beta)
    # yPred = np.dot(data[:,0:3], beta)     ###
    #yPred = np.dot(xTest, beta)
    #### MISTAKE HERE!! I'm not sure what I should pick as the threshold for classification ###
    ### yPred > 0.5 is Hispanic and yPred < 0.5 is non-hispanic ###
    ### Big question : How do I choose threshold? ####
    #for idx in range(len(yPred)):
    #    ### Non-hispanic ###
    #    if(yPred[idx] >= beta[0]):     ## Remove intercept ?? Should i do??
    #        yPred[idx] = 0
    #    ### Hispanic ###
    #    else:
    #        yPred[idx] = 1
    #
    #### Check accurracy ###
    #falsePos = 0
    #falseNeg = 0
    #for j in range(len(yPred)):
    #    if(yPred[j] == 0 and yTest[j] == 1):
    #        falseNeg = falseNeg + 1
    #    if(yPred[j] == 1 and yTest[j] == 0):
    #        falsePos = falsePos + 1
    #print("False pos = {} = {:<5.2f} %".format(falsePos, falsePos * 100/ float(len(yPred) )))
    #print("False neg = {} = {:<5.2f} %".format(falseNeg, falseNeg * 100/ float(len(yPred) )))
    #print("Number of Misclassifcations : {} = {:<5.2f} %".format(falseNeg + falsePos,
    #        (falsePos + falseNeg) * 100 / float(len(yPred))))

    #
    #### Output Vector Quantities to train on w/o having to redo tagging etc ###
    #fout = open("output/vector_quantities.txt", "w+")
    #for y in yPred:
    #    fout.write("{}\n".format(y))
    #fout.close()
