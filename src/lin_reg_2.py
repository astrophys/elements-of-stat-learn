# Author : Ali Snedden
# Date   : 6/27/17
# Purpose: 
#   To understand chapters 2,3,4 and 18 in "Elements of Statistical Learning"
#   by Hastie et al. 
#
#   Conclusion : 
#       I have a pretty decent understanding of linear regression
"""Module that provides useful functions for the computation of linear regression models
"""
import numpy as np
from error import exit_w_error

def linear_regression(X = None, Y = None, XYInteract = False, OnesIncl = False,
                      Method = None
):
    """
    ARGS:
        X          = data      (nMeasurements x nVariable)
        Y          = condition (variable want to predict)
        XYInteract = there is x-y interaction
        OnesIncl   = boolean, : True if ones vector already appended to X
        Method     = 'Normal' : beta = (X^T * X)^{-1} * X^T *  y
                                beta = np.linalg.inv(np.dot(X.T,X)) X.T y # Eqn 2.6 in ESL
                     'QR'     : beta = 
    RETURN:
    DESCRIPTION:
        The point of linear regression is to use some training data to generate 
        predictions of future data. In 'linear' regression, we assume that 
        the coefficients (beta) are linear (note I use '*' to denote matrix
        multiplication).

            Goal: 
                Y_pred = X * beta
    
                where : X      : [N samples, P variables]
                        beta   : [P linear coefficients]
                        Y_pred : [N predicted categories]

        In order to reach the goal, we must use a training set to compute beta.
        This requires minimizing the residuals sum of squares, RSS(beta), e.g. 
            
                RSS(beta) = (Y_train - X_train * beta)^T     * (Y_train- X_train * beta)
                RSS(beta) = (Y_train^T - beta^T * X_train^T) * (Y_train- X_train * beta)

            Minimizing:
                d(RSS(beta))/d(beta) = 0 = -X_train^T * (Y_train - X_train * beta)

        From here there are several ways of solving for beta. You can use the 
        'normal equations' (Hastie eqn 2.5), e.g. 

            Solve for beta:
                beta = (X_train^T * X_train)^{-1} * (X_train^T * Y_train)

        Normal Equations works if X^T * X is invertible (i.e. not singular, det != 0). 
        However if it is singular, one must use QR decomposition. Any Rectangular matrix
        can be decomposed into a Unitary (i.e. Q^{*} * Q = I, {*} = conjugate transpose)
        and upper triangular matrix (which is R)
        
            Starting with RSS(beta) minimization..
                X = QR 
                0 = -(QR)^T * (Y_train - (QR) * beta)
                (QR)^T * (QR) * beta = R^T * Q^T * Y_train  
                R^T  * Q^T * Q * R * beta = R^T * Q^T * Y_train  
                       |_____|
                          |
                       = I b/c unitary
                beta = (R^T * R)^{-1} * R^T * Q^T * Y_train
                beta =  R^{-1} * (R^T)^{-1} *R^T * Q^T * Y_train
                beta =  R^{-1} * Q^T * Y_train

    REFERENCES:
        1. Elements of Statisitcal Learning by Hastie, Chapters 2 & 3.
        2. https://en.wikipedia.org/wiki/QR_decomposition

    DEBUG:
        1. Tested knowledge of lm() in R and linear fitting in Python using ESL.
           NOTE : Linear fitting is named so b/c the coefficients are linear, not
           necessarily the terms!
           Ran x,y = np.random.multivariate_normal(mean, cov, 5000).T, 
           cov = [[81,20],[20,9]], mean = [0,0], seed=42:

           Linear Fit (wo xy-interaction):
                R: 
                    df = read.table("lm_point.dat", sep = "\t", header=TRUE)
                    lm(formula = df$y ~ df$x), 
                    Coefficients:
                    (Intercept)         df$x  
                    0.001488     0.248548  
                    -------------- OR -------------
                    library("caret")
                    lmfit = train(form = y ~ x, data = df, method = "lm")
                    lmfit$finalModel
                Python (this func): 
                    y = 0.248547927215x + 0.00148774715121

           Linear Fit (w/ xy-interaction):
                R : 
                    df = read.table("lm_point.dat", sep = "\t", header=TRUE)
                    lm(formula = df$y ~ df$x + df$x:df$y)  #intercept implied
                    Coefficients:
                    (Intercept)         df$x    df$y:df$x  
                     -0.0150480    0.2485028    0.0008161  
                Python :
                    yhat = 0.000816064802479xy + 0.24850275352x + -0.0150480075827

           Therefor, I understand this quite well and how R deals with formulas.

        2. Found residuals, same as in R: 
                R : 
                    a = lm(df$y ~ df$x + df$x : df$y)
                    sum(resid(a)*resid(a))
                    20447.48
                Python (this func):
                    20447.4830431
        3. predict(a) : Generates the predicted values of y using the model. 
             e.g. yhat = 0.000816064802479xy + 0.24850275352x + -0.0150480075827
                  plugging in the values for x and y in this case. so for
                  [x,y] = [-6.56698893,1.3723568], yhat = -1.654318

    FUTURE:
        1. Figure out how the hell lm() in R determines std. err, t value and 
           Pr(>|t|)
        2. Make it more general so you dont have to know the order of terms, 
           e.g. y = b[0] + b[1]x + b[2]xy vs. b[0] + b[1]xy + b[2]x
        3. Figure out mean squared error : see eqn 2.25 ESL (Got values, not sure
           how to justify the computation)
            see : https://economictheoryblog.com/2016/02/20/rebuild-ols-estimator-manually-in-r/
        4. Debug Method == QR
           --> Check that xyinteract works here
    """
    # If ones already included (for intercept) in X, skip this step
    if(OnesIncl == False):
        length = X.shape[0]         # Add bias/intercept, cols = vars, rows = meas
        ones = np.ones(length)
        # Add intercept by adding a column value == 1 to x, see p45 of Hastie
        X = np.column_stack((ones,X))

    if(XYInteract == True):
        # Fit with x:y interacting term?
        X = np.column_stack((X,X*Y))

    if(Method == "Normal"):
        X2 = np.dot(X.T,X)
        X2inv = np.linalg.inv(X2)
        beta = np.dot(X2inv, np.dot(X.T,Y))

        # compute residuals
        residual  = (Y - np.dot(X,beta))
        residual2 = np.dot(residual.T, residual)
        #print("residual squared= {}".format(residual2))
        
        # Not sure how this gives the std. err. (see summary(lm()))
        # check the diagonals
        # 1 / (5000 - 3.0) * np.dot(residual.T, residual) * np.linalg.inv(np.dot(x.T,x))
    elif(Method == "QR"):
        #(Q,R) = np.linalg.qr(X, mode='complete')  # R is non-invertible here!
        (Q,R) = np.linalg.qr(X)      # Throw in arbitrary np.trans to make R square
        Rinv = np.linalg.inv(R)     # Must be SQUARE matrix here!
        Qt   = np.transpose(Q)
        beta = np.dot(Rinv, np.dot(Qt, Y))
        
    else:
        exit_w_error("ERROR!!! Method == {} invalid option!\n".format(Method))

    return beta         # For 1 input var (e.g. x is 1 dimensional)


def rss(X = None, Y = None, Beta = None):
    """
    ARGS:
        X          = data      (nMeasurements x nVariable)
        Y          = condition (variable want to predict)
    RETURN:
    DESCRIPTION:
    """
