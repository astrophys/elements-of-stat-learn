import numpy as np
from error import exit_w_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def bayes_classifier(RangeX1=None):
    """
    ARGS:
        RangeX1 = [min, max]
    RETURN:
        A list of points that are approx equal to the orange/blue boundary.
    DESCRIPTION:
    NOTES: 
        This function is an attempt to generate Fig. 2.5.  I need to use the
        information on p17.
    
        Per equation 2.23:
            G^{hat}(x) = G_{k} if Prob(G_{k}|X=x) = max_{g \epsilon G} Prob(g|X=x)

        B/c we know that the two classes were drawn from bivariate probability 
        distribution functions.
            Orange : Prob(Orange) = Gaussian((1,0)^{T}, Identity)
            Blue   : Prob(Blue)   = Gaussian((0,1)^{T}, Identity)

        Thus we can find the exact solution for equation 2.23.
        
        From Wolfram Mathworld, the bivariate normal distribution is : 

        P(x1,x2) = \frac{1}{2 \pi s1 s2 (1 - \rho^{2})} exp(\frac{-z}{2(1-\rho^{2}))
        z        = (x1 - u1)^{2}/s1^{2} - 2 \rho (x1-u1)(x2-u2)/(s1 s2) + (x2-u2)^{2}/(s2^{s})
        rho      = cor(x1,x2)

    DEBUG:
    FUTURE:
        1. Do adaptive root finding using Newton's method
    """
    minX1 = RangeX1[0]
    maxX1 = RangeX1[1]
    minX2 = minX1
    maxX2 = maxX1
    tol = 0.01
    dx = 0.01
    boundaryPoints = []

    ## Get initial seed, if multivalued here, we're screwed.

    for x1 in np.arange(minX1, maxX1, dx):
        ### Use something like Newton's method to iteratively solve for the values
        ### Be careful about multiple roots...
        diffV = np.zeros([len(np.arange(minX2, maxX2, dx))])
        #diffDxV = np.zeros([diffV.shape[0] - 1])    ## d(diff)/dx
        #diffD2xV = np.zeros([diffV.shape[0] - 2])   ## d^2(diff)/dx^2
        #x2Roots = []
        
        # Find multiple minima
        for i in range(diffV.shape[0]):
            x2 = minX2 + dx * i
            probBlue = bivariate_gaussian(X1=x1, X2=x2, Mu1=1, Mu2=0)
            probOrange = bivariate_gaussian(X1=x1, X2=x2, Mu1=0, Mu2=1)
            diff = probBlue - probOrange
            diffV[i] = diff
    
        # Find diff values closest to 0. Ignore possibility of multiple values
        minI = 0
        for i in range(diffV.shape[0]):
            if(abs(diffV[minI]) > abs(diffV[i])):
                minI = i
                x2 = minX2 + dx * i
                #print("{} {}".format(x1,i))
        boundaryPoints.append([x1,x2])

        # Find first derivitive
        #for i in range(diffDxV.shape[0]):
        #    diffDxV[i] = (diffV[i+1] - diffV[i])/dx
        # Find potential roots...
        #for i in range(diffDxV.shape[0]):
        #    if(i >= diffDxV.shape[0] - 1):
        #        continue
        #    # Only true if signs switched...Could do adaptive searching w/ tolerance in future
        #    if(diffDxV[i] * diffDxV[i+1] < 0):
        #        x2 = minX2 + dx * i
        #        x2Roots.append(x2)
        #for x2 in x2Roots:
        #    boundaryPoints.append([x1,x2])
            
        
        # ID local minima
    return(boundaryPoints)
        
         

def bivariate_gaussian(X1 = None, X2 = None, Mu1 = None, Mu2 = None):
    """
    ARGS:
        Mu1 = Mean for var 1
        Mu2 = Mean for var 2
    RETURN:
    DESCRIPTION:
        We assume \sigma1 = 1, \sigma2 = 1 and \rho = ((1 0), (0 1)) per 
        Hastie.

    NOTES: 
        From Wolfram Mathworld, the bivariate normal distribution is : 

        P(x1,x2) = \frac{1}{2 \pi s1 s2 (1 - \rho^{2})} exp(\frac{-z}{2(1-\rho^{2}))
        z        = (x1 - u1)^{2}/s1^{2} - 2 \rho (x1-u1)(x2-u2)/(s1 s2) + (x2-u2)^{2}/(s2^{s})
        rho      = cor(x1,x2) 

        In Hastie's notation, I believe that \rho = 0 b/c his notation of Gaussian((1,0), I)
        I think implies that there is 0 correlation between x1,x2. This interpretation
        seems to be followed by Wolfram's MultinormalDistribution() function's arguments
    DEBUG:
    FUTURE:
    """
    s1  = 1.0
    s2  = 1.0
    mu1 = Mu1
    mu2 = Mu2
    x1  = X1
    x2  = X2

    # I already simplified the computation here by eliminating terms dependent on rho
    z   = (x1 - mu1)**2 / s1**2 + (x2 - mu2)**2 / s2**2
    prob= 1.0 / (2 * np.pi * s1 * s2) * np.exp(-1.0 * z / 2.0)
    return(prob)



def plot_bivariate_gaussian(Mu1 = None, Mu2 = None):
    """
    ARGS:
        Mu1 = Mean for var 1
        Mu2 = Mean for var 2
    RETURN:
    DESCRIPTION:
        This is a pain-in-the-ass. Evidently matplot lib doesn't do interpolation
        between points for us in ax.plot_surface(). You must use ax.plot_trisurf

        Can plot in gnuplot to convince ourselves that this works:
                    splot "biv_gaus.txt" using 1:2:3 with pm3d
    NOTES: 
        See : https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    DEBUG:
    FUTURE:
    """
    print("WARNING!!!! Ignoring Mu1 and Mu2!!!")
    minVal = -5.0
    maxVal =  5.0
    dx = 0.1
    points = np.zeros([(len(np.arange(minVal, maxVal, dx)))**2, 3])
    i = 0
    for x in np.arange(minVal, maxVal, dx):
        for y in np.arange(minVal, maxVal, dx):
            z = bivariate_gaussian(X1 = x, X2 = y, Mu1 = Mu1, Mu2 = Mu2)
            #z = (bivariate_gaussian(X1 = x, X2 = y, Mu1 = 0, Mu2 = 1) -
            #     bivariate_gaussian(X1 = x, X2 = y, Mu1 = 1, Mu2 = 0))
            points[i,:] = np.asarray([x,y,z])
            i = i + 1
    #ax.plot_surface(points[:,0], points[:,1], points[:,2], color='b')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], linewidth=0.2, antialiased=True)
    #ax.plot_trisurf(points[:,0], points[:,1], points[:,2], linewidth=0.2)
    ### Write to file for debugging
    fout = open("biv_gaus.txt", "w+")
    pPrev = 0
    for p in points[:]:
        if(pPrev != p[0]):
            fout.write("\n")
        pPrev = p[0]
        fout.write("{} {} {}\n".format(p[0],p[1],p[2]))

    plt.show()

