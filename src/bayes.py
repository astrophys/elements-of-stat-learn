import numpy as np
from error import exit_w_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def bayes_classifier(RangeX1=None):
    """
    ARGS:
        RangeX1 = [min, max]
    RETURN:
        A list of points that are approx equal to the orange/blue boundary.
    DESCRIPTION:
    NOTES: 
        Can plot the means in gnuplot, e.g. 
            plot [-4:4][-4:4] "tmp.txt" index 0 using 1:2, "tmp.txt" index 1 using 1:2

        This function is an attempt to generate Fig. 2.5.  I need to use the
        information on p17 and the mixture_simulation/means.txt to get the 
        exact boundary. There is no other way.

    DEBUG:
    FUTURE:
        1. Do adaptive root finding using Newton's method
    """
    print("Running bayes_classifier()...")
    minX1 = RangeX1[0]
    maxX1 = RangeX1[1]
    minX2 = minX1
    maxX2 = maxX1
    tol = 0.01
    dx = 0.1   ### = 0.025 if want _really_ smooth curve
    boundaryPoints = []
    blueMk = []                        # Mean
    orangeMk = []                      # Mean
    percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    ### Draw 10 means for each samples for ###
    #for i in range(10):
    #    blueMk.append(np.random.multivariate_normal(mean = np.asarray([1,0]),
    #                        cov=np.asarray([[1,0],[0,1]])))
    #    orangeMk.append(np.random.multivariate_normal(mean = np.asarray([0,1]),
    #                        cov=np.asarray([[1,0],[0,1]])))

    ### Directly initialize Means from means.txt ###
    blueMk = np.asarray([[-0.253433158653597, 1.74147879876335],
               [ 0.266693178430279, 0.3712341020785], [ 2.09646921174349, 1.23336417257788],
               [ -0.0612727205045234, -0.208679132507905], [ 2.7035408513268, 0.596828323506115],
               [ 2.37721198219787, -1.18641470384923], [ 1.05690759440227, -0.683893937517459],
               [ 0.578883539500997, -0.0683458017784068], [ 0.624252127188094, 0.598738390086286],
               [ 1.67335495316395, -0.289315921722119]])
    orangeMk = np.asarray([[1.19936869234067,0.248408554753682],
               [-0.302561095070758,0.945418958597202],[0.0572723205731644,2.4197271457479],
               [1.32932203024772,0.819225984741309],[-0.0793842405212738,1.61380166597827],
               [3.50792672672612,1.05298629743892],[1.6139228994926,0.671737825311435],
               [1.00753570231607,1.36830712305429],[-0.454621406426687,1.08606972977247],
               [-1.79801804757754,1.92978056103932]])

    ## Get initial seed, if multivalued here, we're screwed.
    pIdx = 0  # Percent Idx
    ### Sweep along horizontal axis ###
    for x1 in np.arange(minX1, maxX1, dx):
        ## output percent complete
        if( abs(x1 - minX1) / (maxX1 - minX1) > percent[pIdx]):
            print("\t{:<3.0f}% Complete".format(percent[pIdx] * 100))
            pIdx = pIdx + 1
        #print("x1 : {}".format(x1))
        
        diffV = np.zeros([len(np.arange(minX2, maxX2, dx))])
        ### Sweep along verticle axis ###
        for i in range(diffV.shape[0]):
            x2 = minX2 + dx * i
            # Find Orange Prob
            probOrange = 0
            for mean in orangeMk:
                pdf = multivariate_normal(mean=mean, cov=[[0.25,0],[0,0.25]]).pdf([x1,x2])
                probOrange = probOrange + pdf
            # Find Blue Prob
            probBlue = 0
            for mean in blueMk:
                pdf = multivariate_normal(mean=mean, cov=[[0.25,0],[0,0.25]]).pdf([x1,x2])
                probBlue = probBlue + pdf
            diff = probBlue - probOrange
            diffV[i] = diff


        # Use cheap newton's like method for finding roots.  Look only at
        # change in sign, want diff close to 0, but not due to the fact 
        # that we are picking a point far away from BOTH the orange and blue
        # centers.
        minI = 0
        for i in range(diffV.shape[0]-1):
            if(diffV[i] * diffV[i+1] < 0):
                # minI = i
                x2 = minX2 + dx * i
                #print("{} {}".format(x1,i))
                boundaryPoints.append([x1,x2])  ### This permits catching multiple roots
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
    #print("WARNING!!!! Ignoring Mu1 and Mu2!!!")
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

