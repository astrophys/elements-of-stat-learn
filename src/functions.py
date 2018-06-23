from error import exit_w_error
import numpy as np

def order_points(Points = None):
    """ 
    ARGS:
        Points : List or Numpy Array of 2-D points.
    RETURN:
    DESCRIPTION:
        Sometimes when you are finding the boundary between two 
        groups there is multiple boundary points in one column - j.
        When plotted you get a stupid oscillation due to the points
        being misordered.
    
        This function checks for misordered points by at the
        euclidian distance between points. Given a point in a list, it
        orders the _next_ point based on the minimum euclidean dist.

        WARNING : This function can be fooled if the misordering occurs
        on the first point.
    NOTES: 
        Sadly this scales as O(N**2). I could do better but I don't care right now
    DEBUG:
    FUTURE:
    """
    print("Ordering points...")
    ### Get 'typical' delta
    meanDelta = 0
    alreadyAdded = np.zeros(len(Points), dtype=np.int32)  # Flag points already added

    #for i in range(len(Points)):
    #    if(len(Points[i]) != 2):
    #        exit_w_error("ERROR!!! dimension = 2 expected, {}"
    #                     "received".format(len(Points[i])))
    #    if(i != len(Points) - 1):
    #        meanDelta = meanDelta + euclidean_dist(Point1=Points[i], Point2=Points[i+1])
    #meanDelta = meanDelta / len(Points)

    orderedPoints = []
    orderedPoints.append(Points[0])
    alreadyAdded[0] = 1
    for i in range(len(Points)-1):
        #idx = i+1
        #delta = euclidean_dist(Point1=orderedPoints[i], Point2=Points[idx])
        #print("{} {}".format(i,delta))  # Diagnostics..
        delta = 10**9
        for j in range(len(Points)-1):
            if(alreadyAdded[j] == 1):
                continue
            curDelta = euclidean_dist(Point1=orderedPoints[i], Point2=Points[j])
            if(curDelta < delta):
                idx = j
                delta = curDelta
        orderedPoints.append(Points[idx])
        alreadyAdded[idx] = 1

    #for i in range(len(orderedPoints)-1):
    #    delta = euclidean_dist(Point1=orderedPoints[i], Point2=orderedPoints[i+1])
    #    print("{} {}".format(i,delta))  # Diagnostics..
    return(orderedPoints)
    
        

def euclidean_dist(Point1 = None, Point2 = None):
    """ 
    ARGS:
        Point1 : 1D np array or list
        Point2 : 1D np array or list
    RETURN:
        Euclidean distance
    DESCRIPTION:
    NOTES: 
    DEBUG:
    FUTURE:
    """
    if(len(Point1) != len(Point2)):
        exit_w_error("ERROR!!! Dimension mismatch! {} != "
                     "{}\n".format(len(Point1), len(Point2)))
    dist = 0
    for i in range(len(Point1)):
        dist = dist + (Point1[i] - Point2[i])**2
    dist = np.sqrt(dist)
    return(dist)
