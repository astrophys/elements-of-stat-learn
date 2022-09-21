# Author : Ali Snedden
# Date   : 9/7/2022
# License: MIT
"""My first naive neural network to classify numbers scanned from USPS zipcodes
"""
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from error import exit_w_error


class IMAGE:
    """
    class to hold the image truth value and the image array
    """
    def __init__(self, Value=None, Matrix=None):
        """Initialize IMAGE object
        
        Args:
            Value: int, contains truth value in image
            Array: np.array, contains actual image data

        Returns:
            IMAGE object

        Raises:
        """
        self.val = Value
        self.mat = Matrix


def plot_image(Image=None):
    """Plot IMAGE object

    Args:
        Image : IMAGE object

    Returns:
        N/A

    Raises:
    """
    plt.clf()
    plt.imshow(Image.mat, cmap='Greys', interpolation='None')
    plt.title(Image.val)
    plt.show()


def read_zipcode_data(Path=None, Truncate=None):
    """Reads zipcode data. 

    Args:
        Path     : Path to data
        Truncate : int, number of lines to read in file

    Returns:
        A list of IMAGES

    Raises:
        ValueError  :  when line has incorrect number of elements
    """
    imageL = []
    fin = open(Path, "r")
    width = 16  # image width
    height= 16  # image height
    n = 0
    for line in fin:
        lineL = line.split()
        if(len(lineL) != 257):
            raise ValueError("ERROR!!! Expecting length of split line to be 257 elements")
        # Truth value is first number
        val = int(float(lineL[0]))
        matrix = np.zeros([width,height])
        i = 0
        j = 0
        for k in range(1, len(lineL)):
            if((k-1)%16 == 0 and k > 2):
                i += 1
                j = 0
            matrix[i,j] = float(lineL[k])
            j += 1
        image = IMAGE(Value = val, Matrix = matrix)
        imageL.append(image)
        ### Truncate for 
        if(Truncate is not None and n > Truncate):
            break
        n += 1          # Number of lines read
    return(imageL)


def sigmoid(Z):
    """
    Compute the sigmoid function on Z

    Args:
        Z : float 

    Returns:
        float, value of sigmoid(Z)

    Raises: 
    """
    return(1/(1+np.exp(-Z)))


def forward_propagation(ActV = None, WeightML = None, BiasVL = None):
    """
    This does forward propagation 

    Args:
        ActV    : numpy float vector, Initial activation values, 
        WeightML: list of weight matrices, ORDER MATTERS. 
        BiasVL  : list of bias vectors, ORDER MATTERS.

    Returns:
        Activation values of final nodes

    Raises: 
    """
    if(len(WeightML) != len(BiasVL)):
        raise ValueError("ERROR!!! Number of weight matrices to equal number of bias"
                         "vectors")
    for i in range(len(WeightML)):
        if(i==0):
            tmpV = np.dot(ActV, WeightML[i]) + BiasVL[i]
        else:
            tmpV = np.dot(tmpV, WeightML[i]) + BiasVL[i]
        tmpV = sigmoid(tmpV)
    return(tmpV)


def back_propagation(YV = None, LastActV = None, WeightML = None, BiasVL = None):
    """
    This does back propagation, following 9:34 of
    https://www.youtube.com/watch?v=tIeHLnjs5U8

    Args:
        YV      : numpy float vector, truth values
        LastActV: numpy float vector, activation values in last layer
        WeightML: list of weight matrices, ORDER MATTERS. 
        BiasVL  : list of bias vectors, ORDER MATTERS.

    Returns:
        Activation values of final nodes

    Raises: 
    """
    nW = 0         # Number of entries for weights
    nB = 0         # Number of entries for bias 
    for wM in WeightML:
        nW += len(wM)
    for bV in BiasVL:
        nB += len(bV)
    nablaCV = np.zeros(nW+nB)  # Gradient of cost function
    # Now compute gradient for Weights 
    # --> count down, called BACK-propagation for a reason
    for i in range(len(WeightML)-1,-1,-1):
        if(i == len(WeightML)-1):
           
            
        # Each element in wM will have an entry in nablaCV
        wM = WeightML[i]        # rows are previous layers, columns are forward layers
        # Rows, left hand
        for j in range(wM.shape[0]):
            for k in range(wM.shape[1]):
                
        


def compute_cost_function(WeightML = None, BiasVL = None, ImageL = None):
    """
    This computes cost function

    Args:
        WeightML: list of weight matrices, ORDER MATTERS.
        BiasVL  : list of bias vectors, ORDER MATTERS.
        ImageL  : List of IMAGES

    Returns:
        Activation values of final nodes

    Raises: 
    """
    cost = 0
    nVal=10
    for image in ImageL:
        yV= np.zeros(nVal)
        val = image.val
        yV[val] = 1
        a0= image.mat.flatten()     # Initial activation
        a = forward_propagation(ActV=a0, WeightML=WeightML, BiasVL=BiasVL)
        c = np.dot(yV-a, yV-a)      # Partial cost function
        cost += c
    return(cost)
        

def train(ImageL = None):
    """
    This trains the model computes cost function

    Args:
        ImageL  : List of IMAGES

    Returns:
        Activation values of final nodes

    Raises: 
    """
    # Initialize RANDOM weight vectors, with one hidden layer
    w0M = np.random.random([256,16])
    w1M = np.random.random([16,10])
    # Initialize RANDOM bias vectors, with one hidden layer
    b0V = np.random.random([16])
    b1V = np.random.random([10])
    costPrev = 0 
    cost = 1
    thresh = 10**-4
    while (cost - costPrev > thresh):
        costPrev = cost
        # Compute total cost
        cost = compute_cost_function(WeightML=[w0M,w1M], BiasVL=[b0V,b1V], ImageL=imageL)
        


def main():
    """
    Prints to stdout whether two files are identical within the tolerance

    Args:
        None.

    Returns:
        None
    """
    trainPath = "data/zipcode/zip.train"
    imageL = read_zipcode_data(Path=trainPath, Truncate=500)
    #miniL  = imageL[0:200]       # Mini-batch, not so random
    train(ImageL = imageL)
    print(cost)
    

    sys.exit(0)

if __name__ == "__main__":
    main()
