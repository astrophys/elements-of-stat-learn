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


def read_zipcode_data(Path=None):
    """Reads zipcode data. 

    Args:
        None.

    Returns:
        A list of IMAGES

    Raises:
        ValueError  :  when line has incorrect number of elements
    """
    imageL = []
    fin = open(Path, "r")
    width = 16  # image width
    height= 16  # image height
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
    return(imageL)


def main():
    """
    Prints to stdout whether two files are identical within the tolerance

    Args:
        None.

    Returns:
        None
    """
    trainPath = "data/zipcode/zip.train"
    imageL = read_zipcode_data(trainPath)
    sys.exit(0)

if __name__ == "__main__":
    main()
