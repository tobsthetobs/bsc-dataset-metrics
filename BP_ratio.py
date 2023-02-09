from imageio import imread, imwrite
from pylab import *
from skimage.util import img_as_ubyte, img_as_float
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import imshow
from copy import copy

# Read image and initialize variables
im = imread('Images/Maden.jpg')




# Create thresholded picture. To take either 0 or 255 in pixel value. 
def threshold_image(image):
    #Initialize locals: 
    brightPixCounter = 0
    dimPixCounter = 0
    #Convert to gray-scale
    imgs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imbw = imgs>imgs.mean()
    imbw = img_as_ubyte(imbw)
    # Flatten image
    imbw_flt = copy(imbw)
    imbw_flt.flatten()
    for i in imbw_flt:
        for j in i:
            if j == 255:
                brightPixCounter += 1
            else:
                dimPixCounter += 1
    return brightPixCounter, dimPixCounter, imbw       

# Function to print out current processed image.    
def print_out(threshold_tuple):
    BP, DP, im = threshold_tuple
    print("Number of Bright pixels: ",BP)
    print("Number of Dim Pixels: ",DP)
    print("Total number of pixels in image: ", im.size)
    print("Percentage of Bright Pixels: ", BP/im.size * 100)
    print("Percentage of Dim Pixels: ", DP/im.size * 100)
    print("Pixel ratio is: ", BP/DP)
    imshow(im, cmap='gray')
    title('Thresholded image')
    show()

print_out(threshold_image(im))