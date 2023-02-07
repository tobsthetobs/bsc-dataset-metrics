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
brightPixCounter = 0
dimPixCounter = 0

#Convert to gray-scale
gray_image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# Create thresholded picture. To take either 0 or 255 in pixel value. 
imbw = gray_image>gray_image.mean()
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
        
        
print("Number of Bright pixels: ",brightPixCounter)
print("Number of Dim Pixels: ",dimPixCounter)
print("Total number of pixels in image: ", imbw.size)
print("Percentage of Bright Pixels: ", brightPixCounter/imbw.size * 100)
print("Percentage of Dim Pixels: ", dimPixCounter/imbw.size * 100)
print("Pixel ratio is: ", brightPixCounter/dimPixCounter)
imshow(imbw, cmap='gray')
title('Thresholded image')
show()