# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from imageio import imread, imwrite
from pylab import *
from skimage.util import img_as_ubyte, img_as_float
from matplotlib.pyplot import imshow
from copy import copy

# Read image and initialize variables
im = imread('Datasets/EuRoC/cam0/1403638127245096960.png')

# Make function to iterate over the EuRoC dataset specifically.
def load_euroc_dataset():
    # Setup directories using os
    img_folder = 'Datasets/EuRoC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + img_folder
    data = os.listdir(dir)
    
    # Setup empty arrays to store data
    data_mean = [[],[]]
    data_BR = [[],[]]
    axis = 0
    counter = 0
    
    # Iterate over dataset
    for subfolder in data:
        for file in os.listdir(os.path.join(dir, subfolder)):
            image = imread(os.path.join(dir, subfolder, file))
            counter += 1
            if (counter % 100) == 0:
                print(counter)
            # Dataset images are already gray scale by the looks of it so can run just take mean.
            mean = np.mean(image)
            BP, DP, _ = threshold_image(image, False)
            data_mean[axis].append(mean)
            data_BR[axis].append(BP/DP)
        axis += 1
    return data_BR, data_mean

# Create thresholded picture. To take either 0 or 255 in pixel value. 
def threshold_image(image, include):
    # Initialize locals: 
    brightPixCounter = 0
    dimPixCounter = 0
    
    # Check if image has multiple channels:
    if image.ndim < 1:
        #Convert to gray-scale
        imgs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        imgs = image
    
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

    if include:
        return brightPixCounter, dimPixCounter, imbw 
    return brightPixCounter, dimPixCounter, None
    
     
# Function to print out current processed image.    
def print_out(threshold_tuple):
    BP, DP, im = threshold_tuple
    print("Pixel ratio is: ", BP/DP)
    if im != None:
        print("Number of Bright pixels: ",BP)
        print("Number of Dim Pixels: ",DP)
        print("Total number of pixels in image: ", im.size)
        print("Percentage of Bright Pixels: ", BP/im.size * 100)
        print("Percentage of Dim Pixels: ", DP/im.size * 100)
        imshow(im, cmap='gray')
        title('Thresholded image')
        show()

def create_histogram(data_tuple):
    data_br, data_m = data_tuple
    hist(data_m[0])
    show()