# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import tensorflow as tf
import seaborn as sns
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
    _, counts = np.unique(imbw_flt, return_counts=True)
    dimPixCounter = counts[0]
    brightPixCounter = counts[1]

    if include:
        return brightPixCounter, dimPixCounter, imbw 
    
    return brightPixCounter, dimPixCounter, False
    
     
# Function to print out current processed image.    
def print_out(threshold_tuple):
    BP, DP, im = threshold_tuple
    print("Pixel ratio is: ", BP/DP)
    if isinstance(im, np.ndarray):
        print("Number of Bright pixels: ",BP)
        print("Number of Dim Pixels: ",DP)
        print("Total number of pixels in image: ", im.size)
        print("Percentage of Bright Pixels: ", BP/im.size * 100)
        print("Percentage of Dim Pixels: ", DP/im.size * 100)
        imshow(im, cmap='gray')
        title('Thresholded image')
        show()

# Helper function to calculate bins for histogram plotting
def feedman_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1/3)
    bins = round((max(data) - min(data)) / bin_width)
    return bins

# Function to plot histograms
def create_histogram(data_tuple):
    data_br, data_m = data_tuple
    f, ax = plt.subplots(2,2)
    sns.histplot(data_br[0], bins=feedman_bins(data_br[0]), kde=True, ax=ax[0,0])
    ax[0,0].set_title("pixel ratio cam0")
    sns.histplot(data_m[0], bins=feedman_bins(data_m[0]), kde=True, ax=ax[0,1])
    ax[0,1].set_title("Mean image intensity cam0")
    sns.histplot(data_br[1], bins=feedman_bins(data_br[1]), kde=True, ax=ax[1,0])
    ax[1,0].set_title("pixel ratio cam1")
    sns.histplot(data_m[1], bins=feedman_bins(data_m[1]), kde=True, ax=ax[1,1])
    ax[1,1].set_title("Mean image intensity cam1")
    plt.tight_layout()
    plt.show()

create_histogram(load_euroc_dataset())