# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rosbag
from cv_bridge import CvBridge
# import pandas as pd
import os
# import tensorflow as tf
import seaborn as sns
# from tensorflow import keras
import matplotlib.image as mpimg
from imageio import imread, imwrite
from pylab import *
from skimage.util import img_as_ubyte, img_as_float
from matplotlib.pyplot import imshow
from copy import copy


# Read image and initialize variables
im = imread('Datasets/EuRoC/cam0/1403638127245096960.png')
dataset_folder = 'Datasets'


## This section is for functions to load datasets and process accordingly

# Function to iterate over the EuRoC dataset specifically.
def load_euroc_dataset(supress_output: bool):
    # Setup directories using os
    img_folder = 'EuRoC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)

    # Setup empty arrays to store data
    data_mean = [[], []]
    data_BR = [[], []]
    axis = 0
    counter = 0

    # Iterate over dataset
    for subfolder in data:
        for file in os.listdir(os.path.join(dir, subfolder)):
            image = imread(os.path.join(dir, subfolder, file))
            counter += 1
            
            # This is here for debugging
            if ((counter % 100) == 0) & (not (supress_output)):
                print(counter)
            # Dataset images are already gray scale by the looks of it so can run just take mean.
            mean = np.mean(image)
            BP, DP, _ = threshold_image(image, False)
            data_mean[axis].append(mean)
            data_BR[axis].append(BP/DP)
        axis += 1
    return data_BR, data_mean

# Function to iterate over the AQUALOC dataset specifically
# aqualoc is packed as a bag so will use/modify function from https://gist.github.com/wngreene/835cda68ddd9c5416defce876a4d7dd9 to suit this function
def load_aqualoc_dataset(supress_output: bool):
    # Setup directories using os 
    img_folder = 'AQUALOC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)
    
    # Need to find a way to load data from a .bag file
    
    
    return 0

## Section for algorithmic functions

# Make function to process image to correct HSV / YUV color space
def load_to_colorspace(image, COLORSPACE: str):
    assert COLORSPACE == 'HSV' or COLORSPACE == 'YUV', f'ERROR: Wrong COLORSPACE name selected. COLORSPACE should be one of "HSV", "YUV"'
    if COLORSPACE == 'HSV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    elif COLORSPACE == 'YUV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return img

# Create thresholded picture. To take either 0 or 255 in pixel value.
def threshold_image(image, include: bool):
    # Initialize locals:
    brightPixCounter = 0
    dimPixCounter = 0

    # Check if image has multiple channels:
    if image.ndim < 1:
        # Convert to gray-scale
        imgs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        imgs = image

    imbw = imgs > imgs.mean()
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


## Section for utility functions for such as printing information etc.


# Function to print out current processed image. Specifically for use if include = True is passed to function threshold_image
def print_out(threshold_tuple):
    BP, DP, im = threshold_tuple
    print("Pixel ratio is: ", BP/DP)
    if isinstance(im, np.ndarray):
        print("Number of Bright pixels: ", BP)
        print("Number of Dim Pixels: ", DP)
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
def create_histogram_euroc(data_tuple):
    data_br, data_m = data_tuple
    f, ax = plt.subplots(2, 2)
    sns.histplot(data_br[0], bins=feedman_bins(
        data_br[0]), kde=True, ax=ax[0, 0])
    ax[0, 0].set_title("pixel ratio cam0")
    sns.histplot(data_m[0], bins=feedman_bins(
        data_m[0]), kde=True, ax=ax[0, 1])
    ax[0, 1].set_title("Mean image intensity cam0")
    sns.histplot(data_br[1], bins=feedman_bins(
        data_br[1]), kde=True, ax=ax[1, 0])
    ax[1, 0].set_title("pixel ratio cam1")
    sns.histplot(data_m[1], bins=feedman_bins(
        data_m[1]), kde=True, ax=ax[1, 1])
    ax[1, 1].set_title("Mean image intensity cam1")
    plt.tight_layout()
    plt.show()

# Function used for printing all metrics on downloaded EuRoC dataset
def print_euroc_metrics(data_tuple, include: bool):
    data_br, data_m = data_tuple
    
    # Calculate metrics
    br0 = np.mean(data_br[0])
    br1 = np.mean(data_br[1])
    m0 = np.mean(data_m[0])
    m1 = np.mean(data_m[1])
    
    br_comb = (br0 + br1)/2
    m_comb = (m0+m1)/2
    
    br0_var = np.var(data_br[0])
    br1_var = np.var(data_br[1])
    m0_var = np.var(data_br[1])
    m1_var = np.var(data_m[1])
    
    var_br_comb = (br0_var + br1_var)/2
    var_m_comb = (m0_var + m1_var)/2
    
    if include:
        print("Mean of pixel ratio on cam0: ", br0) 
        print("Mean of pixel ratio on cam1: ", br1)
        print("Mean of pixel mean on cam0: ", m0)
        print("Mean of pixel mean on cam1: ", m1)
    
        print("Variance of pixel ratio on cam0: ", br0_var)
        print("Variance of pixel ratio on cam1: ", br1_var)
        print("Variance of pixel mean on cam0: ", m0_var)
        print("Variance of pixel mean on cam0: ", m1_var)
    
    print("Combined pixel ratio of both cams: ", br_comb)
    print("Combined pixel ratio variance on both cams: ", var_br_comb)
    print("Combined mean image intensity of both cams: ", m_comb)
    print("Combined mean image intensity variance of both cams: ", var_m_comb)

# Function used to return all metrics for downloaded EuRoC dataset,
# returns a tuple with 4 values (br_comb, var_br_comb, m_comb, var_m_comb) which corresponds to: (brightness ratio combined, brightness ratio variance, mean image intensity mean, mean image intensity variance)
# if include is passed as True function will then pass 2 arrays containing the metrics for each cam.
def euroc_metrics(data_tuple, include: bool):
    data_br, data_m = data_tuple
    
    # Calculate metrics
    br0 = np.mean(data_br[0])
    br1 = np.mean(data_br[1])
    m0 = np.mean(data_m[0])
    m1 = np.mean(data_m[1])
    
    br_comb = (br0 + br1)/2
    m_comb = (m0+m1)/2
    
    br0_var = np.var(data_br[0])
    br1_var = np.var(data_br[1])
    m0_var = np.var(data_br[1])
    m1_var = np.var(data_m[1])
    
    var_br_comb = (br0 + br1)/2
    var_m_comb = (m0_var + m1_var)/2
    
    if include:
        means = [br0, br1, m0, m1]
        vars = [br0_var, br1_var, m0_var, m1_var]
        return (br_comb, var_br_comb, m_comb, var_m_comb), means, vars
    return (br_comb, var_br_comb, m_comb, var_m_comb)

