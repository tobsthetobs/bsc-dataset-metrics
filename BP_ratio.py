# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
# Function to iterate over the EuRoC dataset specifically. This dataset is shot using a monochrome camera so will use the threshold_image() function to gather metrics
def load_euroc_dataset(supress_output: bool):
    # Setup directories using os
    img_folder = 'EuRoC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)

    # Setup empty lists to store data
    data_mean = [[], []]
    data_BR = [[], []]
    axis = 0
    counter = 0
    sum = 0
    delta_m = 0
    delta_b = 0
    prev_m = 0
    prev_b = 0

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
            
            # Extract largest change in pixel ratio and mean intensity of image:
            if counter == 0:
                prev_m = mean
                prev_b = BP/DP
                delta_m = 0
            if (delta_m < abs(mean - prev_m)):
                delta_m = abs(mean - prev_m)
                prev_m = mean
            if (delta_b < abs(BP/DP - prev_b)):
                delta_b = abs(BP/DP - prev_b)
                prev_b = BP/DP
        sum += counter
        print("Scanning next folder current total of images processed: ", sum)
        counter = 0 
        axis += 1
    return data_BR, data_mean, (delta_b, delta_m)

# Function to iterate over the AQUALOC dataset specifically, photos are already in gray scale.
# Function calculates mean image intensity, doesnt caclulate values from binary image.
def load_aqualoc_dataset(supress_output: bool, COLORSPACE: str):
    # Setup directories using os 
    img_folder = 'AQUALOC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)
    
    # Setup lists to store data
    data_mean = []
    data_BR = []
    counter = 0
    sum = 0
    delta_m = 0
    delta_b = 0
    prev_m = 0
    prev_b = 0
    
    # Iterate over dataset
    for subfolder in data:
        mean_sequence = []
        BR_sequence = []
        for file in os.listdir(os.path.join(dir,subfolder)):
            image = imread(os.path.join(dir, subfolder, file))
            counter += 1
            mean = 0
            
            # This is here for debugging
            if ((counter % 100) == 0) & (not (supress_output)):
                print(counter)
            
            # Check number of image channels, convert to given colorspace if not already a grayscale image then calculate mean intensity and binary image ratio.
            if check_image_dim(image):
                BP, DP, _ = threshold_image(image, False)
                BR_sequence.append(BP/DP)
                im_CS = load_to_colorspace(image, COLORSPACE)
                mean = np.mean(im_CS)
                mean_sequence.append(mean)
            else:
                BP, DP, _ = threshold_image(image, False)
                BR_sequence.append(BP/DP)
                mean = np.mean(image)
                mean_sequence.append(mean)
                
            # Extract largest change in pixel ratio and mean intensity of image:
            if counter == 0:
                prev_m = mean
                prev_b = BP/DP
                delta_m = 0
            if (delta_m < abs(mean - prev_m)):
                delta_m = abs(mean - prev_m)
                prev_m = mean
            if (delta_b < abs(BP/DP - prev_b)):
                delta_b = abs(BP/DP - prev_b)
                prev_b = BP/DP  
        data_BR.append(BR_sequence)
        data_mean.append(mean_sequence)
        sum += counter
        print("Scanning next folder current total of images processed: ", sum)
        counter = 0
    return data_BR, data_mean, (delta_b, delta_m)

# Function to load and process AURORA datset from folder where raw images are extracted.
def load_aurora_dataset(supress_output: bool):
    return 0

## Section for algorithmic functions
# Make function to process image to correct HSV / YUV color space
def load_to_colorspace(image, COLORSPACE: str):
    assert COLORSPACE == 'HSV' or COLORSPACE == 'YUV' or COLORSPACE == 'GRAY', f'ERROR: Wrong COLORSPACE name selected. COLORSPACE should be one of "HSV", "YUV" or "GRAY"'
    if COLORSPACE == 'HSV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        # Return only value channel
        return img[:,:,2]
    elif COLORSPACE == 'YUV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # Return only luminace channel
        return img[:,:,0]
    elif COLORSPACE == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        

# Create thresholded picture. To take either 0 or 255 in pixel value.
def threshold_image(image, include: bool):
    # Initialize locals:
    brightPixCounter = 0
    dimPixCounter = 0

    if check_image_dim(image):
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

def check_image_dim(image):
    if image.ndim > 2:
        return True
    else:
        return False