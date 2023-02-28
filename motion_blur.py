# Import statements
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from imageio import imread, imwrite
import BP_ratio as bp

# Setup dataset folder path:
dataset_folder = 'Datasets'

# Function used to calculate variance of the laplacian of a given image
def motion_blur_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Load AQUALOC dataset and calculate variance in laplacian then return mean of all, and if output is not suppressed will return 
def load_aqualoc_dataset(supress_output: bool):
    # Setup directories using os 
    img_folder = 'AQUALOC/'
    cur_dir = os.getcwd()
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)
    
    # Setup lists and variables
    res = []
    counter = 0
    sum = 0
    
    for subfolder in data: 
        for file in os.listdir(os.path.join(dir,subfolder)):
            res.append(motion_blur_laplacian(bp.load_to_colorspace(imread(os.path.join(dir, subfolder, file)), "GRAY")))
            counter += 1
        sum += counter
        print("Scanning next folder current total of images processed: ", sum)
        counter = 0
    if not supress_output:
        return np.mean(res), res 
    else:
        return np.mean(res), 0