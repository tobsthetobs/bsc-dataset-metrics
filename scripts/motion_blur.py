# Import statements
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from imageio import imread, imwrite
import BP_ratio as bp
from tkinter import filedialog
import tkinter as tk

# Setup dataset folder path:
dataset_folder = 'Datasets'

def select_dir():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()
    return dir_path

# Function used to calculate variance of the laplacian of a given image
def motion_blur_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Load AQUALOC harbor dataset and calculate variance in laplacian then return resulting list
def load_aqualoc_dataset():
    # Setup directories using os 
    img_folder = 'AQUALOC/'
    cur_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    dir = cur_dir + "/" + dataset_folder + "/" + img_folder
    data = os.listdir(dir)
    
    # Setup lists and variables
    res = []
    counter = 0
    sum = 0
    
    # Iterate over folders
    for subfolder in data: 
        placeholder = []
        for file in os.listdir(os.path.join(dir,subfolder)):
            placeholder.append(motion_blur_laplacian(bp.load_to_colorspace(imread(os.path.join(dir, subfolder, file)), "GRAY")))
            counter += 1
        sum += counter
        res.append(placeholder)
        print("Scanning next folder current total of images processed: ", sum)
        counter = 0
        
    return res

# Load downloaded EuRoC dataset and calculate variance in laplacian then return resulting list
def load_euroc_dataset(select: bool):
    if select:
        dir = select_dir()
        data = os.listdir(dir)
    else:
        # Setup directories using os 
        img_folder = 'EuRoC/'
        cur_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        dir = cur_dir + "/" + dataset_folder + "/" + img_folder
        data = os.listdir(dir)
    
    # Setup lists and variables
    res = []
    counter = 0
    sum =  0
    
    # Iterate over folders
    for subfolder in data: 
        placeholder = []
        for file in os.listdir(os.path.join(dir,subfolder)):
            placeholder.append(motion_blur_laplacian(bp.load_to_colorspace(imread(os.path.join(dir, subfolder, file)), "GRAY")))
            counter += 1
        sum += counter
        res.append(placeholder)
        print("Scanning next folder current total of images processed: ", sum)
        counter = 0
        
    return res


