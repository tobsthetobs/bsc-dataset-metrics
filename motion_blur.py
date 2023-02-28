import numpy as np
import matplotlib.pyplot as plt
import cv2

def motion_blur_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()