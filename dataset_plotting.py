# Import statements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from matplotlib.pyplot import imshow


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

# Function to plot histograms for EuRoC
def create_histogram_euroc(data_tuple):
    data_br, data_m, _ = data_tuple
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
    
# Function to plot histograms for AQUALOC
def create_histogram_aqualoc(data, title: str):
    f, ax = plt.subplots(len(data))
    for i in range(len(data)):
        increment = i+1
        sns.histplot(data[i], bins=feedman_bins(data[i]), kde=True, ax=ax[i])
        ax[i].set_title(title + "  " +  str(increment))
    # plt.tight_layout()
    plt.figure().set_figheight(len(data)*30)
    plt.figure().set_figwidth(len(data)*10)
    plt.show()

# Function used for printing all metrics on downloaded EuRoC dataset
def print_euroc_metrics(data_tuple, include: bool):
    data_br, data_m, delta = data_tuple
    
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
    print("Largest delta: ", delta)

# Function used to return all metrics for downloaded EuRoC dataset,
# returns a tuple with 4 values (br_comb, var_br_comb, m_comb, var_m_comb) which corresponds to: (brightness ratio combined, brightness ratio variance, mean image intensity mean, mean image intensity variance)
# if include is passed as True function will then pass 2 arrays containing the metrics for each cam.
def euroc_metrics(data_tuple, include: bool):
    data_br, data_m, delta = data_tuple
    
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
    return (br_comb, var_br_comb, m_comb, var_m_comb, delta)