# Import statements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from matplotlib.pyplot import imshow
import entropy as en


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
    bins = round((np.max(data) - np.min(data)) / bin_width)
    return bins

# Function to plot histograms for EuRoC on a single sequence
def create_b_histogram_euroc(data_tuple, number):
    sns.set_palette("pastel")
    data_unpack_br, data_unpack_m, _ = data_tuple
    data_br = data_unpack_br[number]
    data_m = data_unpack_m[number]
    f, ax = plt.subplots(2, 2)
    sns.histplot(data_br[0], bins=feedman_bins(data_br[0]), kde=True, ax=ax[0, 0])
    ax[0, 0].set_title("pixel ratio cam0")
    ax[0, 0].set_xlabel("Brightpixels / Dimpixels")
    sns.histplot(data_m[0], bins=feedman_bins(data_m[0]), kde=True, ax=ax[0, 1])
    ax[0, 1].set_title("Mean image intensity cam0")
    ax[0, 1].set_xlabel("Image mean intensity")
    sns.histplot(data_br[1], bins=feedman_bins(data_br[1]), kde=True, ax=ax[1, 0])
    ax[1, 0].set_title("pixel ratio cam1")
    ax[1, 0].set_xlabel("Brightpixels / Dimpixels")
    sns.histplot(data_m[1], bins=feedman_bins(data_m[1]), kde=True, ax=ax[1, 1])
    ax[1, 1].set_title("Mean image intensity cam1")
    ax[1, 1].set_xlabel("Image mean intensity")
    plt.tight_layout()
    plt.show()
    
# Function to plot histograms for AQUALOC
def create_b_histogram_aqualoc(data, title: str):
    sns.set_palette("pastel")
    f, ax = plt.subplots(len(data))
    for i in range(len(data)):
        increment = i+1
        sns.histplot(data[i], bins=feedman_bins(data[i]), kde=True, ax=ax[i])
        ax[i].set_title(title + "  " +  str(increment))

    f.set_figheight(len(data)*3)
    plt.tight_layout()
    plt.show()

# Function to plot AURORA
def create_b_histogram_aurora(data):
    data = data[0]
    sns.set_palette("pastel")
    sns.histplot(data, bins=feedman_bins(data), kde=True)
    plt.tight_layout()
    plt.show()

# Function used for printing all metrics on downloaded EuRoC dataset
def print_euroc_b_metrics(data_tuple, number, include: bool):
    data_unpack_br, data_unpack_m, delta_unpack = data_tuple
    data_br = data_unpack_br[number]
    data_m = data_unpack_m[number]
    delta = delta_unpack[number]
    
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
    print("Largest delta: (pixel ratio, mean) ", delta)

# Function used to return all metrics for downloaded EuRoC dataset,
# returns a tuple with 4 values (br_comb, var_br_comb, m_comb, var_m_comb) which corresponds to: (brightness ratio combined, brightness ratio variance, mean image intensity mean, mean image intensity variance)
# if include is passed as True function will then pass 2 arrays containing the metrics for each cam.
def euroc_b_metrics(data_tuple, number, include: bool):
    data_unpack_br, data_unpack_m, delta_unpack = data_tuple
    data_br = data_unpack_br[number]
    data_m = data_unpack_m[number]
    delta = delta_unpack[number]
    
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

# Function used for printing all metrics on downloaded AQUALOC dataset
def aqualoc_b_metrics(data_tuple, name: str, include: bool):
    data_br, data_m, delta = data_tuple
    
    # Calculate metrics
    br = 0
    placeholder = []
    for i in range(len(data_br)):
        placeholder.append(np.mean(data_br[i]))
    br = np.mean(placeholder)
    
    br_var = 0
    placeholder.clear()
    for i in range(len(data_m)):
        placeholder.append(np.var(data_br[i]))
    br_var = np.mean(placeholder)
    
    m_m = 0
    placeholder.clear()
    for i in range(len(data_m)):
        placeholder.append(np.mean(data_m[i]))
    m_m = np.mean(placeholder)
    
    m_var = 0
    placeholder.clear()
    for i in range(len(data_m)):
        placeholder.append(np.var(data_m[i]))
    m_var = np.mean(placeholder)
    
    if include:
        print("Mean of pixel ratio on: ", name, br) 
        print("Mean of image intensity on: ", name, m_m)
        print("Variance of pixel ratio on: ", name, br_var)
        print("Variance of image intensity on: ", name, m_var)
        print("Largest delta of all sequences: (pixel ratio, mean)", delta)
    
    return (br, br_var, m_m, m_var)


## Section for motion blur plotting functions
# Function for EuRoC
# Function to plot histograms for AQUALOC
def create_mb_boxplot(data, title: str):
    sns.set_palette("pastel")
    f, ax = plt.subplots(len(data))
    for i in range(len(data)):
        increment = i+1
        sns.boxplot(data[i], ax=ax[i])
        ax[i].set_title(title + "  " +  str(increment))

    f.set_figheight(len(data)*3)
    plt.tight_layout()
    plt.show()
 
def create_mb_histplot(data, title: str):
    sns.set_palette("pastel")
    f, ax = plt.subplots(len(data))
    for i in range(len(data)):
        increment = i+1
        sns.histplot(data[i],bins = feedman_bins(data[i]), ax=ax[i])
        ax[i].set_title(title + "  " +  str(increment))

    f.set_figheight(len(data)*3)
    plt.tight_layout()
    plt.show()

def print_mb_metrics(data):
    return 0


## Section for printing entropy of sequences
# EuRoC
def print_euroc_entropy(data):
    count = 1
    count2 = 1
    switch1 = False
    switch2 = False
    curmax = 0
    curmin = 999999
    prevmax = 0
    prevmin = 0
    index = 0
    index2 = 0
    for i in data:
        if not switch1 and not switch2:
            print("Machine Hall:", count)
        if switch1 and not switch2:
            print("Vicon Room 1: ", count2)
            count2 += 1
        if switch1 and switch2:
            print("Vicon Room 2: ", count2)
            count2 += 1
        for j in i:
            cur = en.calculate_shannon_entropy(j)
            print("Entropy of sequence: ", cur)
            curmax = max(curmax, cur)
            curmin = min(curmin, cur)
            if curmax != prevmax: 
                index = count
            if curmin != prevmin: 
                index2 = count
            prevmax = curmax
            prevmin = curmin
        count += 1
        if count == 6: 
            switch1 = True
        if count == 9:
            switch2 = True
            count2 = 1
    print("Maximum entropy on sequence: ", index, " with the value: ", curmax)
    print("Minimum entropy on sequence: ", index2, " with the value: ", curmin) 