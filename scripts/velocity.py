# Import statements
import numpy as np
import scipy.integrate 
import pandas as pd
import os
import sys
# Import local code
bachelorpath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
scriptpath = bachelorpath + '/scripts/'
sys.path.insert(0, scriptpath)
import BP_ratio as bp
import dataset_plotting as dp
import motion_blur as mb 
import entropy as en


## Functions to load data
def load_euroc_data():
    euroc_path = bachelorpath + '/ground_truth_datasets/EUROC/'
    data = os.listdir(euroc_path)
    dataframes = []
    for subfolder in data:
        placeholder = 0
        for file in os.listdir(os.path.join(euroc_path, subfolder)):
            placeholder = pd.read_csv(os.path.join(euroc_path, subfolder, file))
            dataframes.append(placeholder.iloc[:,8:11].apply(np.linalg.norm,axis=1))
    return dataframes

def load_aqualoc_data():
    aqualoc_path = bachelorpath + '/imu_datasets/AQUALOC'
    dataframes = []
    for file in os.listdir(aqualoc_path):
        placeholder = pd.read_csv(os.path.join(aqualoc_path,file))
        dt = compute_average_dt(placeholder,"AQUALOC")
        dataframes.append(placeholder.iloc[:,4:7].apply(np.linalg.norm,axis=1))
    return dataframes

## Helper functions
def compute_average_dt(data, dataset: str):
    if dataset == "AQUALOC":
        return data["#timestamp [ns]"].diff().mean() * 10**-9
    return 1

def get_time(df,dt):
    N = len(df.index)
    return np.linspace(0.0, N*dt, N)

def calculate_velocity_np(axis_accel, dt):
    return np.trapz(axis_accel, x=dt)

def calculate_velocity_scipy(accel, time):
    return scipy.integrate.cumtrapz(accel,x=time)

def add_to_dataframe(df, arr, label):
    df[label] = pd.Series(arr)
    return df

def convert_slice_to_ndarry(slice, x):
    return np.array(slice.iloc[:,x].tolist())[:,np.newaxis]

def convert_slice_to_list(slice, x):
    return slice.iloc[:,x].tolist()