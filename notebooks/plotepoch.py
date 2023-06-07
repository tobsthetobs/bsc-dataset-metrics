# Imports
import os
import numpy as np
import sys
import pandas as pd 
import matplotlib.pyplot as plt

curdir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/bsc-dataset-metrics/'

low_entropy = (pd.read_csv(curdir + '/Tensorboard_files/saved/train_low.csv'),pd.read_csv(curdir + '/Tensorboard_files/saved/validation_low.csv'))
high_entropy = (pd.read_csv(curdir + '/Tensorboard_files/saved/train.csv'),pd.read_csv(curdir + '/Tensorboard_files/saved/validation.csv'))

low_entropy[0].plot()


