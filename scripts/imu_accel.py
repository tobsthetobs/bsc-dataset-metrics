import pandas as pd 
import tkinter as tk
from tkinter import filedialog




# Two helper functions to select csv files or folder
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path
def select_dir():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()
    return dir_path