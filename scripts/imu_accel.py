import pandas as pd 
import tkinter as tk
import numpy as np
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

# pass a csv on function call or omit to select using explorer
def create_aqualoc_imu_dataframe(**kwargs):
    title = ""
    select = True
    for k,v in kwargs.items():
        if k == "csv":
            df = pd.read_csv(v)
            df = df.iloc[:,4:7]
            select = False
        if k == "title":
            title = v
    if select:
        csv = select_file()
        df = pd.read_csv(csv)
        df = df.iloc[:,4:7]

    # make data list
    data = []
    data.append(df.diff().max())
    data.append(df.mean())
    data.append(df.var())
    df_res = pd.DataFrame(data,index=['Largest difference', 'Mean', 'Var'])
    styles = [dict(selector="caption", props=[("text-align", "center"), ("font-size", "150%")])]
    df_res = df_res.style.set_caption(title).set_table_styles(styles)
    df.plot(kind='line', title=title)
    return df_res

    
    