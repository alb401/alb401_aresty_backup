import pandas as pd
import numpy as np
import holoviews as hv
import bokeh as bk
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Constants --------------------------------------------------------------------------------

directory = '/home/alec/Documents/2024-2025/Aresty/python/092324/data'

LO_TEMP = 2
HI_TEMP = 300
NUM_PTS = (HI_TEMP-LO_TEMP)*10 + 1

WINDOW_LENGTH = 50; # Window length for Savitsky-Golay filter

# Functions --------------------------------------------------------------------------------

def readFiles(directory):
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                newpath = directory + '/' + file
                df = pd.read_csv(newpath)

                process(df)


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def plot(df):
    return 0

def process(df):
    df = df.iloc[::-1] # Reverses dataset from decreasing to increasing temp
    df.drop_duplicates(inplace = True) # Removes all duplicates from dataset
    df = df.to_numpy() # Converts dataframe to numpy array for interpolation

    lin_temps = np.linspace(LO_TEMP, HI_TEMP, NUM_PTS)
    lin_res = np.interp(lin_temps, df[:,0], df[:,1], period=None)

    lin_res = savgol_filter(lin_res, WINDOW_LENGTH, 2)
    lin_res = np.array(lin_res) 

    # First Derivative
    der1_res = np.gradient(lin_res, lin_temps, edge_order=2)
    der1_res = savgol_filter(der1_res, 2*WINDOW_LENGTH, 2)

    # Second Derivative
    der2_res = np.gradient(der1_res, lin_temps, edge_order=2)
    # der2_res = np.diff(lin_res, )
    der2_res = savgol_filter(der2_res, 2*WINDOW_LENGTH, 2)

    # Natural Log
    ln1_res = np.log(lin_res);

    fig, axs = plt.subplots(2, 2)
    # fig.suptitle(material_name) 

    axs[0, 0].plot(lin_temps, lin_res, 'tab:red')
    axs[0, 0].set(xlabel='T(K)', ylabel='$\\rho_{xx}$', title="Resistivity")

    axs[0, 1].plot(lin_temps, der1_res, 'tab:green')
    axs[0, 1].set(xlabel='T(K)', ylabel='$\\frac{d}{dt}(\\rho_{xx})$', title="First Derivative")

    axs[1, 0].plot(lin_temps, der2_res, 'tab:blue')
    axs[1, 0].set(xlabel='T(K)', ylabel='$\\frac{d^2}{dt^2}(\\rho_{xx})$', title="Second Derivative")

    axs[1, 1].plot(lin_temps,  ln1_res, 'tab:orange')
    axs[1, 1].set(xlabel='T(K)', ylabel='$ln(\\rho_{xx})$', title="Natural Log")
    plt.show()

def myfunc(data):
    temps = [] # Temperatures
    res = [] # Resistivities

def main():
    readFiles(directory)
    


    
