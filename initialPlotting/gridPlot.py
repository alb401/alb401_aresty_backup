import pandas as pd
import numpy as np
import holoviews as hv
import bokeh as bk
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter
import os

# Statics ################################################################################

material_name = "$Eu_2Ir_2O_7$" 

file_path = '/home/alec/Documents/2024-2025/Aresty/python/092324/data/1_RT_EIO4732.csv'
# file_path = '/home/alec/Documents/2024-2025/Aresty/python/092324/data/2_RT_EIODTO4704.csv'
# file_path = '/home/alec/Documents/2024-2025/Aresty/python/092324/data/3_RT_YIO4749_130nm.csv'

# Font sizes for plots --------------------------------------------------------------------

SMALL_SIZE = 14 MEDIUM_SIZE = 16 BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Function for data normalization
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# Processing #############################################################################

df = pd.read_csv(file_path)

if(df.iloc[0,0] > 30):
    df = df.iloc[::-1] # Reverses dataset from decreasing to increasing temp
df.drop_duplicates(inplace = True) # Removes all duplicates from dataset

df = df.to_numpy() # Converts dataframe to numpy array for interpolation

# Creates evenly spaced values for temperature
lo_temp = 2
hi_temp = 300
num_pts = (hi_temp-lo_temp)*10 + 1

# Create evenly spaced linear interpolation for smoothing and analysis
lin_temps = np.linspace(lo_temp, hi_temp, num_pts)
lin_res = np.interp(lin_temps, df[:,0], df[:,1], period=None)

WINDOW_LENGTH = 100; # Window length for Savitsky-Golay filter

# Savitsky-Golay filter for smoothing
lin_res = savgol_filter(lin_res, WINDOW_LENGTH, 2)
lin_res = np.array(lin_res) 

# First Derivative
der1_res = np.gradient(lin_res, lin_temps, edge_order=2)
der1_res = savgol_filter(der1_res, 2*WINDOW_LENGTH, 2)

# Second Derivative
der2_res = np.gradient(der1_res, lin_temps, edge_order=2)
der2_res = savgol_filter(der2_res, 2*WINDOW_LENGTH, 2)

# Natural Log
ln1_res = np.log(lin_res);

# 1st Derivative of Natural Log
ln1_der1_res = np.gradient(ln1_res, lin_temps, edge_order=2)


# ***  Inverse Temperatures *** -----------------------------------------------------------


# inv_temps = 1/lin_temps
inv_temps = np.linspace(1/300, 1.99666666666667, 599)
# inv_res = np.interp(inv_temps, df[:,0], df[:,1], period=None)
inv_f = scipy.interpolate.interp1d(lin_temps, lin_res, bounds_error = False, fill_value = "extrapolate")
inv_res = inv_f(inv_temps)

# inv_res = savgol_filter(inv_res, WINDOW_LENGTH, 2)
inv_res = np.array(inv_res)

# First Derivative
inv_der1_res = np.gradient(inv_res, inv_temps, edge_order=2)
inv_der1_res = savgol_filter(inv_der1_res, 2*WINDOW_LENGTH, 2)

# Second Derivative
inv_der2_res = np.gradient(inv_der1_res, inv_temps, edge_order=2)
inv_der2_res = savgol_filter(inv_der2_res, 2*WINDOW_LENGTH, 2)

# Natural Log
inv_ln1_res = np.log(inv_res); 

# Plotting #############################################################################

fig, axs = plt.subplots(2, 2)

fig.suptitle(material_name) 
# fig.tight_layout(pad = 5.0)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

axs[0, 0].plot(lin_temps, lin_res, 'tab:red', label="Resistivity")
axs[0, 0].set(xlabel='T(K)', ylabel='$\\rho_{xx}$', title="$\\rho_{xx}$")

axs[0, 1].plot(lin_temps, der1_res, 'tab:green', label="$\\frac{d}{dT}(\\rho_{xx})$")
axs[0, 1].set(xlabel='T(K)', ylabel='$\\frac{d}{dT}(\\rho_{xx})$', title="First Derivative w/ respect to T")

axs[1, 0].plot(lin_temps, der2_res, 'tab:blue', label="$\\frac{d^2}{dT^2}(\\rho_{xx})$")
axs[1, 0].set(xlabel='T(K)', ylabel='$\\frac{d^2}{dT^2}(\\rho_{xx})$', title="Second Derivative w/ respect to T")

axs[1, 1].plot(lin_temps,  ln1_res, 'tab:orange', label="$ln(\\rho_{xx})$")
axs[1, 1].set(xlabel='T(K)', ylabel='$ln(\\rho_{xx})$', title="Natural Log")

axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

# *** Inverse Plotting ***

# Shows actual data in red and extrapolated data in blue
# plt.figure(1)
# plt.plot(lin_temps, lin_res, 'tab:red')
# plt.plot(inv_temps, inv_res, 'tab:blue')

# fig, axs = plt.subplots(2, 2)
# fig.suptitle(material_name) 

# axs[0, 0].plot(inv_temps, inv_res, 'tab:red')
# axs[0, 0].set(xlabel='1/T(K)', ylabel='$\\rho_{xx}$', title="Resistivity")

# axs[0, 1].plot(inv_temps, inv_der1_res, 'tab:green')
# axs[0, 1].set(xlabel='1/T(K)', ylabel='$\\frac{d}{dt}(\\rho_{xx})$', title="First Derivative")

# axs[1, 0].plot(inv_temps, inv_der2_res, 'tab:blue')
# axs[1, 0].set(xlabel='1/T(K)', ylabel='$\\frac{d^2}{dt^2}(\\rho_{xx})$', title="Second Derivative")

# axs[1, 1].plot(inv_temps,  inv_ln1_res, 'tab:orange')
# axs[1, 1].set(xlabel='1/T(K)', ylabel='$ln(\\rho_{xx})$', title="Natural Log")

# Combined figures -----------------------------------------------------------------------

f3 = plt.figure()
# f4 = plt.figure()

norm_lin_res = normalize(lin_res)
norm_der1_res = normalize(der1_res)
norm_der2_res = normalize(der2_res)
norm_ln1_res = normalize(ln1_res)

plt.figure(f3)

plt.plot(lin_temps, norm_lin_res, 'tab:red')
plt.plot(lin_temps, norm_der1_res, 'tab:green')
plt.plot(lin_temps, norm_der2_res, 'tab:blue')
plt.plot(lin_temps, norm_ln1_res, 'tab:orange')
plt.title(material_name + ' Combined & Normalized')
plt.xlabel('Temperature (K)')   

# # *** Inverse Plotting ***

# norm_inv_res = normalize(inv_res)
# norm_inv_der1_res = normalize(inv_der1_res)
# norm_inv_der2_res = normalize(inv_der2_res)
# norm_inv_ln1_res = normalize(inv_ln1_res)

# plt.figure(f4)
# plt.plot(inv_temps, norm_inv_res, 'tab:red')
# plt.plot(inv_temps, norm_inv_der1_res, 'tab:green')
# plt.plot(inv_temps, norm_inv_der2_res, 'tab:blue')
# plt.plot(inv_temps, norm_inv_ln1_res, 'tab:orange')
# plt.title(material_name + ' Combined & Normalized')
# plt.xlabel('1 / Temperature (K)')

plt.show()
