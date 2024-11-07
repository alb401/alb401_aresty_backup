import pandas as pd
import numpy as np
import holoviews as hv
import bokeh as bk
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

temps = [] # Temperatures
res = [] # Resistivities
material_name = "$Eu_2Ir_2O_7$" 

# **Investigate doing everything with numpy, might save time** #

# Function for data normalization
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# Holoviews stuff -------------------------------------------------------------------

# material_info = pd.read_csv('1_RT_EIO4732.csv')
# material_info.head()

# scatter = hv.Scatter(material_info, 'Temp_EIO', 'Rhoxx_EIO')

# Pandas stuff ----------------------------------------------------------------------

df = pd.read_csv('/home/alec/Documents/2024-2025/Aresty/python/092324/1_RT_EIO4732.csv')

'''
The raw data from this isn't wonderful,
should probably be cleaned up, have some duplicate values removed, etc. 

After that, generate 3000 temp values or whatever, linearly interpolate, savgol, and so on
As you did in OriginPro
'''

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

# ---------------------------------------------------------------------------------------------

WINDOW_LENGTH = 50;

# Savitsky-Golay filter for smoothing
lin_res = savgol_filter(lin_res, WINDOW_LENGTH, 2)
lin_res = np.array(lin_res) 

# First Derivative
der1_res = np.gradient(lin_res, lin_temps, edge_order=2)
der1_res = savgol_filter(der1_res, 2*WINDOW_LENGTH, 2)
print(der1_res)

# Second Derivative
der2_res = np.gradient(der1_res, lin_temps, edge_order=2)
# der2_res = np.diff(lin_res, )
der2_res = savgol_filter(der2_res, 2*WINDOW_LENGTH, 2)
print(der2_res)


# Plotting #############################################################################

# Resistivity vs. Temperature ----------------------------------------------------------
plt.figure(1)
plt.plot(lin_temps, lin_res, 'r-')
plt.title(material_name + ' Resistivity vs. Temperature')
plt.xlabel('Temperature (K)')
plt.ylabel('Resistivity $(\Omega m)$')

# First Derivative ---------------------------------------------------------------------
plt.figure(2)
plt.plot(lin_temps, der1_res, 'g-')
plt.title(material_name + ' 1st Derivative Resistivity vs. Temperature')
plt.xlabel('Temperature (K)')


# Second Derivative ---------------------------------------------------------------------
plt.figure(3)
plt.plot(lin_temps, der2_res, 'b-')
plt.title(material_name + ' 2nd Derivative Resistivity vs. Temperature')
plt.xlabel('Temperature (K)')

# Natural Log ---------------------------------------------------------------------------
ln1_res = np.log(lin_res);

plt.figure(4)
plt.plot(lin_temps, ln1_res, 'm-')
plt.title(material_name + ' Natural Log vs. Temperature')
plt.xlabel('Temperature (K)')


# Combined figure -----------------------------------------------------------------------
norm_lin_res = normalize(lin_res)
norm_der1_res = normalize(der1_res)
norm_der2_res = normalize(der2_res)
norm_ln1_res = normalize(ln1_res)

plt.figure(5)
plt.plot(lin_temps, norm_lin_res, 'r-')
plt.plot(lin_temps, norm_der1_res, 'g-')
plt.plot(lin_temps, norm_der2_res, 'b-')
plt.plot(lin_temps, norm_ln1_res, 'm-')
plt.title(material_name + ' Combined')
plt.xlabel('Temperature (K)')
plt.show()




# directory = os.path.join("c:\\","path")
# for root,dirs,files in os.walk(directory):
#     for file in files:
#        if file.endswith(".csv"):
#            f = pd.read_csv(file)
#            print(file.head())



# print("\n")
# print(ynew)
# # print(ynew.size)

# df = pd.DataFrame({'Temp_EIO': lin_temps, 'Rhoxx_EIO': ynew})
# print(df)


# df.plot(kind = 'line', x = 'Temp_EIO', y = 'Rhoxx_EIO')

# plt.title(material_name + ' Resistivity vs. Temperature')
# plt.xlabel('Temperature (K)')
# plt.ylabel('Resistivity $(\Omega m)$')

# plt.show()


# def savgol(x, wl=3, p=2):
#     return savgol_filter(x, window_length=wl, polyorder=p)  

# df['savgol'] = df.groupby('Temp_EIO')['Rhoxx_EIO'].transform(lambda x: savgol_filter(x, 5,2))


