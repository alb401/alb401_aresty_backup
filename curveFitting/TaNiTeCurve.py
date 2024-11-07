import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, curve_fit

# Function Definitions

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def func1(x, a, b, c):
    return a * np.power(x, b) + c

def func2(x, a, b, c):
    return a * np.power(x, 2) + b*x + c

def rsquare(y, f):
    ymean = np.mean(y)
    ss_res = np.sum((y - f)**2)
    ss_tot = np.sum((y - ymean)**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

### FILE READING ### ------------------------------------------------------------------------------------
'''
IN ORDER FOR READING FILE TO WORK, ALL TEXT EXTRA HEADER INFORMATION MUST BE DELETED FROM FILE
'''

# Properly Read in .dat file
df = pd.read_csv('/home/alec/Documents/2024-2025/Aresty/python/102224/TaNiTeData.dat', delimiter='\t')
# df2 = pd.readcsv()

# --------------------------------------------------------------------------------------------------------

# Get arrays from data frame to work on explicitly 
temp = df["T"]
res1 = df["Rhoxx_Dec_2023"]
res2 = df["Rhoxx_Sept_2024"]

# # Find the minimum temperature from 'temp' array
# min_temp = temp.idxmin()

# # Remove half of the array, so we're left with the data from just the temperature increase
# temp = temp[min_temp:]
# res1 = res1[min_temp:]
# res3 = res3[min_temp:]

# Add modified arrays into a new dataframe and then convert them to numpy
# df = pd.concat([temp, res1, res3], axis=1)

if(df.iloc[0,0] > 30):
    df = df.iloc[::-1] # Reverses dataset from decreasing to increasing temp

df.drop_duplicates(inplace = True) # Removes all duplicates from dataset
df = df.to_numpy()

# Creates evenly spaced values for temperature
lo_temp = 2
hi_temp = 300
num_pts = (hi_temp-lo_temp)*10 + 1

plt.figure('scatter')
plt.plot(df[:,0], 1/df[:,1], 'r-' )

# Create evenly spaced linear interpolation for smoothing and analysis
lin_temps = np.linspace(lo_temp, hi_temp, num_pts)
dec23_lin_res = np.interp(lin_temps, df[:,0], df[:,1], period=None)
sept24_lin_res = np.interp(lin_temps, df[:,0], df[:,2], period=None)

e = 2.7182818

neg_sqrt_temps = -1 * (lin_temps)**0.5
relation = np.power(e, neg_sqrt_temps)
print(relation)
plt.figure()
plt.plot(lin_temps, relation, 'b-')

plt.show()

# Calculate and print fitpower of data
# print("power: ", fitpower(dec23_lin_res, lin_temps))

WINDOW_LENGTH = 100; # Window length for Savitsky-Golay filter

# Savitsky-Golay filter for smoothing
dec23_lin_res = savgol_filter(dec23_lin_res, WINDOW_LENGTH, 2)
dec23_lin_res = np.array(dec23_lin_res) 

sept24_lin_res = savgol_filter(sept24_lin_res, WINDOW_LENGTH, 2)
sept24_lin_res = np.array(sept24_lin_res)

# 1st Derivative
dec23_der_res = np.gradient(dec23_lin_res, lin_temps, edge_order=2)
dec23_der_res = savgol_filter(dec23_der_res, 4*WINDOW_LENGTH, 2)
dec23_der_res_max_x = np.argmax(dec23_der_res)
dec23_der_res_max_y = np.max(dec23_der_res)

sep24_der_res = np.gradient(sept24_lin_res, lin_temps, edge_order=2)
sep24_der_res = savgol_filter(sep24_der_res, 4*WINDOW_LENGTH, 2)
sep24_der_res_max_x = np.argmax(sep24_der_res)
sep24_der_res_max_y = np.max(sep24_der_res)

# Normalize and split data at point of maximum Derivative
# **This is done to curve fit for 2 separate regions**
dec23_lin_res = normalize(dec23_lin_res)
lin_temps = normalize(lin_temps)
dec23_lin_res_1 = dec23_lin_res[1:dec23_der_res_max_x]
dec23_lin_res_2 = dec23_lin_res[dec23_der_res_max_x:]
dec23_lin_temps_1 = lin_temps[1:dec23_der_res_max_x]
dec23_lin_temps_2 = lin_temps[dec23_der_res_max_x:]
popt1_1, pcov1_1 = curve_fit(func1, dec23_lin_temps_1, dec23_lin_res_1, maxfev=500000)
popt1_2, pcov1_2 = curve_fit(func1, dec23_lin_temps_2, dec23_lin_res_2, maxfev=500000)
popt1_3, pcov1_3 = curve_fit(func2, dec23_lin_temps_1, dec23_lin_res_1, maxfev=50000)
popt1_4, pcov1_4 = curve_fit(func2, dec23_lin_temps_2, dec23_lin_res_2, maxfev=50000)

sept24_lin_res = normalize(sept24_lin_res)
sept24_lin_res_1 = sept24_lin_res[1:sep24_der_res_max_x]
sept24_lin_res_2 = sept24_lin_res[sep24_der_res_max_x:]
lin_temps2_1 = lin_temps[1:sep24_der_res_max_x]
lin_temps2_2 = lin_temps[sep24_der_res_max_x:]
popt2_1, pcov2_1 = curve_fit(func1, lin_temps2_1, sept24_lin_res_1, maxfev=50000)
popt2_2, pcov2_2 = curve_fit(func1, lin_temps2_2, sept24_lin_res_2, maxfev=50000)
popt2_3, pcov2_3 = curve_fit(func2, lin_temps2_1, sept24_lin_res_1, maxfev=50000)
popt2_4, pcov2_4 = curve_fit(func2, lin_temps2_2, sept24_lin_res_2, maxfev=50000)

dec23_r2_sec1_fit1 = rsquare(dec23_lin_res_1, func1(dec23_lin_temps_1, *popt1_1))
dec23_r2_sec1_fit2 = rsquare(dec23_lin_res_1, func2(dec23_lin_temps_1, *popt1_3))

dec23_r2_sec2_fit1 = rsquare(dec23_lin_res_2, func1(dec23_lin_temps_2, *popt1_2))
dec23_r2_sec2_fit2 = rsquare(dec23_lin_res_2, func2(dec23_lin_temps_2, *popt1_4))

sept24_r2_sec1_fit1 = rsquare(sept24_lin_res_1, func1(sept24_lin_res_1, *popt2_1))
sept24_r2_sec1_fit2 = rsquare(sept24_lin_res_1, func2(sept24_lin_res_1, *popt2_3))

sept24_r2_sec2_fit1 = rsquare(sept24_lin_res_2, func1(sept24_lin_res_2, *popt2_2))
sept24_r2_sec2_fit2 = rsquare(sept24_lin_res_2, func2(sept24_lin_res_2, *popt2_4))

# ----------------------------------------------------------------------------------------------------

# Plotting
# '''

LINE_WIDTH = 3 
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 32 

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# COMBINED DECEMBER 2023 -----------------------------------------------------------------------------

# plt.figure("December 2023 Data")
# plt.title("TaNiTe Combined Fit Curves - Dec'23", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")
#
# actual1_1 = plt.plot(dec23_lin_temps_1, dec23_lin_res_1, c = 'b', label='Actual | Sec 1', linewidth=LINE_WIDTH)
# actual1_2 = plt.plot(dec23_lin_temps_2, dec23_lin_res_2, 'g-', label='Actual | Sec 2', linewidth=LINE_WIDTH)
#
# fit1_1 = plt.plot(dec23_lin_temps_1, func1(dec23_lin_temps_1, *popt1_1), 'r-', label='$ax^b+c$ fit | Sec 1')
# fit1_2 = plt.plot(dec23_lin_temps_2, func1(dec23_lin_temps_2, *popt1_2), 'm-', label='$ax^b+c$ fit | Sec 2')
#
# fit1_3 = plt.plot(dec23_lin_temps_1, func2(dec23_lin_temps_1, *popt1_3), 'k-', label='$ax^2+bx+c$ fit | Sec 1')
# fit1_4 = plt.plot(dec23_lin_temps_2, func2(dec23_lin_temps_2, *popt1_4), 'y-', label='$ax^2+bx+c$ fit | Sec 2')
#
# plt.legend()
#
# # SECTION 1 december 2023 -------------------------------------------------------------------------------
# plt.figure("December 2023 Section 1")
# plt.title("TaNiTe Fit Curves Section 1 - Oct'23", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")
#
# actual1_1 = plt.plot(dec23_lin_temps_1, dec23_lin_res_1, c = 'b', label='Actual', linewidth=LINE_WIDTH)
# fit1_1 = plt.plot(dec23_lin_temps_1, func1(dec23_lin_temps_1, *popt1_1), 'r-', label='$ax^b+c$ fit' )
# fit1_3 = plt.plot(dec23_lin_temps_1, func2(dec23_lin_temps_1, *popt1_3), 'k-', label='$ax^2+bx+c$ fit')
#
# plt.legend()
#
# # SECTION 2 december 2023 -------------------------------------------------------------------------------
# plt.figure("December 2023 Section 2")
# plt.title("TaNiTe Fit Curves Section 2 - Oct'23", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")
#
# actual1_2 = plt.plot(dec23_lin_temps_2, dec23_lin_res_2, c = 'g', label='Actual', linewidth=LINE_WIDTH)
# fit1_2 = plt.plot(dec23_lin_temps_2, func1(dec23_lin_temps_2, *popt1_2), 'm-', label='$ax^b+c$ fit')
# fit1_4 = plt.plot(dec23_lin_temps_2, func2(dec23_lin_temps_2, *popt1_4), 'y-', label='$ax^2+bx+c$ fit')
#
# plt.legend()

# # COMBINED SEPTEMBER 2024 ------------------------------------------------------------------------------

# plt.figure("September 2024 Data")
# plt.title("TaNiTe Combined Fit Curves -Sep'24", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")

# actual2_1 = plt.plot(lin_temps2_1, sept24_lin_res_1, c = 'b', label='Actual | Sec 1', linewidth=LINE_WIDTH)
# actual2_2 = plt.plot(lin_temps2_2, sept24_lin_res_2, 'g-', label='Actual | Sec 2', linewidth=LINE_WIDTH)

# fit2_1 = plt.plot(lin_temps2_1, func1(lin_temps2_1, *popt2_1), 'r-', label='$ax^b+c$ fit | Sec 1' )
# fit2_3 = plt.plot(lin_temps2_1, func2(lin_temps2_1, *popt2_3), 'k-', label='$ax^2+bx+c$ fit | Sec 1')

# fit2_2 = plt.plot(lin_temps2_2, func1(lin_temps2_2, *popt2_2), 'm-', label='$ax^b+c$ fit | Sec 2')
# fit2_4 = plt.plot(lin_temps2_2, func2(lin_temps2_2, *popt2_4), 'y-', label='$ax^2+bx+c$ fit | Sec 2')

# plt.legend()

# # # SECTION 1 SEPTEMBER 2024 -----------------------------------------------------------------------------

# plt.figure("September 2024 Section 1")
# plt.title("TaNiTe Fit Curves Section 1 - Sep'24", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")

# actual2_1 = plt.plot(lin_temps2_1, sept24_lin_res_1, c = 'b', label='Actual', linewidth=LINE_WIDTH)
# fit2_1 = plt.plot(lin_temps2_1, func1(lin_temps2_1, *popt2_1), 'r-', label='$ax^b+c$ fit')
# fit2_3 = plt.plot(lin_temps2_1, func2(lin_temps2_1, *popt2_3), 'k-', label='$ax^2+bx+c$ fit')

# plt.legend()

# # # SECTION 2 SEPTEMBER 2024 -----------------------------------------------------------------------------

# plt.figure("September 2024 Section 2")
# plt.title("TaNiTe Fit Curves Section 2 - Sep'24", fontsize=BIGGER_SIZE)
# plt.xlabel("Temperature (Normalized)")
# plt.ylabel("$\\rho_{xx}$ (Normalized)")

# actual2_2 = plt.plot(lin_temps2_2, sept24_lin_res_2, 'g-', label='Actual', linewidth=LINE_WIDTH)
# fit2_2 = plt.plot(lin_temps2_2, func1(lin_temps2_2, *popt2_2), 'm-', label='$ax^b+c$ fit' )
# fit2_4 = plt.plot(lin_temps2_2, func2(lin_temps2_2, *popt2_4), 'y-', label='$ax^2+bx+c$ fit')

# plt.legend()

# --------------------------------------------------------------------------------------------------------
# plt.figure("September 2024 Data #1")
# actual2_1 = plt.plot(lin_temps2_1, sept24_lin_res_1, 'b-', label='Actual 2_1')
# fit1 = plt.plot(lin_temps2_1, func1(lin_temps2_1, *popt2_1), 'r-', label='Fit 2_1')
# #
# actual2 = plt.plot(lin_temps2_2, sept24_lin_res_2, 'g-', label='Actual 2_2')perr = np.sqrt(np.diag(pcov))
# fit2 = plt.plot(lin_temps2_2, func1(lin_temps2_2, *popt2_2), 'm-', label='Fit 2_2')

# '''

print("\n\n----------------------------------------------\n")

print("popt1_1: ", popt1_1)
print("popt1_3: ", popt1_3)

print("R^2 of fit #1 on Section 1: ", dec23_r2_sec1_fit1)
print("R^2 of fit #2 on Section 1: ", dec23_r2_sec1_fit2)
print("R^2 difference (R^2 #2 - R^2 #1): ", dec23_r2_sec1_fit1 - dec23_r2_sec1_fit2)

print("\n----------------------------------------------\n")

print("popt1_2: ", popt1_2)
print("popt1_4: ", popt1_4)

print("R^2 of fit #1 on Section 2: ", dec23_r2_sec2_fit1)
print("R^2 of fit #2 on Section 2: ", dec23_r2_sec2_fit2)
print("R^2 difference (R^2 #2 - R^2 #1): ", dec23_r2_sec2_fit2 - dec23_r2_sec2_fit1)

print("\n----------------------------------------------\n")

print("\n\n----------------------------------------------\n")

print("popt2_1: ", popt2_1)
print("popt2_3: ", popt2_3)

print("R^2 of fit #1 on Section 1: ", sept24_r2_sec1_fit1)
print("R^2 of fit #2 on Section 1: ", sept24_r2_sec1_fit2)
print("R^2 difference (R^2 #2 - R^2 #1): ", sept24_r2_sec1_fit1 - sept24_r2_sec1_fit2)

print("\n----------------------------------------------\n")

print("popt2_2: ", popt2_2)
print("popt2_4: ", popt2_4)

print("R^2 of fit #1 on Section 2: ", sept24_r2_sec2_fit1)
print("R^2 of fit #2 on Section 2: ", sept24_r2_sec2_fit2)
print("R^2 difference (R^2 #2 - R^2 #1): ", sept24_r2_sec2_fit2 - sept24_r2_sec2_fit1)

print("\n----------------------------------------------\n")

plt.show()

# least_squares(fun, x0, jac='2-point', bounds=(-inf, inf), method='trf',
#      ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear',
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html



# Fit to T^4
