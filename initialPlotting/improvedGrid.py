import pandas as pd
import numpy as np
import holoviews as hv
import bokeh as bk
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.signal import savgol_filter

fig, axs = plt.figure()
spec = mpl.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0:2]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,2:4])
ax3 = fig.add_subplot(spec[0,4:])
ax4 = fig.add_subplot(spec[1,1:3]) # row 0 with axes spanning 2 cols on odds
ax5 = fig.add_subplot(spec[1,3:5])

plt.show()