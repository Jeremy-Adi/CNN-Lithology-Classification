# # # This Script Sould Be Combined With Script "02_Calculate The Accuracy Of Prediction.py" # # #
# # # Some Variables contained in this script are belong to Script "02_Calculate The Accuracy Of Prediction.py" # # #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # # # # # # # # # # # PLOTTING FACIES # # # # # # # # # # #

from matplotlib.colors import ListedColormap as cmp
from matplotlib.colorbar import ColorbarBase as cb

dictionary = dict(zip(F3, facies__))
dictionary_ = dict(zip(F4, facies__pred))
print(dictionary)
print(dictionary_)

Facies_OG = np.vstack((F3,F3)).T
Facies_PR = np.vstack((F4,F4)).T

rows,cols = 1,5

data_for_plot = Well_Pred_Posei2[['GR','TNP','RHOB','DEPTH']]

logs = data_for_plot.sort_values(by='DEPTH')
top = data_for_plot.DEPTH.min()
bot = data_for_plot.DEPTH.max()

# # COLOR CODE FOR EACH LITHOLOGY

# Calcarenite = Aqua / #00FFFF
# Calcilutite = blue / #0000FF
# Claystone = gray / #A9A9A9
# Sandstone = yellow
# Siltstone = brown / #A52A2A
# Volcanics = black
# Limestone / Chert= dark blue / #00008B
# Argillaceous Siltstone = cream / #fffdd0
# Silty Claystone = dimgray / #696969
# Silty Sandstone = Yellow green / #9acd32

# COLORMAP
cmap = cmp(['#A9A9A9','yellow','#A52A2A','black']) #Changeable
cmap2 = cmp(['#A9A9A9','yellow','#A52A2A','black','#0000FF']) #Changeable

# # # SOURCE CODE FOR LOGS AND LITHOLOGY PLOTTING IS CREATED BY AGUS ABDULLAH, Ph.D. # # #

ff, axx = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 12))
axx[0].plot(data_for_plot.GR, data_for_plot.DEPTH, color='green', linewidth = '0.5')
axx[1].plot(data_for_plot.TNP, data_for_plot.DEPTH, color='red', linewidth = '0.5')
axx[2].plot(data_for_plot.RHOB, data_for_plot.DEPTH, color='black', linewidth = '0.5')
a = axx[3].imshow(Facies_OG,cmap=cmap,aspect='auto',extent=[0,1,max(data_for_plot["DEPTH"]),min(data_for_plot["DEPTH"])])
b = axx[4].imshow(Facies_PR,cmap=cmap2,aspect='auto',extent=[0,1,max(data_for_plot["DEPTH"]),min(data_for_plot["DEPTH"])])

for i in range(len(axx)-2): # EXCLUDE LITHOLOGY
    axx[i].set_ylim(top, bot)
    axx[i].invert_yaxis()
    axx[i].grid(which='major', linestyle='-', linewidth='0.5', color='green')
    axx[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    axx[i].minorticks_on()

axx[0].set_xlabel("GR")
axx[0].set_xlim(data_for_plot.GR.min(), data_for_plot.GR.max())
axx[0].set_ylabel("Depth(m)")
axx[0].set_ylim(data_for_plot.DEPTH.max(), data_for_plot.DEPTH.min())
axx[1].set_xlabel("NPHI")
axx[1].set_xlim(data_for_plot.TNP.min(), data_for_plot.TNP.max())
axx[2].set_xlabel("RHOB")
axx[2].set_xlim(data_for_plot.RHOB.min(), data_for_plot.RHOB.max())
axx[3].set_xlabel("Facies Actual")
axx[4].set_xlabel("Facies Predicted Simplified")

axx[0].set_yticklabels([]);
axx[0].set_yticks(());
axx[1].set_yticklabels([]);
axx[2].set_yticklabels([]);
axx[3].set_yticklabels([]);
axx[4].set_yticklabels([]);
# ax[6].set_yticklabels([]);

ff.suptitle('Well:POSEIDON #2', fontsize=14, y=0.94)

# # CREATE MORE AXES FOR COLORBAR

ff.subplots_adjust(right=0.8)
cbar_ax = ff.add_axes([0.91, 0.15, 0.05, 0.7])
cbar_ax2 = ff.add_axes([0.85, 0.15, 0.05, 0.7])
cb2 =  ff.colorbar(a, cax=cbar_ax)
cb3 =  ff.colorbar(b,cax=cbar_ax2)
cb2.set_label('Facies Actual')
cb3.set_label('Facies Predicted Simplified')
plt.show()