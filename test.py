import librosa
from matplotlib import pyplot as plt
import numpy as np
from Spiro_package import Spiro, Spiro_feature
import json

# Open the JSON file
with open('spiro_data_collect.json', 'r') as f:
    # Load the JSON data into a Python dictionary
    data = json.load(f)
dataarray = []
for d in data:
    dataarray.append((data[d]['dataArray']))


Spiro = Spiro()
feature = Spiro_feature()
#filtered_data = Spiro.filter_data(dataarray[2])
# plt.plot(dataarray[0])
# plt.show()
# print(len(dataarray[0])/300)
print(len(dataarray))

print("Tidal Vol ", feature.sig_tidal_vol(dataarray[0]))
print("Fev 1", feature.sig_FEV1(dataarray[0]))
print("Fvc ", feature.sig_FVC(dataarray[0]))
print("Ratio ", feature.sig_fev1_fvc_ratio(dataarray[0]))
print("Tidal vol ", feature.sig_tidal_vol(dataarray[0]))
print("Inspiration Time ", feature.sig_inspiratory_time(
    dataarray[0], int(len(dataarray[0])/300)))
print("Expiration Time ", feature.sig_expiratory_time(
    dataarray[0], int(len(dataarray[0])/300)))
print("Inspiratory Flow rate", feature.sig_inspiratory_flow_rate(
    dataarray[0], int(len(dataarray[0])/300)))
print("Inspiratory Flow rate", feature.sig_expiratory_flow_rate(
    dataarray[0], int(len(dataarray[0])/300)))
print("Flatline ", feature.sig_flatline(dataarray[0]))
print("Zero Hitting ", feature.sig_zero_hitting(dataarray[0]))
print("Shannon Energy ", feature.sig_shannon_entropy(dataarray[0]))


print()
