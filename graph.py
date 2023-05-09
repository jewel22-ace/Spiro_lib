import librosa
from matplotlib import pyplot as plt
import numpy as np
from Spiro_package import Spiro, Spiro_feature
import json

Spiro = Spiro()
feature = Spiro_feature()

# Open the JSON file
with open('spiro_data_collect_.json', 'r') as f:
    # Load the JSON data into a Python dictionary
    data = json.load(f)
dataarray = []
for d in data:
    dataarray.append((data[d]['dataArray']))
# plt.plot(dataarray[0])
data_in_use = dataarray[7]

plt.plot(data_in_use)
plt.show()
