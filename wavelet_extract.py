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
data_in_use = dataarray[8]
p = []
for i in range(len(data_in_use)):
    try:
        if data_in_use[i-2] < 0 and data_in_use[i-1] < 0 and data_in_use[i] > 0 and data_in_use[i+1] > 0:
            p.append(i)
    except Exception as e:
        pass
wavelets = []
for j in range(len(p)-1):
    wavelets.append(data_in_use[int(p[j]):int(p[j+1])])
print(len(wavelets))
# plt.plot(data_in_use)
# plt.plot(wavelets[0])
# plt.show()

fev1_lst = []
fvc = []
tidal_vol = []
p_s_t = []
n_s_t = []
for i in range(len(wavelets)):
    # print("Fev 1", feature.sig_FEV1(wavelets[i]))
    fev1_lst.append(feature.sig_FEV1(
        wavelets[i]))
    fvc.append(feature.sig_FVC(
        wavelets[i]))
    tidal_vol.append(feature.sig_tidal_vol(
        wavelets[i]))
    p_s_t.append(feature.sig_positive_plateau_sustain_time(
        wavelets[i]))
    n_s_t.append(feature.sig_negative_plateau_sustain_time(
        wavelets[i]))


print('FEV1 : ', round(sum(fev1_lst)/len(fev1_lst), 3))
print('Fvc : ', round(sum(fvc)/len(fvc), 3))
print('Ratio :')
print('Tidal Vol : ', round(sum(tidal_vol)/len(tidal_vol), 3))
print('RR:', len(wavelets))
print('Inspiratory time :', feature.sig_inspiratory_time(
    data_in_use, int(len(data_in_use)/60)))
print('Expiratory time :', feature.sig_expiratory_time(
    data_in_use, int(len(data_in_use)/60)))
print('Inspiratory Flow rate :', feature.sig_inspiratory_flow_rate(
    data_in_use, int(len(data_in_use)/60)))
print('Expiratory Flow rate :', feature.sig_expiratory_flow_rate(
    data_in_use, int(len(data_in_use)/60)))
print('Signal Continuance:', feature.sig_continuance(
    data_in_use))
print("sig_consecutive_positive_peak_distance",
      feature.sig_consecutive_positive_peak_distance(data_in_use))
print("sig_consecutive_negative_peak_distance",
      feature.sig_consecutive_negative_peak_distance(data_in_use))
print("sig_difference_largest_positive_peaks",
      feature.sig_difference_largest_positive_peaks(data_in_use))
print("sig_difference_largest_negative_peaks",
      feature.sig_difference_largest_negative_peaks(data_in_use))
print('Positive plateau Sustain Time : ', round(sum(p_s_t)/len(p_s_t), 3))
print('Negative plateau Sustain Time : ', round(sum(n_s_t)/len(n_s_t), 3))

print("Flatline ", round(feature.sig_flatline(data_in_use), 3))

print('Rms ', feature.sig_rms(
    data_in_use, 10, 20))
print('mfcc ', feature.sig_mfccs(
    data_in_use, len(data_in_use)/60))
print('Shannon Entropy:', round(feature.sig_shannon_entropy(
    data_in_use), 3))
print('sig_central_centroid ', feature.sig_central_centroid(
    data_in_use, len(data_in_use)/60))
print('sig_spectral_bandwidth ', feature.sig_spectral_bandwidth(
    data_in_use, len(data_in_use)/60))
print('sig_spectral_rolloff ', feature.sig_spectral_rolloff(
    data_in_use, len(data_in_use)/60))
print('zero_crossing', feature.sig_zero_crossing(
    data_in_use))
print('sig_spectral_flatness', feature.sig_spectral_flatness(
    data_in_use))


# print('mfcc ', feature.sig_mfccs(
#     data_in_use, len(data_in_use)/60))
# print('tempo ', feature.sig_spectral_bandwidth(
#     data_in_use, len(data_in_use)/60))


# print(p)

# plt.plot(data_in_use)
# plt.plot(data_in_use[int(p[1]):int(p[2])])

# plt.show()
# print("Fev 1", feature.sig_FEV1(data_in_use[int(p[1]):int(p[2])]))
