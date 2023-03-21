# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:28:51 2022

@author: KIIT
"""
from Spiro_package import Spiro, Spiro_feature
import numpy as np

from matplotlib import pyplot as plt
import librosa

Spiro = Spiro()
feature = Spiro_feature()
# pd,co2=Spiro.get_data('COM3')
pd = Spiro.read_data('spiro_data.csv')
filtered_data = Spiro.filter_data(pd)
x_0 = []
# for i in range(0, len(pd)):
#    x_0.append(0)
w = Spiro.extract_wavelet(pd)
# plt.figure(800)
# plt.xlabel("Time")
#plt.ylabel("Diifrential Pressure")
# plt.plot(filtered_data)
# plt.plot(x_0)
# plt.show()
# print(len(w))
# for i in range(0, len(w)):
#    plt.figure((i+1)*100)
#    plt.plot(w[i])
#    plt.plot(x_0)
#    plt.show()

wavelet1 = w[0]

print('1', feature.estimate_rms(wavelet1, 100, 2))
print('2', feature.estimate_mfccs(wavelet1, 44100))
print('3', feature.estimate_shannon_entropy(wavelet1))
print('4', feature.estimate_spectral_bandwidth(wavelet1, 44100))
print('5', feature.estimate_central_centroid(wavelet1, 44100))
print('6', feature.estimate_spectral_rolloff(wavelet1, 44100))
print('6', feature.estimate_zero_crossing(wavelet1))
print('7', feature.estimate_zero_crossing_rate(wavelet1))
print('8', feature.estimate_tempo(wavelet1, 44100))
print('9', feature.estimate_spectral_bandwidth(wavelet1, 44100))
print('10', feature.estimate_spectral_contrast(wavelet1, 44100))
print('11', feature.estimate_spectral_flatness(wavelet1))
print('12', feature.estimate_spectral_rolloff(wavelet1, 44100))

