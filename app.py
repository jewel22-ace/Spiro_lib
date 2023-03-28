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
wavelet2 = w[1]
print(feature.sig_inspiratory_time(w[0], 351.2))
