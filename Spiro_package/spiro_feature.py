# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:16:26 2022

@author: KIIT / Mosaif Ali
"""

from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import neurokit2 as nk
import librosa
import collections
from Spiro_package import Spiro
from scipy.stats import entropy


class Spiro_feature:

    """

    Class Spiro_feature.
    Functions for spiro feature extraction operations .

    """

    def __init__(self):
        """

        Empty Default Constructor

        """

        pass

    def sig_filter(self, signal, sampling_rate=1000, lowcut=None, highcut=None, method='butterworth', order=2):
        """

        sig_filter()

        =>function to filter the signal .

        :param signal : Signal to be filtered.
        :type string: []

        :param sampling_rate: Sample Rate of the Signal default is set to 1000.
        :type string: int

        :param lowcut: lowcut value of the filter.
        :type string: int

        :param highcut: highcut value of the filter.
        :type string: int

        :param method: filter type. Default set to Butterworth.
        :type string: str

        :param order: filter order. Default set to 2.
        :type string: int


        :return: Filtered Signal
        :rtype: [] 

        """

        return nk.signal_filter(signal, sampling_rate, lowcut, highcut, method, order)

    def sig_flatline(self, signal, threshold=0.01):
        """

        sig_flatline()

        =>Return the Flatline Percentage of the Signal

        :param signal : Signal .
        :type string: []

        :param threshold: Flatline threshold relative to the biggest change in the signal.
        :type string: int


        :return: Percentage of signal where the absolute value of the derivative is lower then the threshold.
        :rtype: float  

        """

        return nk.signal_flatline(signal, threshold)

    def sig_decompose(self, signal, method='emd'):
        """

        sig_decompose()

        =>Signal decomposition into different sources using different methods, such as Empirical Mode Decomposition (EMD) or Singular spectrum analysis (SSA)-based signal separation method. .

        :param signal : Signal .
        :type lst: []


        :param method: he decomposition method. Can be one of "emd" or "ssa".
        :type string: str


        :return: Components of the decomposed signal.
        :rtype: [] 

        """

        return nk.signal_decompose(signal, method)

    def sig_continuance(self, signal):
        """

        sig_continuance()

        =>Finds the contiunance of the signal in the positive and negative directions.

        :param signal : Signal .
        :type lst: []


        :return: Positive and Negative continuance of the signal.
        :rtype: int

        """

        counter_forward = 0
        counter_backward = 0
        for s in signal:
            if s > 1:
                counter_forward += 1
            else:
                counter_backward += 1
        return counter_forward, counter_backward

    def sig_consecutive_positive_peak_distance(self, signal):
        """

        sig_consecutive_positive_peak_distance()

        =>Finds the distance between consecutive peaks in positive .

        :param signal : Signal.
        :type lst: []


        :return: List of distance between consecutive peaks.
        :rtype: []

        """

        signal = np.array(signal)
        peak, _ = find_peaks(signal, height=0)
        dist_consecutive_peak_positive = []
        for i in range(0, len(peak)-1):

            # y-axis=x[p]
            # x-axis=p

            dist = peak[i+1]-peak[i]
            dist_consecutive_peak_positive.append(dist)
        return dist_consecutive_peak_positive

    def sig_consecutive_negative_peak_distance(self, signal):
        """

        sig_consecutive_negative_peak_distance()

        =>Finds the distance between consecutive peaks.

        :param signal : Signal .
        :type lst: []


        :return: List of distance between consecutive peaks.
        :rtype: []

        """

        neg_x = []
        for n in signal:
            neg_x.append(-(n))
        neg_x = np.array(neg_x)

        peak_neg, _ = find_peaks(neg_x, height=0)
        dist_consecutive_peak_negative = []
        for i in range(0, len(peak_neg)-1):

            # y-axis=x[p]
            # x-axis=p

            dist = peak_neg[i+1]-peak_neg[i]
            dist_consecutive_peak_negative.append(dist)
        return dist_consecutive_peak_negative

    def sig_difference_largest_positive_peaks(self, signal):

        signal = np.array(signal)
        peak, _ = find_peaks(signal, height=0)
        lst = []
        for p in peak:
            lst.append(float(signal[p]))

        lst.sort()
        signal = signal.tolist()
        index_first = lst[-1]
        index_second = lst[-2]

        first = (signal.index(index_first))
        second = (signal.index(index_second))
        if first > second:

            diff_between_two_largest_positive_peaks = first-second
            return diff_between_two_largest_positive_peaks
        else:
            diff_between_two_largest_positive_peaks = -1*(first-second)
            return diff_between_two_largest_positive_peaks

    def sig_difference_largest_negative_peaks(self, signal):

        neg_x = []
        for n in signal:
            neg_x.append(-(n))
        neg_x = np.array(neg_x)

        peak_neg, _ = find_peaks(neg_x, height=0)

        lst_neg = []
        for p in peak_neg:
            lst_neg.append(float(-1*signal[p]))

        lst_neg.sort()

        index_first_neg = lst_neg[-1]
        index_second_neg = lst_neg[-2]

        first_neg = (signal.index(-1*index_first_neg))
        second_neg = (signal.index(-1*index_second_neg))
        if first_neg > second_neg:

            diff_between_two_largest_negative_peaks = first_neg-second_neg
            return diff_between_two_largest_negative_peaks
        else:
            diff_between_two_largest_negative_peaks = -1*(first_neg-second_neg)
            return diff_between_two_largest_negative_peaks

    def sig_positive_plateau_sustain_time(self, signal):

        peak, _ = find_peaks(signal, height=0)
        lst = []
        for p in peak:
            lst.append(float(signal[p]))

        max_start = max(lst[:2])
        max_end = max(lst[-3:])
        peak_start = signal.index(max_start)
        peak_end = signal.index(max_end)

        plateau_sustain_positive_time = peak_end-peak_start

        return plateau_sustain_positive_time

    def sig_negative_plateau_sustain_time(self, signal):

        neg_x = []
        for n in signal:
            neg_x.append(-(n))
        neg_x = np.array(neg_x)

        peak_neg, _ = find_peaks(neg_x, height=0)

        lst_neg = []
        for p in peak_neg:
            lst_neg.append(float(-1*signal[p]))

        max_start = max(lst_neg[:2])
        max_end = max(lst_neg[-3:])
        peak_start_neg = signal.index(-1*max_start)
        peak_end_neg = signal.index(-1*max_end)

        plateau_sustain_negative_time = peak_end_neg-peak_start_neg

        return plateau_sustain_negative_time

    def sig_zero_hitting(self, signal):

        peak, _ = find_peaks(signal, height=0)
        lst = []
        for p in peak:
            lst.append(float(signal[p]))

        max_start = max(lst[-3:])
        for d in signal:
            if d < 0:
                max_end = d
                break

        peak_start = signal.index(max_start)
        peak_end = signal.index(max_end)

        zero_hiting = peak_end-peak_start

        return zero_hiting

    def sig_FEV1(self, signal):
        avg = len(signal) / float(5)
        out = []
        last = 0.0

        while last < len(signal):
            out.append(signal[int(last):int(last + avg)])
            last += avg

        avg = sum(out[0]) / len(out[0])
        num = ((3.14159)*(0.0092**4)*(avg))
        deno = (8*1.145*0.0035)
        fev1 = (num/deno)*1000
        return fev1

    def sig_FVC(self, signal):
        avg = len(signal) / float(5)
        out = []
        last = 0.0

        while last < len(signal):
            out.append(signal[int(last):int(last + avg)])
            last += avg
        avg = []
        for x in out:
            avg.append(sum(x)/len(x))
        v = []
        for a in avg:
            num = ((3.14159)*(0.0092**4)*(a))
            deno = (8*1.145*0.0035)
            volume = (num/deno)*1000
            v.append(volume)

        return sum(v)

    def sig_fev1_fvc_ratio(self, signal):
        fev1 = self.sig_FEV1(signal)
        fvc = self.sig_FVC(signal)
        return fev1/fvc

    # ======================= Librosa =====================================

    def sig_energy(self, signal, frame, hop):
        energy = np.array([
            sum(abs(signal[i:i+frame]**2))
            for i in range(0, len(signal), hop)
        ])
        return energy

    def sig_rms(self, signal, frame, hop):

        signal = [float(x) for x in signal]
        signal = np.array(signal)
        rmse = librosa.feature.rms(
            y=signal, frame_length=frame, hop_length=hop, center=True)
        return rmse[0]

    def sig_mfccs(self, signal, sr):

        # returns 13 features
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
        delta_mfccs = librosa.feature.delta(mfccs, mode='constant')
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, mode='constant')
        comprehensive_mfccs = np.concatenate(
            (mfccs, delta_mfccs, delta2_mfccs))
        return comprehensive_mfccs

    def sig_shannon_entropy(self, signal):
        bases = collections.Counter([tmp_base for tmp_base in signal])

        # define distribution
        dist = [x/sum(bases.values()) for x in bases.values()]

        # use scipy to calculate entropy
        entropy_value = entropy(dist, base=2)

        return entropy_value

    def sig_spectral_bandwidth(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal, sr=float(sr))
        return spectral_bandwidth[0]

    def sig_central_centroid(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        return spectral_centroid[0]

    def sig_spectral_rolloff(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        return spectral_rolloff[0]

    def sig_zero_crossing(self, signal):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        zero_crossings = librosa.zero_crossings(y=signal)
        return sum(zero_crossings)

    def sig_zero_crossing_rate(self, signal):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        zcrs = librosa.feature.zero_crossing_rate(y=signal)
        return zcrs[0]

    def sig_tempo(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        tempo = librosa.feature.tempo(y=signal, sr=sr)
        return tempo

    def sig_spectral_bandwidth(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal, sr=sr)
        return spectral_bandwidth

    def sig_spectral_contrast(self, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        return spectral_contrast

    def sig_spectral_flatness(self, signal):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_flatness = librosa.feature.spectral_flatness(y=signal)
        return spectral_flatness

    def sig_spectral_rolloff(sefl, signal, sr):
        signal = [float(x) for x in signal]
        signal = np.array(signal)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
        return spectral_rolloff

    # ============================ BPAT =========================================

    def volume(self, p_diff):
        num = ((3.14159)*(0.0092**4)*(p_diff))
        deno = (8*1.145*0.0035)
        volume = (num/deno)*1000
        return round(volume, 5)

    def sig_respiratory_rate(self, signal, sr):

        t = len(signal)/int(sr)
        b = len(Spiro().extract_wavelet(signal))

        return round((b/t)*60, 0)

    def sig_tidal_vol(self, signal):
        vol_up = self.volume(
            sum([x for x in signal if x > 0])/len([x for x in signal if x > 0]))
        vol_down = self.volume(
            sum([x for x in signal if x < 0])/len([x for x in signal if x < 0]))

        return round(vol_up+(-1)*vol_down, 5)

    def sig_inspiratory_time(self, signal, sr):

        return round(len([x for x in signal if x < 0])/int(sr), 5)

    def sig_expiratory_time(self, signal, sr):

        return round(len([x for x in signal if x > 0])/int(sr), 2)

    def sig_inspiratory_flow_rate(self, signal, sr):
        vol_down = self.volume(
            sum([x for x in signal if x < 0])/len([x for x in signal if x < 0]))
        return -1*round(vol_down/self.sig_inspiratory_time(signal, sr), 5)

    def sig_expiratory_flow_rate(self, signal, sr):
        vol_up = self.volume(
            sum([x for x in signal if x > 0])/len([x for x in signal if x > 0]))
        return round(vol_up/self.sig_inspiratory_time(signal, sr), 5)
