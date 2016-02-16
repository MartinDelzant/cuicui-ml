import numpy as np
import scipy as sc
from scipy.signal import (butter, lfilter, spectrogram)
import math
import matplotlib.pyplot as plt


def butter_bandpass_filter(data, lowcut=500, highcut=5000, fs=44100, order=10):

    def butter_bandpass(lowcut, highcut, fs=44100, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def spectro_conv(signal, fs):
    f, t, spectro = spectrogram(signal, fs=fs, window=("kaiser", 8.0), nperseg=256, noverlap=64)
    return f, t, spectro


def plot_spectrogram(f, t, spectro):
    plt.pcolormesh(t, f, spectro)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def compute_amplitudes(spectro):
    return 20*np.log10(np.abs(spectro))


# incertitude : use same frequency to find high cut and low cut or max over all frequencies ???
def extract_syllabe(spectro, beta=20, stop_amp=40, all_freq=True):  # test stop_amp value (proportional to beta)
    # compute amplitudes
    amplitudes = compute_amplitudes(spectro)
    # look for indice of max in spectrogram
    spectro_abs = np.abs(spectro)
    max_power = np.max(spectro_abs)
    max_ampli = 20*math.log(max_power, 10)
    if max_ampli <= stop_amp:
        return -1, -1, -1, -1
    # find index of row and column corresponding to max power/amplitude
    t_max, f_max = np.unravel_index(spectro_abs.argmax(), spectro_abs.shape)
    # browse neighbor amplitudes until it has decreased enough
    # forward search
    high_cut = spectro.shape[0] - 1
    for t in range(spectro.shape[0]):
        if all_freq:
            curr_f_max = np.argmax(amplitudes[t_max + t, :])
            curr_value = amplitudes[t_max + t, curr_f_max]
        else:
            curr_value = amplitudes[t_max + t, f_max]
        if t_max + t >= spectro.shape[0] - 1:
            break
        if curr_value < max_ampli - beta:
            high_cut = t_max + t
            break
    # backward search
    low_cut = 0
    for t in range(spectro.shape[0]):
        if all_freq:
            curr_f_max = np.argmax(amplitudes[t_max - t, :])
            curr_value = amplitudes[t_max - t, curr_f_max]
        else:
            curr_value = amplitudes[t_max - t, f_max]
        if t_max - t <= 0:
            break
        if curr_value < max_ampli - beta:
            low_cut = t_max - t
            break
    return low_cut, high_cut, t_max, max_power


def extract_all_syllabes(spectro, beta=20, stop_amp=10, n_max=100, all_freq=True):
    low_cuts = []
    high_cuts = []
    idxs_max = []
    max_powers = []

    for i in range(n_max):
        low_cut, high_cut, idx_max, max_power = extract_syllabe(spectro, beta=beta, stop_amp=stop_amp, all_freq=all_freq)
        # this means that there are no longer syllabes to extract
        if low_cut == -1:
            break
        low_cuts.append(low_cut)
        high_cuts.append(high_cut)
        idxs_max.append(idx_max)
        max_powers.append(max_power)
        # put values to zero to avoid retrieving same syllabe multiple times
        spectro[low_cut:high_cut+1, :] = np.zeros((high_cut-low_cut+1, spectro.shape[1]))

    return low_cuts, high_cuts, idxs_max, max_powers

