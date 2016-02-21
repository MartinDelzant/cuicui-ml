import numpy as np
from scipy.signal import (butter, lfilter, spectrogram)
import math
import matplotlib.pyplot as plt
import copy

sampling_rate = 44100


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
    f, t, spectro = spectrogram(
        signal, fs=fs, window=("kaiser", 8.0), nperseg=256, noverlap=64)
    return f, t, spectro


def plot_spectrogram(f, t, spectro):
    plt.pcolormesh(t, f, spectro)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def compute_amplitudes(spectro):
    return 20 * np.log10(np.abs(spectro))


# incertitude : use same frequency to find high cut and low cut or max
# over all frequencies ???
# test stop_amp value (proportional to beta)
def extract_syllabe(spectro, beta=20, stop_amp=40, all_freq=True):
    # compute amplitudes
    amplitudes = compute_amplitudes(spectro)
    # look for indice of max in spectrogram
    spectro_abs = np.abs(spectro)
    max_power = np.max(spectro_abs)
    max_ampli = 20 * math.log(max_power, 10)
    if max_ampli <= stop_amp:
        return -1, -1, -1, -1
    # find index of row and column corresponding to max power/amplitude
    f_max, t_max = np.unravel_index(spectro_abs.argmax(), spectro_abs.shape)
    # browse neighbor amplitudes until it has decreased enough
    # forward search
    high_cut = spectro.shape[1] - 1
    for t in range(spectro.shape[1]):
        if all_freq:
            curr_value = np.max(amplitudes[:, t_max + t])
        else:
            curr_value = amplitudes[f_max, t_max + t]
        if t_max + t >= spectro.shape[1] - 1:
            break
        if curr_value < max_ampli - beta:
            high_cut = t_max + t
            break
    # backward search
    low_cut = 0
    for t in range(spectro.shape[1]):
        if all_freq:
            curr_value = np.max(amplitudes[:, t_max - t])
        else:
            curr_value = amplitudes[f_max, t_max - t]
        if t_max - t <= 0:
            break
        if curr_value < max_ampli - beta:
            low_cut = t_max - t
            break
    return low_cut, high_cut, t_max, max_power


def extract_all_syllabes(spectro, beta=30, stop_amp=10,
                         n_max=100, all_freq=True):
    low_cuts = []
    high_cuts = []
    idxs_max = []
    max_powers = []
    spectro_bis = spectro.copy()

    for i in range(n_max):
        low_cut, high_cut, idx_max, max_power = extract_syllabe(
            spectro_bis, beta=beta, stop_amp=stop_amp, all_freq=all_freq)
        # this means that there are no longer syllabes to extract
        if low_cut == -1:
            break
        low_cuts.append(low_cut)
        high_cuts.append(high_cut)
        idxs_max.append(idx_max)
        max_powers.append(max_power)
        # put values to zero to avoid retrieving same syllabe multiple times
        spectro_bis[:, low_cut:high_cut + 1] = 0

    return low_cuts, high_cuts, idxs_max, max_powers


def gen_syllabes(sig, low, high):
    for i, idx_low in enumerate(low):
        idx_high = high[i]
        yield sig[idx_low * 192:(idx_high + 1) * 192]


def segmentation(sig, fs=sampling_rate, beta=70,
                 stop_amp=90, n_max=1000, all_freq=False):
    f, t, spectro = spectro_conv(sig, fs)
    low, high, _, _ = extract_all_syllabes(
        spectro, beta=beta, stop_amp=stop_amp, n_max=n_max, all_freq=all_freq)
    conv_ratio = 192000. / 44100.
    try:
        print("%d syllabes extracted" % len(low))
        print("Average length : %.2f ms" %
              (conv_ratio * (np.array(high) - np.array(low))).mean())
        print("Min length :  %.2f ms" %
              np.min(conv_ratio * (np.array(high) - np.array(low))))
        print("Max length :  %.2f ms" %
              np.max(conv_ratio * (np.array(high) - np.array(low))))
        syllabe_gen = gen_syllabes(sig, low, high)
        return syllabe_gen
    except:
        return [sig]
