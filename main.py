import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import os

dir_path = './AmazonBird50_training_input/'
sampling_rate = 44100  # Hz

labels = pd.read_csv(
    './challenge_output_data_training_file_classify_bird_songs.csv', sep=";")

# Attention -> Données assez grosses ...
# Préférez l'itérateur en dessous !!!


def read_wav(dir_path=dir_path, verbose=0, max_files=None):
    raw_data = []
    for _, _, files in os.walk(dir_path):
        for i, file_name in enumerate(sorted(files)[:max_files]):
            if verbose:
                print(file_name)
            _, data = wavfile.read(dir_path + file_name)
            raw_data.append(data.tolist())
    return raw_data


def read_wav_iter(dir_path=dir_path, verbose=0):
    for _, _, files in os.walk(dir_path):
        for i, file_name in enumerate(sorted(files)):
            if verbose:
                print(file_name)
            _, data = wavfile.read(dir_path + file_name)
            yield data

data_iter = read_wav_iter()