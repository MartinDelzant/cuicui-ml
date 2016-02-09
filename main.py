import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import os
from sklearn.ensemble import RandomForestClassifier
import subprocess

train_path = './AmazonBird50_training_input/'
test_path = './AmazonBird50_testing_input/'

JOHAN_PATH_TO_CONFIG = '../../Programmes/openSMILE-2.1.0/config' + \
                       '/MFCC12_0_D_A.conf'

MARTIN_PATH_TO_CONFIG = '/home/martin/Applications/openSMILE-2.2rc1' + \
                        '/config/MFCC12_0_D_A.conf'
sampling_rate = 44100  # Hz

labels = pd.read_csv(
    './challenge_output_data_training_file_classify_bird_songs.csv', sep=";")

# Attention -> Données assez grosses ...
# Préférez l'itérateur en dessous !!!


def read_wav(dir_path=train_path, verbose=0, max_files=None):
    raw_data = []
    for _, _, files in os.walk(dir_path):
        for i, file_name in enumerate(sorted(files)[:max_files]):
            if verbose:
                print(file_name)
            _, data = wavfile.read(dir_path + file_name)
            raw_data.append(data.tolist())
    return raw_data


def read_wav_iter(dir_path=train_path, verbose=0, return_file_name=False):
    for _, _, files in os.walk(dir_path):
        for i, file_name in enumerate(sorted(files)):
            if return_file_name:
                yield file_name
            else:
                if verbose:
                    print(file_name)
                _, data = wavfile.read(dir_path + file_name)
                yield data


def get_features_welch(sig, min_range=1000, max_range=10000,
                       nperseg=256 * 4, **kwargs):
    # On prend des moyennes, max, std pour voir :
    f, Pxx = signal.welch(sig, sampling_rate, nperseg=nperseg, **kwargs)
    list_sep = np.arange(min_range, max_range, 500)
    features = [
        Pxx.mean(),
        Pxx.std()]
    for mini, maxi in zip(list_sep[:-2], list_sep[2:]):
        temp_Pxx = Pxx[np.logical_and(mini < f, f <= maxi)]
        features.append(temp_Pxx.mean())
        features.append(temp_Pxx.std())
    return features


def create_MFCC(path_to_config, train=True, verbose=0):
    if train:
        dir_path = train_path
        output_dir = '/train'
    else:
        dir_path = test_path
        output_dir = '/test'
    for _, _, files in os.walk(dir_path):
        for i, file_name in enumerate(sorted(files)):
            if verbose:
                print(file_name)
            out_file_name = file_name.replace('.wav', '.csv')
            line_command = 'SMILExtract -C ' + path_to_config + \
                           ' -I ' + dir_path + file_name + \
                           ' -O ' + \
                           'MFCC' + output_dir + '/MFCC_' + out_file_name
            subprocess.call(line_command, shell=True)


print('train features')
X_welch = np.array([get_features_welch(x) for x in read_wav_iter()])

print('test features')
X_welch_test = np.array(
    [get_features_welch(x) for x in read_wav_iter(test_path)])

y = labels.Class.values

model = RandomForestClassifier(n_estimators=800, max_features=5)
print('Fitting')
model.fit(X_welch, y)

print('Predict')
y_pred = model.predict(X_welch_test)

filenames = list(
    map(lambda x: x.split('.')[0],
        list(read_wav_iter(test_path, return_file_name=True)))
)

df_pred = pd.DataFrame({'ID': filenames, 'Class': y_pred})
df_pred.index = df_pred['ID']
df_pred.drop('ID', axis=1).to_csv('truc_muche.csv')
