import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal, stats
import os
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.cross_validation import (cross_val_score,
                                      StratifiedKFold,
                                      cross_val_predict)
from joblib import Parallel, delayed
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
y = labels.Class.values
cv = StratifiedKFold(y, n_folds=7, shuffle=True, random_state=42)
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


def read_MFCC(path, train=True, drop=['frameIndex', 'frameTime']):
    if train:
        complete_path = './MFCC/train/' + path
    else:
        complete_path = './MFCC/test/' + path
    df = pd.read_csv(complete_path, sep=';')
    if drop:
        df = df.drop(drop, axis=1)
    return df


def aggregateMfcc(df, drop=['count'], use_kurt=True,
                  use_skew=True):
    # describe donne -> min max mean std 25% etc, pour toutes les col
    describe = df.describe().drop(drop)
    shape = describe.shape
    result = describe.values.reshape(shape[0] * shape[1])
    if use_kurt:
        kurt = stats.kurtosis(df, axis=0)
        assert(len(kurt) == shape[1])
        result = np.hstack((result, kurt))
    if use_skew:
        skew = stats.skew(df, axis=0)
        assert(len(skew) == shape[1])
        result = np.hstack((result, skew))
    return result


def getStatsOnMfcc(train=True, drop_col=['frameIndex', 'frameTime'],
                   drop_line=['count'], use_kurt=True, use_skew=True):
    dir_path = 'train/' if train else 'test/'
    result = []
    for _, _, files in os.walk('./MFCC/' + dir_path):
        for i, file_name in enumerate(sorted(files)):
            result.append(aggregateMfcc(
                                       read_MFCC(file_name,
                                                 train=train,
                                                 drop=drop_col),
                                       drop=drop_line,
                                       use_kurt=use_kurt,
                                       use_skew=use_skew)
                          )
    return np.array(result)


def n_estimators_path(model, n_estimators_range, X_train,
                      y_train, X_test, y_test, imputer=None):
    """
    Given a gdb model,
    computes the cross-val score
    for every value of n_estimators in n_estimators_range
    """
    result = []
    if imputer is not None:
        tempX_train = imputer.fit_transform(X_train, y_train)
        tempX_test = imputer.transform(X_test)
    for n_estimators in n_estimators_range:
        model.set_params(n_estimators=n_estimators)
        if imputer is None:
            model.fit(X_train, y_train)
            result.append(model.score(X_test, y_test))
        else:
            model.fit(tempX_train, y_train)
            result.append(model.score(tempX_test, y_test))
    return result


def cross_val_gdb(X, y, cv=cv, n_estimators_range=range(50, 1201, 50),
                  n_jobs=1, imputer=None, **kwargs):
    """
    Launches in parallel n_estimators_path on each fold of the cv object
    Computes the mean across all folds in the end (stage-wise)
    """
    all_res = Parallel(n_jobs=n_jobs)(
                            delayed(n_estimators_path)(
                                GradientBoostingClassifier(
                                    warm_start=True, **kwargs
                                ),
                                n_estimators_range,
                                X[train, :], y[train],
                                X[test, :], y[test],
                                imputer=imputer
                            ) for train, test in cv)
    res = np.mean(all_res, axis=0)
    return res

if __name__ == '__main__':
    print('\nCalculating train features ...')
    X_welch = np.array([get_features_welch(x) for x in read_wav_iter()])

    print('test features...')
    X_welch_test = np.array(
        [get_features_welch(x) for x in read_wav_iter(test_path)])

    y = labels.Class.values

    model = RandomForestClassifier(n_estimators=800, max_features=5)
    print('Fitting ...')
    model.fit(X_welch, y)

    print('Predict ...')
    y_pred = model.predict(X_welch_test)

    filenames = list(
        map(lambda x: x.split('.')[0],
            list(read_wav_iter(test_path, return_file_name=True)))
    )

    df_pred = pd.DataFrame({'ID': filenames, 'Class': y_pred})
    df_pred.index = df_pred['ID']
    df_pred.drop('ID', axis=1).to_csv('truc_muche.csv', sep=";")
