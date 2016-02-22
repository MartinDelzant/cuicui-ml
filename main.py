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

PATH_TO_CONFIG = 'OpenSmileConfig/MFCC12_0_D_A.conf'
PATH_TO_CONFIG_PLP = 'OpenSmileConfig/PLP_0_D_A.conf'

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


def augment_data(sig, length=44100):
    if len(sig) < length:
        return [sig]
    else:
        n = len(sig)
        augment_1 = np.split(sig, range(0, n, length)[1:-1])
        augment_2 = np.split(sig[length // 2:],
                             range(0, n - length // 2, length)[1:-1])
        augment_1.extend(augment_2)
        return augment_1


def augmented_data_iter(length=44100):
    for i, dat in enumerate(read_wav_iter()):
        curr_label = y[i]
        for aug_dat in augment_data(dat, length=length):
            yield (i, curr_label, aug_dat)


def get_features_welch(sig, min_range=1000, max_range=10000,
                       window_length=500, nperseg=256 * 4, **kwargs):
    # On prend des moyennes, max, std pour voir :
    f, Pxx = signal.welch(sig, sampling_rate, nperseg=nperseg, **kwargs)
    list_sep = np.arange(min_range, max_range, window_length)
    features = [
        Pxx.mean(),
        Pxx.std()]
    for mini, maxi in zip(list_sep[:-2], list_sep[2:]):
        temp_Pxx = Pxx[np.logical_and(mini < f, f <= maxi)]
        features.append(temp_Pxx.mean())
        features.append(temp_Pxx.std())
    return features


def aug_cross_val(true_idx, initial_cv=cv):
    """
    true_idx : Array containing the idx
    of the current data in the original dataset

    initial_cv : cv object from initial dataset
    """
    if initial_cv is None:
        raise ValueError
    for train, test in initial_cv:
        matching_train_idx = np.where(
            [elem in train for elem in true_idx])[0]
        matching_test_idx = np.where(
            [elem in test for elem in true_idx])[0]
        yield matching_train_idx, matching_test_idx


def aggregatePrediction(y_pred, true_idx):
    y_pred_agg = []
    for i in range(np.max(true_idx) + 1):
        idxs = np.where(true_idx == i)[0]
        if len(idxs) > 0:
            y_pred_agg.append(np.argmax(np.bincount(y_pred[idxs])))
        else:
            print(i)
            y_pred_agg.append(26)
    return np.array(y_pred_agg)


def create_MFCC(path_to_config=PATH_TO_CONFIG, train=True, verbose=0):
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
    if train is None:
        complete_path = path
    elif train:
        complete_path = './MFCC/train/' + path
    else:
        complete_path = './MFCC/test/' + path
    df = pd.read_csv(complete_path, sep=';')
    if type(drop) is list:
        if not np.all([col in df.columns for col in drop]):
            for col in drop:
                if col in df.columns:
                    df = df.drop(col, axis=1)
        else:
            df = df.drop(drop, axis=1)
    return df


def create_PLP(path_to_config=PATH_TO_CONFIG_PLP, train=True, verbose=0):
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
                           'PLP' + output_dir + '/PLP_' + out_file_name
            subprocess.call(line_command, shell=True)


def read_PLP(path, train=True, drop=['frameIndex', 'frameTime']):
    if train is None:
        complete_path = path
    elif train:
        complete_path = './PLP/train/' + path
    else:
        complete_path = './PLP/test/' + path
    df = pd.read_csv(complete_path, sep=';')
    if type(drop) is list:
        if not np.all([col in df.columns for col in drop]):
            for col in drop:
                if col in df.columns:
                    df = df.drop(col, axis=1)
        else:
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


def getStatsOnPlp(train=True, drop_col=['frameIndex', 'frameTime'],
                   drop_line=['count'], use_kurt=True, use_skew=True):
    dir_path = 'train/' if train else 'test/'
    result = []
    for _, _, files in os.walk('./PLP/' + dir_path):
        for i, file_name in enumerate(sorted(files)):
            result.append(aggregatePlp(
                read_PLP(file_name,
                          train=train,
                          drop=drop_col),
                drop=drop_line,
                use_kurt=use_kurt,
                use_skew=use_skew)
            )
    return np.array(result)


def runOpenSmile(sig, path_to_config=PATH_TO_CONFIG,
                 sampling_rate=sampling_rate,
                 clean_file=True, drop=['frameIndex', 'frameTime']):
    # from digital signal, calculate MFCC with openSmile
    tmp_idx = 0
    while os.path.isfile('tmp_sig' + str(tmp_idx)):
        tmp_idx += 1
    sig_filename = 'tmp_sig' + str(tmp_idx) + '.wav'
    out_filename = sig_filename + '_out'
    wavfile.write(sig_filename, sampling_rate, sig)
    line_command = 'SMILExtract -C ' + path_to_config + \
                   ' -I ' + sig_filename + \
                   ' -O ' + out_filename
    subprocess.call(line_command, shell=True)
    df = read_MFCC(out_filename, train=None, drop=drop)
    if clean_file:
        os.remove(sig_filename)
        os.remove(out_filename)
    return df


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


def savePrediction(filename, y_pred):
    filenames = list(
        map(lambda x: x.split('.')[0],
            list(read_wav_iter(test_path, return_file_name=True)))
    )
    df_pred = pd.DataFrame({'ID': filenames, 'Class': y_pred})
    df_pred.index = df_pred['ID']
    df_pred.drop('ID', axis=1).to_csv(filename, sep=";")
    return

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
    X_kmean = []
    true_idx_kmean = []

    for i in range(np.max(true_idx)+1):
        idxs = np.where(true_idx == i)[0]
        if len(idxs) > 3:
            X_temp = X[idxs,:]
            km = KMeans(n_clusters=3)
            clusters = km.fit_predict(X_temp)
            for j in np.unique(clusters):
                cl_idxs = np.where(clusters == j)[0]
                if len(cl_idxs)>0:
                    X_to_add = X_temp[cl_idxs,:].mean(axis=0)
                    if X_to_add.shape[0] != 351:
                        print(X_to_add.shape, len(cl_idxs))
                    X_kmean.append(X_to_add)
                    true_idx_kmean.append(i)
        elif len(idxs) >0:
            X_to_add = X_temp.mean(axis=0)
            if X_to_add.shape[0] != 351:
                print(X_to_add.shape, len(idxs), 'second')
            X_kmean.append(X_to_add)
            true_idx_kmean.append(i)
        else:
            print(i)
