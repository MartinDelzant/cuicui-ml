from scipy import signal
from segmentation import *
from main import *
import time
from joblib import Parallel, delayed


sampling_rate = 44100


def filter_sig(sig, width=10 / sampling_rate, ripple_db=60.,
               cutoff_hz=6000):
    N, beta = signal.kaiserord(ripple_db, width)
    taps = signal.firwin(
        N, cutoff_hz * 2 / sampling_rate, window=('kaiser', beta))
    filtered_sig = signal.lfilter(taps, 1.0, sig)
    filtered_sig = np.round(filtered_sig).astype(np.int16)
    return filtered_sig


class Preprocess(object):

    def __init__(self, width=10 / sampling_rate, ripple_db=60.,
                 cutoff_hz=6000, use_kurt=True, use_skew=True,
                 n_jobs=1):
        self.width = width
        self.ripple_db = ripple_db
        self.cutoff_hz = cutoff_hz
        self.use_kurt = use_kurt
        self.use_skew = use_skew
        self.true_idx = []
        self.debug = None
        self.n_too_small = 0
        self.debug_i = 0
        self.filter_time = 0
        self.segment_time = 0
        self.smileTime = 0
        self.mfcc_shape = []
        self.n_jobs = 1
        return

    def fit(self, X, y=None):
        return

    def transform_one_syllabe(self, syllabe, i):
        print(len(syllabe))
        t0 = time.time()
        df = runOpenSmile(syllabe)
        self.smileTime += time.time() - t0
        self.mfcc_shape.append(df.shape[0])
        return aggregateMfcc(df, use_kurt=self.use_kurt,
                             use_skew=self.use_skew)

    def transform_one_signal(self, sig, i):
        if i % 100 == 0:
            print('\n' * 30)
            print(i)
            print('\n' * 30)
        t0 = time.time()
        filtered_sig = filter_sig(sig, width=self.width,
                                  ripple_db=self.ripple_db,
                                  cutoff_hz=self.cutoff_hz)
        self.filter_time += time.time() - t0
        self.debug = filtered_sig  # To remove
        self.debug_i = i           # To remove
        t0 = time.time()
        segmented_sig = segmentation(filtered_sig)
        self.segment_time += time.time() - t0
        segmented_sig = list(filter(lambda x: len(x) > 1600,
                                    segmented_sig))
        if not segmented_sig:
            segmented_sig = [filtered_sig]
        self.true_idx.extend([i] * len(segmented_sig))
        return Parallel(n_jobs=self.n_jobs)(
            delayed(transform_one_syllabe(self, syllabe, i))
            for syllabe in segmented_sig)

    def transform(self, X, y=None):
        t0_tot = time.time()
        self.true_idx = []
        self.n_too_small = 0
        self.filter_time = 0
        self.segment_time = 0
        self.smileTime = 0
        for i, sig in enumerate(X):
            if i % 100 == 0:
                print('\n' * 30)
                print(i)
                print('\n' * 30)
            t0 = time.time()
            filtered_sig = filter_sig(sig, width=self.width,
                                      ripple_db=self.ripple_db,
                                      cutoff_hz=self.cutoff_hz)
            self.filter_time += time.time() - t0
            self.debug = filtered_sig  # To remove
            self.debug_i = i           # To remove
            t0 = time.time()
            segmented_sig = segmentation(filtered_sig)
            self.segment_time += time.time() - t0
            segmented_sig = list(filter(lambda x: len(x) > 1600,
                                        segmented_sig))
            if not segmented_sig:
                segmented_sig = [filtered_sig]
            for syllabe in segmented_sig:
                print(len(syllabe))
                t0 = time.time()
                df = runOpenSmile(syllabe)
                self.smileTime += time.time() - t0
                self.mfcc_shape.append(df.shape[0])
                self.true_idx.append(i)
                yield aggregateMfcc(df, use_kurt=self.use_kurt,
                                    use_skew=self.use_skew)

        print('Filter time :', self.filter_time)
        print('Segment time:', self.segment_time)
        print('Smile time:', self.smileTime)
        print('Total time:', time.time() - t0_tot)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=None)
