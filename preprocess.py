from scipy import signal

sampling_rate = 44100

from segmentation import *
from main import *


class Preprocess(object):

    def __init__(self, width=10 / sampling_rate, ripple_db=60.,
                 cutoff_hz=5000):
        self.width = width
        self.ripple_db = ripple_db
        self.cutoff_hz = cutoff_hz
        self.true_idx = []
        return

    def fit(self, X, y=None):
        return

    def transform(self, X, y=None):
        for i, sig in enumerate(X):
            N, beta = signal.kaiserord(self.ripple_db, self.width)
            taps = signal.firwin(
                N, self.cutoff_hz * 2 / sampling_rate, window=('kaiser', beta))
            filtered_sig = signal.lfilter(taps, 1.0, sig)
            syl_gen = segmentation(filtered_sig)
            for syllabe in syl_gen:
                self.true_idx.append(i)
                yield runOpenSmile(syllabe)
        return

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=None)
