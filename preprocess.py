from scipy import signal
from segmentation import *
from main import *


sampling_rate = 44100


class Preprocess(object):

    def __init__(self, width=10 / sampling_rate, ripple_db=60.,
                 cutoff_hz=5000):
        self.width = width
        self.ripple_db = ripple_db
        self.cutoff_hz = cutoff_hz
        self.true_idx = []
        self.debug = None
        self.n_too_small = 0
        self.debug_i = 0
        return

    def fit(self, X, y=None):
        return

    def transform(self, X, y=None):
        self.true_idx = []
        self.n_too_small = 0
        for i, sig in enumerate(X):
            N, beta = signal.kaiserord(self.ripple_db, self.width)
            taps = signal.firwin(
                N, self.cutoff_hz * 2 / sampling_rate, window=('kaiser', beta))
            filtered_sig = signal.lfilter(taps, 1.0, sig)
            filtered_sig = np.round(filtered_sig).astype(np.int16)
            self.debug = filtered_sig  # To remove
            self.debug_i = i  # To remove
            for syllabe in segmentation(filtered_sig):
                if len(syllabe) < 44100 * 0.04:
                    self.n_too_small += 1
                    continue
                self.true_idx.append(i)
                print(len(syllabe))
                yield runOpenSmile(syllabe)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=None)
