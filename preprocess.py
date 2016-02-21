from scipy import signal

sampling_rate = 44100


class Preprocess(object):

    def __init__(self, width=10 / sampling_rate, ripple_db=60.,
                 cutoff_hz=5000):
        self.width = width
        self.ripple_db = ripple_db
        self.cutoff_hz = cutoff_hz
        return

    def fit(self, X, y=None):
        return

    def transform(self, X, y=None):
        for sig in X:
            N, beta = signal.kaiserord(self.ripple_db, self.width)
            taps = signal.firwin(
                N, self.cutoff_hz * 2 / sampling_rate, window=('kaiser', beta))
            yield signal.lfilter(taps, 1.0, sig)
        return

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=None)
