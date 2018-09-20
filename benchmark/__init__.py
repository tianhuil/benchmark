from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

from data import dataset_iter


class Benchmark(object):
    def __init__(self, estimator):
        self.estimator = estimator

    def cross_val(self, **kwargs):
        for dataset in dataset_iter():
            X, y = shuffle(dataset.data, dataset.target, random_state=42)
            scores = cross_val_score(self.estimator, X, y, **kwargs)
            print
            print "JSON_DATA_FIELD", {
                "dataset": dataset.name,
                "estimator": self.estimator.__class__.__name__,
                "scores": list(scores),
                "kwargs": kwargs,
            }
