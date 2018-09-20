from sklearn.model_selection import cross_val_score

from data import dataset_iter
from cv import cv


class Benchmark(object):
	def __init__(self, estimator):
		self.estimator = estimator

	def cross_val(self, **kwargs):

		for dataset in dataset_iter():
			n_splits = kwargs.pop('n_splits', 5)
			_cv = cv(n_splits, dataset.target, self.estimator)
			scores = cross_val_score(self.estimator, dataset.data, dataset.target, cv=_cv, **kwargs)
			kwargs.update({'n_splits': n_splits})
			print
			print "JSON_DATA_FIELD", {
				"dataset": dataset.name,
				"estimator": self.estimator.__class__.__name__,
				"scores": list(scores),
				"kwargs": kwargs,
			}
