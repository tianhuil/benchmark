from sklearn.model_selection import cross_val_score

from data import dataset_iter

class Benchmark(object):
	def __init__(self, estimator):
		self.estimator = estimator

	def cross_val(self, **kwargs):
		for dataset in dataset_iter():
			scores = cross_val_score(self.estimator, dataset.data, dataset.target, **kwargs)
			print
			print "JSON_DATA_FIELD", {
				"dataset": dataset.name,
				"estimator": self.estimator.__class__.__name__,
				"scores": list(scores),
				"kwargs": kwargs,
			}
