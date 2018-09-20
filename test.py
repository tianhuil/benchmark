from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from data import dataset_iter

def benchmark_estimator(Estimator, **kwargs):
	est = Estimator()
	for dataset in dataset_iter():
		scores = cross_val_score(est, dataset.data, dataset.target, **kwargs)
		print
		print {
			"dataset": dataset.name,
			"estimator": est.__class__.__name__,
			"scores": list(scores),
			"kwargs": kwargs,
		}

def test(capsys):
	with capsys.disabled():
		benchmark_estimator(LinearRegression, cv=5)
