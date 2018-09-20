from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression

LOADERS = [
    datasets.load_boston,
    datasets.load_iris,
    datasets.load_diabetes,
    datasets.load_digits,
    datasets.load_linnerud,
    # datasets.load_wine,
    datasets.load_breast_cancer,
]

def _dataset_iter():
	for loader in LOADERS:
		yield loader()

def benchmark_estimator(Estimator, **kwargs):
	est = Estimator()
	for data in _dataset_iter():
		scores = cross_val_score(est, data.data, data.target, **kwargs)
		print
		print scores
		print

def test(capsys):
	with capsys.disabled():
		benchmark_estimator(LinearRegression)
