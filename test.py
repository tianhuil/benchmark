from sklearn.linear_model import LinearRegressionCV

from benchmark import Benchmark

def test(capsys):
	with capsys.disabled():
		Benchmark(LinearRegressionCV()).cross_val(cv=5)
