from sklearn.linear_model import LinearRegression

from benchmark import Benchmark

def test(capsys):
	with capsys.disabled():
		Benchmark(LinearRegression()).cross_val(cv=5)
