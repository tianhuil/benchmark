from sklearn.linear_model import RidgeCV

from benchmark import Benchmark

def test(capsys):
	with capsys.disabled():
		Benchmark(RidgeCV()).cross_val(cv=5)
