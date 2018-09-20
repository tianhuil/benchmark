from collections import namedtuple

from sklearn import datasets

Dataset = namedtuple("Dataset", ["name", "data", "target", "tags"])

def make_sklearn_dataset(name, loader):
    data = loader()
    return Dataset(name, data.data, data.target)

_datasets = [
    make_sklearn_dataset("boston", datasets.load_boston),
    make_sklearn_dataset("iris", datasets.load_iris),
    make_sklearn_dataset("diabetes", datasets.load_diabetes),
    make_sklearn_dataset("digits", datasets.load_digits),
    make_sklearn_dataset("linnerud", datasets.load_linnerud),
    make_sklearn_dataset("wine", datasets.load_wine),
    make_sklearn_dataset("breast_cancer", datasets.load_breast_cancer),
]

def dataset_iter():
    for dataset in _datasets:
        yield dataset
