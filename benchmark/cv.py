from sklearn.base import is_classifier
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import KFold, StratifiedKFold

RANDOM_STATE = 42

def cv(n_splits, y, estimator):
    if (is_classifier(estimator) and (y is not None) and
            (type_of_target(y) in ('binary', 'multiclass'))):
        return StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        return KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
