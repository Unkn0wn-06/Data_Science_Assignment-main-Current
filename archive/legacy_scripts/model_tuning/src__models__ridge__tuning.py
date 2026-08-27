"""Historical broad Ridge Regression search configuration."""


SEARCH_SPACE = {
    "preprocessor__numeric__imputer__strategy": ["mean", "median"],
    "preprocessor__numeric__scaler__with_mean": [True, False],
    "preprocessor__categorical__onehot__drop": [None, "first"],
    "preprocessor__categorical__onehot__min_frequency": [None, 2, 5, 10, 20],
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0],
    "model__fit_intercept": [True, False],
    "model__solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
    "model__tol": [1e-5, 1e-4, 1e-3, 1e-2],
    "model__max_iter": [2000, 5000, 10000],
}
SEARCH_ITERATIONS = 60

