"""Historical broad KNN search configuration."""


SEARCH_SPACE = {
    "preprocessor__numeric__imputer__strategy": ["mean", "median"],
    "preprocessor__numeric__scaler__with_mean": [True, False],
    "preprocessor__numeric__scaler__with_std": [True, False],
    "preprocessor__categorical__onehot__drop": [None, "first"],
    "preprocessor__categorical__onehot__min_frequency": [None, 2, 5, 10, 20],
    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50],
    "model__weights": ["uniform", "distance"],
    "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "model__leaf_size": [10, 20, 30, 40, 60, 100],
    "model__metric": ["minkowski", "manhattan", "euclidean", "chebyshev"],
    "model__p": [1, 2, 3],
}
SEARCH_ITERATIONS = 60

