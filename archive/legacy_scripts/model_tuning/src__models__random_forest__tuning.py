"""Historical broad Random Forest search configuration."""


SEARCH_SPACE = {
    "preprocessor__numeric__imputer__strategy": ["mean", "median"],
    "preprocessor__categorical__onehot__drop": [None, "first"],
    "preprocessor__categorical__onehot__min_frequency": [None, 2, 5, 10, 20],
    "model__n_jobs": [1],
    "model__n_estimators": [150, 250, 400, 600, 800],
    "model__criterion": ["squared_error", "poisson"],
    "model__max_depth": [None, 8, 12, 16, 24, 32],
    "model__min_samples_split": [2, 4, 6, 10, 15],
    "model__min_samples_leaf": [1, 2, 3, 5, 8],
    "model__max_features": [1.0, 0.75, 0.5, "sqrt", "log2"],
    "model__bootstrap": [True, False],
    "model__max_leaf_nodes": [None, 50, 100, 200, 400],
}
SEARCH_ITERATIONS = 80

