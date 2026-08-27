"""Historical broad Gradient Boosting search configuration."""


SEARCH_SPACE = {
    "preprocessor__numeric__imputer__strategy": ["mean", "median"],
    "preprocessor__categorical__onehot__drop": [None, "first"],
    "preprocessor__categorical__onehot__min_frequency": [None, 2, 5, 10, 20],
    "model__loss": ["squared_error", "huber", "absolute_error"],
    "model__n_estimators": [100, 150, 250, 400, 600, 800],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
    "model__max_depth": [2, 3, 4, 5, 6, 8],
    "model__min_samples_split": [2, 4, 6, 10, 15],
    "model__min_samples_leaf": [1, 2, 3, 5, 8, 12],
    "model__subsample": [0.6, 0.75, 0.9, 1.0],
    "model__max_features": [None, 1.0, 0.75, 0.5, "sqrt", "log2"],
    "model__alpha": [0.8, 0.9, 0.95],
}
SEARCH_ITERATIONS = 100

