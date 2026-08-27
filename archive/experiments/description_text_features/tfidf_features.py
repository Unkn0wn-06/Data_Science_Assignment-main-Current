"""Outer-fold-local TF-IDF and deterministic TruncatedSVD transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


WORD_TFIDF_PARAMS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "max_features": 5000,
    "sublinear_tf": True,
}
CHAR_TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (3, 5),
    "min_df": 5,
    "max_features": 3000,
    "sublinear_tf": True,
}


class FoldTextTransformer:
    """Fit vocabulary, IDF, and SVD solely from supplied training text."""

    def __init__(self, n_components: int, analyzer: str = "word", random_state: int = 42):
        self.n_components = int(n_components)
        self.analyzer = analyzer
        self.random_state = random_state

    def fit(self, text):
        params = WORD_TFIDF_PARAMS if self.analyzer == "word" else CHAR_TFIDF_PARAMS
        self.vectorizer_ = TfidfVectorizer(**params)
        sparse = self.vectorizer_.fit_transform(pd.Series(text).fillna("").astype(str))
        if self.n_components >= min(sparse.shape):
            raise ValueError("SVD component count must be smaller than TF-IDF dimensions.")
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.svd_.fit(sparse)
        self.vocabulary_size_ = len(self.vectorizer_.vocabulary_)
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_.copy()
        return self

    def transform(self, text, index=None) -> pd.DataFrame:
        sparse = self.vectorizer_.transform(pd.Series(text).fillna("").astype(str))
        dense = self.svd_.transform(sparse)
        columns = [f"description_svd_{number:02d}" for number in range(1, self.n_components + 1)]
        return pd.DataFrame(dense, columns=columns, index=index)

    def fit_transform_pair(self, train_text, validation_text, train_index=None, validation_index=None):
        self.fit(train_text)
        return (
            self.transform(train_text, train_index),
            self.transform(validation_text, validation_index),
        )
