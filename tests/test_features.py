# tests/test_build_features.py
#
# What this file tests:
#   src/features/make_features.py :  build_preprocessor()
#
# These tests make sure our preprocessing step correctly
# transforms raw data into a format models can actually use.
#
# Fixtures used (defined in conftest.py):
#   sample_X  to  fake feature data, no Churn column

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer


# -------------------------------------------------------
# TESTS FOR build_preprocessor()
# -------------------------------------------------------

class TestBuildPreprocessor:

    def test_returns_a_column_transformer(self):
        """
        build_preprocessor() must return a sklearn ColumnTransformer.
        This is the object that applies different transformations
        to numeric vs categorical columns separately.
        If this fails, something is wrong with the function itself.
        """
        from src.features.make_features import build_preprocessor

        prep = build_preprocessor()

        assert isinstance(prep, ColumnTransformer)

    def test_output_has_same_number_of_rows(self, sample_X):
        """
        After transformation, the number of rows must stay the same.
        If we put in 100 rows, we must get 100 rows back.
        Transformation should never add or remove samples.
        """
        from src.features.make_features import build_preprocessor

        prep = build_preprocessor()
        X_transformed = prep.fit_transform(sample_X)

        assert X_transformed.shape[0] == len(sample_X)

    def test_output_has_more_columns_than_input(self, sample_X):
        """
        OneHotEncoder expands categorical columns into multiple binary columns.
        So the output should have MORE columns than the input (10 columns).
        If column count stayed the same or shrunk, encoding didn't happen.
        """
        from src.features.make_features import build_preprocessor

        prep = build_preprocessor()
        X_transformed = prep.fit_transform(sample_X)

        assert X_transformed.shape[1] > sample_X.shape[1]

    def test_all_output_values_are_finite(self, sample_X):
        """
        After scaling and encoding, every single value must be a
        real finite number - no NaN, no infinity.
        NaN values would silently break model training.
        """
        from src.features.make_features import build_preprocessor

        prep = build_preprocessor()
        X_transformed = prep.fit_transform(sample_X)

        assert np.all(np.isfinite(X_transformed))

    def test_unseen_categories_do_not_cause_error(self, sample_X):
        """
        In production, our model might see a gender value it never
        saw during training. handle_unknown='ignore' in your OneHotEncoder
        should handle this gracefully instead of crashing.
        We simulate this by fitting on normal data, then transforming
        data that has a brand new category value.
        """
        from src.features.make_features import build_preprocessor

        prep = build_preprocessor()
        prep.fit(sample_X)

        X_unseen = sample_X.copy()
        X_unseen["Gender"] = "NonBinary"   # a value never seen during fit

        # This line must NOT raise an exception
        prep.transform(X_unseen)