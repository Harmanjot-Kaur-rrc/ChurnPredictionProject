# tests/test_make_dataset.py
#
# What this file tests:
#   src/data/make_dataset.py    load_data() and split_data()
#
# These tests make sure your data loading and splitting
# logic works correctly before any model ever sees the data.
#
# Fixtures used (defined in conftest.py):
#   sample_df  to full fake churn dataframe including the Churn column

import pytest
import pandas as pd
import numpy as np


# -------------------------------------------------------
# TESTS FOR load_data()
# -------------------------------------------------------

class TestLoadData:

    def test_returns_a_dataframe(self, tmp_path, sample_df):
        """
        tmp_path is a pytest built-in fixture that gives a
        temporary folder that gets deleted after the test.
        I save a CSV there and check load_data() gives back
        a pandas DataFrame - not a list, dict, or anything else.
        """
        from src.data.make_dataset import load_data

        csv_path = tmp_path / "churn.csv"
        sample_df.to_csv(csv_path, index=False)

        df = load_data(str(csv_path))

        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_original(self, tmp_path, sample_df):
        """
        Makes sure no rows are accidentally dropped or duplicated
        during loading. If we saved 100 rows, we expect 100 back.
        """
        from src.data.make_dataset import load_data

        csv_path = tmp_path / "churn.csv"
        sample_df.to_csv(csv_path, index=False)

        df = load_data(str(csv_path))

        assert len(df) == len(sample_df)

    def test_churn_column_exists(self, tmp_path, sample_df):
        """
        The target column 'Churn' must be present after loading.
        If it's missing, nothing else in our pipeline will work.
        """
        from src.data.make_dataset import load_data

        csv_path = tmp_path / "churn.csv"
        sample_df.to_csv(csv_path, index=False)

        df = load_data(str(csv_path))

        assert "Churn" in df.columns


# -------------------------------------------------------
# TESTS FOR split_data()
# -------------------------------------------------------

class TestSplitData:

    def test_total_rows_are_preserved(self, sample_df):
        """
        After splitting into train / val / test, all rows must
        still be accounted for. No row should be lost or duplicated.
        train + val + test should equal the original 100 rows.
        """
        from src.data.make_dataset import split_data

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df)

        assert len(X_train) + len(X_val) + len(X_test) == len(sample_df)

    def test_X_and_y_lengths_match_in_each_split(self, sample_df):
        """
        For every split, the number of feature rows (X) must equal
        the number of labels (y). A mismatch here would crash model training.
        """
        from src.data.make_dataset import split_data

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_df)

        assert len(X_train) == len(y_train)
        assert len(X_val)   == len(y_val)
        assert len(X_test)  == len(y_test)

    def test_churn_column_not_in_features(self, sample_df):
        """
        This is a data leakage check. The target column 'Churn'
        must NOT appear in X_train, X_val, or X_test.
        If it did, our model would be cheating during training.
        """
        from src.data.make_dataset import split_data

        X_train, X_val, X_test, *_ = split_data(sample_df)

        for split in (X_train, X_val, X_test):
            assert "Churn" not in split.columns

    def test_train_set_is_the_largest_split(self, sample_df):
        """
        With a 70/15/15 split, the training set must always be
        bigger than val and test. If it's not, our model is being
        trained on too little data.
        """
        from src.data.make_dataset import split_data

        X_train, X_val, X_test, *_ = split_data(sample_df)

        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)