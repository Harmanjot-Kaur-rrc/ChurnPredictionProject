# tests/conftest.py
#
# Shared fixtures for the entire test suite.
# pytest automatically discovers this file and makes all
# fixtures defined here available to every test file
# without needing to import anything.
#
# Fixtures moved here from:
#   test_make_dataset.py   :  sample_df
#   test_build_features.py : sample_X
#   test_evaluate.py       :  sample_X, sample_y, trained_pipeline
#   test_voting.py         : sample_X, sample_y, fake_trained_models

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# -------------------------------------------------------
# CORE DATA FIXTURES
# -------------------------------------------------------

@pytest.fixture
def sample_df():
    """
    Full fake churn dataframe (100 rows) including the target column.
    Used by: test_make_dataset.py
    Mirrors the exact columns of the real Kaggle dataset so tests
    are realistic without needing the actual CSV file.
    """
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Age":               np.random.randint(18, 70, n),
        "Tenure":            np.random.randint(1, 60, n),
        "Usage Frequency":   np.random.randint(1, 30, n),
        "Support Calls":     np.random.randint(0, 10, n),
        "Payment Delay":     np.random.randint(0, 30, n),
        "Total Spend":       np.random.uniform(100, 1000, n),
        "Last Interaction":  np.random.randint(1, 30, n),
        "Gender":            np.random.choice(["Male", "Female"], n),
        "Subscription Type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "Contract Length":   np.random.choice(["Monthly", "Quarterly", "Annual"], n),
        "Churn":             np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_X():
    """
    Fake feature data only — no Churn column (100 rows).
    Used by: test_build_features.py, test_evaluate.py, test_voting.py
    This is what the preprocessor and model actually receive
    during a real pipeline run after the target column is separated.
    """
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Age":               np.random.randint(18, 70, n),
        "Tenure":            np.random.randint(1, 60, n),
        "Usage Frequency":   np.random.randint(1, 30, n),
        "Support Calls":     np.random.randint(0, 10, n),
        "Payment Delay":     np.random.randint(0, 30, n),
        "Total Spend":       np.random.uniform(100, 1000, n),
        "Last Interaction":  np.random.randint(1, 30, n),
        "Gender":            np.random.choice(["Male", "Female"], n),
        "Subscription Type": np.random.choice(["Basic", "Standard", "Premium"], n),
        "Contract Length":   np.random.choice(["Monthly", "Quarterly", "Annual"], n),
    })


@pytest.fixture
def sample_y():
    """
    Fake binary churn labels (0 = stayed, 1 = churned) — 100 rows.
    Used by: test_evaluate.py, test_voting.py
    """
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100), name="Churn")


# -------------------------------------------------------
# SHARED PREPROCESSOR HELPER
# Defined once here so trained_pipeline and fake_trained_models
# both build the exact same preprocessing steps.
# -------------------------------------------------------

def _make_preprocessor():
    """
    Returns a fresh ColumnTransformer that scales numeric columns
    and one-hot encodes categorical columns.
    Called internally by trained_pipeline and fake_trained_models.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), [
            "Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"
        ]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [
            "Gender", "Subscription Type", "Contract Length"
        ])
    ])


# -------------------------------------------------------
# MODEL FIXTURES
# -------------------------------------------------------

@pytest.fixture
def trained_pipeline(sample_X, sample_y):
    """
    A full sklearn Pipeline (preprocessor + LogisticRegression)
    that has already been fitted on sample_X / sample_y.
    Used by: test_evaluate.py
    We use LogisticRegression because it trains in milliseconds.
    We are NOT testing model accuracy — only that evaluate_model()
    returns the correct structure and value ranges.
    """
    pipe = Pipeline([
        ("preprocessor", _make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(sample_X, sample_y)
    return pipe


@pytest.fixture
def fake_trained_models(sample_X, sample_y):
    """
    Simulates the dict that train_models() returns after GridSearchCV.
    Each value has a .best_estimator_ attribute pointing to a fitted pipeline.
    Used by: test_voting.py

    We use lightweight LogisticRegression stand-ins so tests run in
    seconds rather than minutes (no XGBoost / RandomForest training here).
    The type() call dynamically creates a minimal class that satisfies
    the .best_estimator_ contract expected by build_voting().
    """
    fake_models = {}
    for key in ["Logistic", "RandomForest", "GradientBoosting", "XGBoost"]:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000))
        ])
        pipe.fit(sample_X, sample_y)
        fake_models[key] = type("FakeGridSearch", (), {"best_estimator_": pipe})()

    return fake_models