# tests/test_voting.py
#
# What this file tests:
#   src/models/voting.py  →  build_voting()
#
# These tests make sure your VotingClassifier is assembled
# correctly from the individually trained models.
#
# Fixtures used (defined in conftest.py):
#   fake_trained_models  →  dict of fitted pipeline stand-ins with .best_estimator_

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier


# -------------------------------------------------------
# TESTS FOR build_voting()
# -------------------------------------------------------

class TestBuildVoting:

    def test_returns_a_voting_classifier(self, fake_trained_models):
        """
        build_voting() must return a sklearn VotingClassifier object.
        If it returns anything else, the rest of our pipeline
        that calls .fit() and .predict_proba() on it will crash.
        """
        from src.models.voting import build_voting

        vc = build_voting(fake_trained_models)

        assert isinstance(vc, VotingClassifier)

    def test_uses_soft_voting(self, fake_trained_models):
        """
        Soft voting uses predicted probabilities to make the final
        decision, which is better than hard voting for churn prediction
        because it captures confidence levels, not just yes/no.
        This test makes sure we didn't accidentally use hard voting.
        """
        from src.models.voting import build_voting

        vc = build_voting(fake_trained_models)

        assert vc.voting == "soft"

    def test_has_exactly_four_estimators(self, fake_trained_models):
        """
        Our voting ensemble should contain exactly 4 base models:
        Logistic, RandomForest, GradientBoosting, XGBoost.
        MLP is excluded from voting in your voting.py.
        If this count is wrong, a model was accidentally added or dropped.
        """
        from src.models.voting import build_voting

        vc = build_voting(fake_trained_models)

        assert len(vc.estimators) == 4

    def test_xgboost_has_higher_weight(self, fake_trained_models):
        """
        In our voting.py we gave XGBoost a weight of 2 and
        all others a weight of 1. This test checks that the weights
        list contains a 2, meaning XGBoost's vote counts double.
        This is intentional - XGBoost typically performs best on
        tabular data like this churn dataset.
        """
        from src.models.voting import build_voting

        vc = build_voting(fake_trained_models)

        assert 2 in vc.weights