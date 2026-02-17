# tests/test_evaluate.py
#
# What this file tests:
#   src/models/evaluate.py  →  evaluate_model()
#
# These tests make sure your evaluation function correctly
# computes all the metrics you care about for churn prediction.
#
# Fixtures used (defined in conftest.py):
#   sample_X:         fake feature data, no Churn column
#   sample_y:         fake binary churn labels
#   trained_pipeline:  fitted sklearn Pipeline (preprocessor + LogisticRegression)

import pytest
import pandas as pd
import numpy as np


# -------------------------------------------------------
# TESTS FOR evaluate_model()
# -------------------------------------------------------

class TestEvaluateModel:

    def test_returns_a_dictionary(self, trained_pipeline, sample_X, sample_y):
        """
        evaluate_model() must return a dictionary.
        All the metrics need to be accessible by key name
        like metrics['roc_auc'], metrics['accuracy'], etc.
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        assert isinstance(result, dict)

    def test_all_expected_metric_keys_are_present(self, trained_pipeline, sample_X, sample_y):
        """
        The dictionary must contain all 6 keys your pipeline depends on.
        If any key is missing, code downstream that reads metrics['recall']
        for example would crash with a KeyError.
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        expected_keys = {"roc_auc", "accuracy", "precision", "recall", "f1", "confusion_matrix"}
        assert expected_keys.issubset(result.keys())

    def test_roc_auc_is_between_0_and_1(self, trained_pipeline, sample_X, sample_y):
        """
        ROC-AUC is always a value between 0 and 1 by definition.
        0.5 = random guessing, 1.0 = perfect model.
        Anything outside this range means something is badly wrong.
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_accuracy_is_between_0_and_1(self, trained_pipeline, sample_X, sample_y):
        """
        Accuracy is the fraction of correct predictions.
        It must always be between 0 (all wrong) and 1 (all correct).
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_is_2x2(self, trained_pipeline, sample_X, sample_y):
        """
        For binary classification (churned vs not churned),
        the confusion matrix must always be a 2x2 grid:
        [[TN, FP],
         [FN, TP]]
        A different shape means something went wrong with predictions.
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        assert result["confusion_matrix"].shape == (2, 2)

    def test_confusion_matrix_counts_sum_to_total_samples(self, trained_pipeline, sample_X, sample_y):
        """
        Every sample must appear exactly once in the confusion matrix.
        TN + FP + FN + TP should equal the total number of rows.
        If this fails, some predictions are being lost or double counted.
        """
        from src.models.evaluate import evaluate_model

        result = evaluate_model(trained_pipeline, sample_X, sample_y)

        assert result["confusion_matrix"].sum() == len(sample_y)