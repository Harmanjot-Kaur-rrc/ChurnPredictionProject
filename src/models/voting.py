from sklearn.ensemble import VotingClassifier


def build_voting(trained_models):

    estimators = [
        ("log", trained_models["Logistic"].best_estimator_),
        ("rf", trained_models["RandomForest"].best_estimator_),
        ("gb", trained_models["GradientBoosting"].best_estimator_),
        ("xgb", trained_models["XGBoost"].best_estimator_),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1, 1, 1, 2],
    )

