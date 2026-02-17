from sklearn.ensemble import VotingClassifier
<<<<<<< HEAD

=======
 
 
>>>>>>> c3ef9a4760b0bb7f096cf15f5ff9acc26640b276
def build_voting(trained_models):
 
    estimators = [
        ("log", trained_models["Logistic"].best_estimator_),
        ("rf", trained_models["RandomForest"].best_estimator_),
        ("gb", trained_models["GradientBoosting"].best_estimator_),
        ("xgb", trained_models["XGBoost"].best_estimator_)
    ]
 
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[1, 1, 1, 2]
    )
