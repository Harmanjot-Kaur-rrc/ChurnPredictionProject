from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train, preprocessor):

    models = {}

    def grid(pipe, params):
        g = GridSearchCV(pipe, params, scoring="roc_auc", cv=3, n_jobs=-1)
        g.fit(X_train, y_train)
        return g

    models["Logistic"] = grid(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        {"model__C": [10]},
    )

    models["RandomForest"] = grid(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(random_state=42)),
            ]
        ),
        {"model__n_estimators": [200]},
    )

    models["GradientBoosting"] = grid(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        {"model__n_estimators": [150]},
    )

    models["XGBoost"] = grid(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=42,
                    ),
                ),
            ]
        ),
        {"model__n_estimators": [150]},
    )

    models["MLP"] = grid(
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", MLPClassifier(max_iter=100, random_state=42)),
            ]
        ),
        {"model__hidden_layer_sizes": [(64,)]},
    )

    return models
