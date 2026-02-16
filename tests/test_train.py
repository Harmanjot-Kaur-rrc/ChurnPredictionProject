import pandas as pd
from src.models.train import train_model
from src.features.make_features import build_preprocessor

def test_train_model_runs():
    X = pd.DataFrame({"Age":[20,30], "Gender":["M","F"]})
    y = pd.Series([0,1])

    pre = build_preprocessor(["Age"], ["Gender"])
    model = train_model(X, y, pre)

    assert model is not None
