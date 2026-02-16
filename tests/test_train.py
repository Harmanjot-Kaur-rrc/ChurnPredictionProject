import pandas as pd
from src.models.train import train_models  
from src.features.make_features import build_preprocessor

def test_train_model_runs():
    X = pd.DataFrame({"Age":[20,30], "Gender":["M","F"]})
    y = pd.Series([0,1])

    pre = build_preprocessor(["Age"], ["Gender"])
    model = train_models(X, y, pre)  # This returns a dictionary of models

    assert model is not None
    assert isinstance(model, dict)  # train_models returns a dictionary
    assert "Logistic" in model  # Check that at least one model is present