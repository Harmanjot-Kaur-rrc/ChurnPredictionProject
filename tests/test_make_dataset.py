import pandas as pd
from src.data.make_dataset import load_data

def test_load_data(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"A":[1], "Churn":[0]}).to_csv(csv, index=False)

    df = load_data(csv)
    assert "Churn" in df.columns

