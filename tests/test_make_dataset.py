import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import pytest
from src.data.make_dataset import load_data

def test_load_data(tmp_path):
    # Create a temporary CSV file
    csv = tmp_path / "data.csv"
    pd.DataFrame({"A": [1], "Churn": [0]}).to_csv(csv, index=False)
    
    # Test the load_data function
    df = load_data(csv)
    assert "Churn" in df.columns
    assert len(df) == 1

