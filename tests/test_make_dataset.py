from src.data.make_dataset import load_data

def test_load_data_shape():
    df = load_data()
    assert df.shape[0] > 0  # it loaded rows
    assert 'Churn' in df.columns  # target exists
