from src.features.make_features import build_preprocessor

def test_preprocessor_creation():
    pre = build_preprocessor(["Age"], ["Gender"])
    assert pre is not None
