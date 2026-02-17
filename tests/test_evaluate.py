from src.models.evaluate import evaluate_model

def test_evaluate_output():
    scores = evaluate_model([0,1], [0,1], [0.1,0.9])
    assert "accuracy" in scores
