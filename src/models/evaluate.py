from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
 
 
def evaluate_model(model, X, y):
 
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
 
    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }