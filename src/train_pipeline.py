import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------


def main():
    print("\n Pipeline started...\n")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    df = pd.read_csv("data/processed/cleaned_churn_data.csv")

    # ---------------------------------------------------
    # EDA VISUALIZATIONS
    # ---------------------------------------------------

    print(" Generating EDA visualizations...")

    # Churn Distribution
    plt.figure(figsize=(6, 5))
    palette = ["#2ECC71", "#E74C3C"]
    ax = sns.countplot(x="Churn", hue="Churn", data=df, palette=palette)

    total = len(df)
    for p in ax.patches:
        percentage = 100 * p.get_height() / total
        ax.annotate(
            f"{percentage:.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )

    plt.title("Churn Distribution")
    plt.savefig("models/churn_distribution.png")
    plt.close()

    # Numerical Histograms
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[numerical_cols].hist(
        bins=15, figsize=(15, 10), color="#3498DB", edgecolor="white"
    )

    plt.tight_layout()
    plt.savefig("models/numerical_distributions.png")
    plt.close()

    # Categorical Distributions
    categorical_cols = df.select_dtypes(include=["object"]).columns

    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 4 * n_rows))

    for i, col in enumerate(categorical_cols, 1):
        ax = plt.subplot(n_rows, n_cols, i)
        order = df[col].value_counts().index
        total = len(df)

        sns.countplot(x=col, data=df, hue=col, order=order, palette="viridis", ax=ax)

        for p in ax.patches:
            percent = 100 * p.get_height() / total
            ax.annotate(
                f"{percent:.1f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(f"{col} Distribution")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("models/categorical_distributions.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap="RdBu_r", square=True)
    plt.title("Correlation Heatmap")
    plt.savefig("models/correlation_heatmap.png")
    plt.close()

    print(" EDA visualizations saved.\n")

    # ---------------------------------------------------
    # MODELING
    # ---------------------------------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Logistic": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }

    results = []

    print("Training models...\n")

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        results.append(
            {
                "Model": name,
                "ROC-AUC": roc_auc_score(y_val, y_proba),
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred),
                "Recall": recall_score(y_val, y_pred),
                "F1": f1_score(y_val, y_pred),
            }
        )

        models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

    print("Validation Results")
    print("-" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    print(f"\n Best Model: {best_model_name}")

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    y_pred = best_model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Best Model")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    print("\n Classification Report")
    print("-" * 50)
    print(classification_report(y_val, y_pred))

    # ---------------------------------------------------
    # ROC CURVE
    # ---------------------------------------------------

    y_proba = best_model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)

    plt.figure()
    plt.plot(
        fpr, tpr, label=f"{best_model_name} (AUC={roc_auc_score(y_val, y_proba):.3f})"
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("models/roc_curve.png")
    plt.close()

    # ---------------------------------------------------
    # FEATURE IMPORTANCE (XGBoost only)
    # ---------------------------------------------------

    if best_model_name == "XGBoost":
        model = best_model.named_steps["model"]
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

        importance = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(8, 6))
        plt.barh(fi_df["Feature"], fi_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 15 Feature Importances")
        plt.savefig("models/feature_importance.png")
        plt.close()

        print("Feature importance saved.")

        # ---------------------------------------------------
        # SHAP
        # ---------------------------------------------------

        print("Generating SHAP summary plot...")

        X_val_transformed = best_model.named_steps["preprocessor"].transform(X_val)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_transformed)

        shap.summary_plot(
            shap_values, X_val_transformed, feature_names=feature_names, show=False
        )

        plt.savefig("models/shap_summary.png", bbox_inches="tight")
        plt.close()

        print("SHAP summary saved.")

    print("\n Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
