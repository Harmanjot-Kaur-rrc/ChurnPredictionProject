"""
scripts/batch_predict.py
─────────────────────────
Runs predictions on the first N rows of the processed dataset
and saves results to models/batch_predictions.csv.

Usage:
    python scripts/batch_predict.py --key analyst-key-456 --model rf --rows 10
"""
import argparse
import pandas as pd
from api_client import predict_churn, list_models

# ── CLI args ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch churn prediction")
parser.add_argument("--key",   required=True, help="Your API key")
parser.add_argument("--model", default="rf",  help="Model ID to use (default: rf)")
parser.add_argument("--rows",  type=int, default=5, help="Number of rows to predict (default: 5)")
args = parser.parse_args()

# ── Step 1: check which models you can access ──────────────
print("\nFetching available models...")
models_info = list_models(args.key)
print(f"Your allowed models: {models_info['your_allowed_models']}")

if args.model not in models_info["your_allowed_models"]:
    print(f"\nError: your key is not authorized to use '{args.model}'.")
    print(f"Choose from: {models_info['your_allowed_models']}")
    exit(1)

# ── Step 2: load data ──────────────────────────────────────
df = pd.read_csv("data/processed/cleaned_churn_data.csv")
sample = df.head(args.rows)

print(f"\nRunning predictions on {args.rows} rows using model '{args.model}'...\n")

# ── Step 3: predict each row ───────────────────────────────
results = []

for i, row in sample.iterrows():
    payload = {
        "model_id": args.model,
        "Age": int(row["Age"]),
        "Gender": row["Gender"],
        "Tenure": int(row["Tenure"]),
        "Usage Frequency": int(row["Usage Frequency"]),
        "Support Calls": int(row["Support Calls"]),
        "Payment Delay": int(row["Payment Delay"]),
        "Subscription Type": row["Subscription Type"],
        "Contract Length": row["Contract Length"],
        "Total Spend": float(row["Total Spend"]),
        "Last Interaction": int(row["Last Interaction"]),
    }

    result = predict_churn(payload, api_key=args.key)

    if result:
        print(f"Row {i:>4} | Churn: {result['churn_prediction']} | Probability: {result['churn_probability']:.4f}")
        results.append({
            "row": i,
            "churn_prediction": result["churn_prediction"],
            "churn_probability": result["churn_probability"],
            "model_id": result["model_id"],
        })
    else:
        print(f"Row {i:>4} | FAILED")

# ── Step 4: save results ───────────────────────────────────
if results:
    output_path = "models/batch_predictions.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} predictions to {output_path}")