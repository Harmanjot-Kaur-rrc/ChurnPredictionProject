import pandas as pd
import requests

URL = "http://127.0.0.1:8000/predict"

df = pd.read_csv("data/processed/cleaned_churn_data.csv")

results = []

for i, row in df.head(5).iterrows():
    payload = {
        "Age": int(row["Age"]),
        "Gender": row["Gender"],
        "Tenure": int(row["Tenure"]),
        "Usage_Frequency": int(row["Usage Frequency"]),
        "Support_Calls": int(row["Support Calls"]),
        "Payment_Delay": int(row["Payment Delay"]),
        "Subscription_Type": row["Subscription Type"],
        "Contract_Length": row["Contract Length"],
        "Total_Spend": float(row["Total Spend"]),
        "Last_Interaction": int(row["Last Interaction"]),
    }

    response = requests.post(URL, json=payload)

    print(f"\nRow {i}")
    print("Status Code:", response.status_code)
    print("Response:", response.json())

    results.append(response.json())

predictions = pd.DataFrame(results)
print("\nFinal Predictions DataFrame:")
print(predictions)

