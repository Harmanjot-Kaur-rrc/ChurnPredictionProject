import requests

URL = "http://127.0.0.1:8000/predict"

payload = {
    "Age": 42,
    "Gender": "Female",
    "Tenure": 18,
    "Usage_Frequency": 12,
    "Support_Calls": 2,
    "Payment_Delay": 1,
    "Subscription_Type": "Premium",
    "Contract_Length": "Annual",
    "Total_Spend": 2450,
    "Last_Interaction": 7
}

response = requests.post(URL, json=payload)

print("Status code:", response.status_code)
print("Response:", response.json())
