import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()

model = joblib.load("models/gbr_model.pkl")

print("Initiating évaluation du modèle. Oui oui baguette.")
predictions = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

metrics = {
    "mse": mse,
    "r2": r2,
    "mae": mae
}

print(f"Métriques : {metrics}")

os.makedirs("metrics", exist_ok=True)

with open("metrics/scores.json", "w") as outfile:
    json.dump(metrics, outfile, indent=4)

df_preds = pd.DataFrame(predictions, columns=["silica_concentrate_predicted"])
df_preds.to_csv("data/prediction.csv", index=False)

print("Evaluation OK. J'ai envie d'un grec en fait.")