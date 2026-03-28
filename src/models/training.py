import pandas as pd
import joblib
import yaml
from sklearn.ensemble import GradientBoostingRegressor
import os

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

best_params = joblib.load("models/best_params.pkl")

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
random_state = params["split"]["random_state"]

print(f"Entraînement du modèle avec les paramètres : {best_params}")

model = GradientBoostingRegressor(**best_params, random_state=random_state)

model.fit(X_train_scaled, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/gbr_model.pkl")
print("Model train and save OK. Mister Garden maintenant ?")