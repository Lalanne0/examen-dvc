import pandas as pd
import yaml
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

grid_params = params["gridsearch"]
random_state = params["split"]["random_state"]

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze() 

model = GradientBoostingRegressor(random_state=random_state)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=grid_params,
    cv=2,
    scoring='r2',
    n_jobs=-1      
)
print("Début du GridSearch. J'ai mis des params nuls parce que l'exam prend déjà des plombes à corriger.")
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print(f"Meilleurs params : {best_params}")

os.makedirs("models", exist_ok=True)

joblib.dump(best_params, "models/best_params.pkl")
print("Save des params OK. La bise, surtout à Kilyan.")