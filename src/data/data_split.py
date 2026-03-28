import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]

data_path = "data/raw_data/raw.csv"
df = pd.read_csv(data_path)

X = df.drop("date", axis=1)
X = X.drop("silica_concentrate", axis=1)
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=random_state
)

os.makedirs("data/processed_data", exist_ok=True)

X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split OK. Coucou seb ou kilyan !")