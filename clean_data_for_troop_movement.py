import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import json

def clean_data():
    df = pd.read_csv("troop_movements10m.csv")

    df["unit_type"] = df["unit_type"].replace("invalid_unit", "unknown")
    df["location_x"].ffill(inplace=True)
    df["location_y"].ffill(inplace=True)

    with open("home_worlds.json", "r") as file:
        home_worlds = json.load(file)
        home_worlds_mapping = {entry["id"]: entry["name"] for entry in home_worlds}

    df["homeworld"] = df["homeworld"].map(home_worlds_mapping)

    le_unit_type = LabelEncoder()
    df['unit_type'] = le_unit_type.fit_transform(df['unit_type'].astype(str))

    df.to_parquet("troop_movements10m.parquet", index=False)
    print("Data cleaned and saved as troop_movements10m.parquet")

def predict_data():
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    df = pd.read_parquet("troop_movements10m.parquet")

    df["is_resistance"] = model.predict(df[["homeworld", "unit_type"]])

    print(df.head())

clean_data()
predict_data()
