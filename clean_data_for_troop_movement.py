import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def clean_data():
    df = pd.read_csv("troop_movements10m.csv")

    df["unit_type"] = df["unit_type"].replace("invalid_unit", "unknown")
    df["location_x"].ffill(inplace=True)
    df["location_y"].ffill(inplace=True)
    
    df.to_parquet("troop_movements10m.parquet", index=False)
    print("Data cleaned and saved as troop_movements10m.parquet")

def predict_data():
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    df = pd.read_parquet("troop_movements10m.parquet")

    print(df.columns)

    df["is_resistance"] = model.predict(df[["homeworld", "unit_type"]])
    
    print(df.head())

clean_data()
predict_data()
