import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def clean_data():
    df = pd.read_csv("troop_movements10m.csv")

    df["unit_type"] = df["unit_type"].replace("invalid_unit", "unknown")
    df["location_x"].ffill(inplace=True)
    df["location_y"].ffill(inplace=True)

    # Assuming that homeworld and unit_type are categorical features
    le_homeworld = LabelEncoder()
    le_unit_type = LabelEncoder()

    df['homeworld'] = le_homeworld.fit_transform(df['homeworld'].astype(str))
    df['unit_type'] = le_unit_type.fit_transform(df['unit_type'].astype(str))

    df.to_parquet("troop_movements10m.parquet", index=False)
    
    # Save the fitted LabelEncoders for later use in decoding
    with open('le_homeworld.pkl', 'wb') as f:
        pickle.dump(le_homeworld, f)
    with open('le_unit_type.pkl', 'wb') as f:
        pickle.dump(le_unit_type, f)

    print("Data cleaned and saved as troop_movements10m.parquet")

def predict_data():
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load the fitted LabelEncoders
    with open('le_homeworld.pkl', 'rb') as f:
        le_homeworld = pickle.load(f)
    with open('le_unit_type.pkl', 'rb') as f:
        le_unit_type = pickle.load(f)

    df = pd.read_parquet("troop_movements10m.parquet")

    print(df.columns)

    df["is_resistance"] = model.predict(df[["homeworld", "unit_type"]])

    # Decode the 'homeworld' identifiers back into names
    df['homeworld'] = le_homeworld.inverse_transform(df['homeworld'])

    print(df.head())

clean_data()
predict_data()
