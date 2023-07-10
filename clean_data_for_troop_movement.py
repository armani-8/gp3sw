import pandas as pd

def clean_data():
    df = pd.read_csv("troop_movements10m.csv")
    
    # Replace invalid_unit with unknown in unit_type
    df["unit_type"] = df["unit_type"].replace("invalid_unit", "unknown")
    
    # Fill missing location_x and location_y values using ffill
    df["location_x"].ffill(inplace=True)
    df["location_y"].ffill(inplace=True)
    
    # Save the cleaned data into a Parquet file
    df.to_parquet("troop_movements10m.parquet", index=False)
    print("Data cleaned and saved as troop_movements10m.parquet")

def predict_data():
    # Load the pickled model
    with open("trained_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Load the data from the Parquet file
    df = pd.read_parquet("troop_movements10m.parquet")
    
    # Run the data through the model and add predicted values
    df["is_resistance"] = model.predict(df[["home_world", "unit_type"]])
    
    print(df.head())

clean_data()
predict_data()
