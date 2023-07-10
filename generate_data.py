import random
import json
import pandas as pd

unit_types = [
    "stormtrooper", "tie_fighter", "at-st", "x-wing",
    "resistance_soldier", "at-at", "tie_silencer", "unknown"
]

def generate_data():
    data = []
    home_worlds = load_home_worlds()
    
    for unit_id in range(1, 1001):
        record = {
            "timestamp": pd.Timestamp.now(),
            "unit_id": unit_id,
            "unit_type": random.choice(unit_types),
            "empire_or_resistance": random.choice(["empire", "resistance"]),
            "location_x": random.randint(0, 100),
            "location_y": random.randint(0, 100),
            "destination_x": random.randint(0, 100),
            "destination_y": random.randint(0, 100),
            "home_world": random.choice(home_worlds)
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv("troop_movements.csv", index=False)
    print("Data generated and saved to troop_movements.csv")

def load_home_worlds():
    with open("home_worlds.json", "r") as file:
        home_worlds = json.load(file)
    return home_worlds

generate_data()
