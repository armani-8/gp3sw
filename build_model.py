import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier

def explore_data():
    df = pd.read_csv("troop_movements.csv")

    #print statements
    print(df["empire_or_resistance"].value_counts())
    print(df["home_world"].value_counts())
    print(df["unit_type"].value_counts())
    df["is_resistance"] = df["empire_or_resistance"] == "resistance"

    sns.countplot(x="empire_or_resistance", data=df)
    plt.show()

def build_model():
    df = pd.read_csv("troop_movements.csv")
    
    df["is_resistance"] = df["empire_or_resistance"] == "resistance"
    
    X = df[["home_world", "unit_type"]]
    y = df["is_resistance"]
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    importance = model.feature_importances_
    sns.barplot(x=importance, y=X.columns)
    plt.show()

    with open("trained_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Model trained and saved as trained_model.pkl")

explore_data()
build_model()
