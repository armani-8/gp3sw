import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier

def explore_data():
    df = pd.read_csv("troop_movements.csv")
    
    # Counts of empire vs resistance
    print(df["empire_or_resistance"].value_counts())
    
    # Counts of characters by homeworld
    print(df["home_world"].value_counts())
    
    # Counts of characters by unit_type
    print(df["unit_type"].value_counts())
    
    # Engineer a new feature called is_resistance
    df["is_resistance"] = df["empire_or_resistance"] == "resistance"
    
    # Create a bar plot showing Empire vs Resistance distribution
    sns.countplot(x="empire_or_resistance", data=df)
    plt.show()

def build_model():
    df = pd.read_csv("troop_movements.csv")
    
    # Engineer a new feature called is_resistance
    df["is_resistance"] = df["empire_or_resistance"] == "resistance"
    
    # Create features and target variables
    X = df[["home_world", "unit_type"]]
    y = df["is_resistance"]
    
    # Create the decision tree classifier model
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    # Create a bar plot showing feature importance
    importance = model.feature_importances_
    sns.barplot(x=importance, y=X.columns)
    plt.show()
    
    # Save the trained model as a pickle file
    with open("trained_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Model trained and saved as trained_model.pkl")

explore_data()
build_model()
