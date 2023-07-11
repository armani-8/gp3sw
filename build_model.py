import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def explore_data():
    df = pd.read_csv("troop_movements.csv")

    #print statements
    print(df["empire_or_resistance"].value_counts())
    print(df["homeworld"].value_counts())
    print(df["unit_type"].value_counts())
    df["is_resistance"] = df["empire_or_resistance"] == "resistance"

    sns.countplot(x="empire_or_resistance", data=df)
    plt.show()

def build_model():
    df = pd.read_csv("troop_movements.csv")
    
    #Extract 'id' from 'home_world'
    df['homeworld'] = df['homeworld'].apply(lambda x: eval(x)['id'])

    df['is_resistance'] = df['empire_or_resistance'] == 'resistance'

    #Convert 'unit_type' to numerical format using label encoding
    le = LabelEncoder()
    df['unit_type'] = le.fit_transform(df['unit_type'])
    # df["is_resistance"] = df["empire_or_resistance"] == "resistance"
    
    X = df[["homeworld", "unit_type"]]
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
