import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Map diseases → possible file names
DATASETS = {
    "diabetes": ["diabetes.csv"],
    "heart":    ["heart.csv"],
    "kidney":   ["kidney_disease.csv", "kidney.csv"],
    "liver":    ["Indian Liver Patient Dataset (ILPD).csv", "liver.csv"]
}

models = {}
scalers = {}

def locate_file(candidates):
    """Return the first existing path from candidates, or raise."""
    for fn in candidates:
        if os.path.exists(fn):
            return fn
    raise FileNotFoundError(f"None of these files found: {candidates}")

def preprocess_data(name, df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if name == "kidney":
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        if 'classification' in df.columns:
            df['classification'] = df['classification'].map({'ckd':1,'notckd':0})
        df = df.select_dtypes(include=[np.number])

    elif name == "liver":
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].replace({1:1,2:0})

    return df

def get_target_column(name, df):
    # priority list for each disease
    cands = {
        "diabetes":[df.columns[-1]],
        "heart":   [df.columns[-1]],
        "kidney":  ["classification","class","target"],
        "liver":   ["Dataset","class","target"]
    }
    for col in cands[name]:
        if col in df.columns:
            return col
    return df.columns[-1]

def train_one(name):
    # 2. locate file
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)

    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    print(f"✅ {name.capitalize()} model accuracy: {acc:.2f}")

    models[name] = m
    scalers[name] = scaler

def load_and_train_all():
    for name in DATASETS:
        train_one(name)

def predict_disease(name, inputs):
    if name not in models:
        raise ValueError(f"Model for '{name}' not loaded")
    m = models[name]
    s = scalers[name]
    return m.predict(s.transform([inputs]))[0]

def get_feature_info(name):
    path = locate_file(DATASETS[name])
    df = pd.read_csv(path)
    df = preprocess_data(name, df)
    target = get_target_column(name, df)
    X = df.drop(columns=[target])
    # return list of (feature, min, max)
    return [(col, float(X[col].min()), float(X[col].max())) for col in X.columns]
