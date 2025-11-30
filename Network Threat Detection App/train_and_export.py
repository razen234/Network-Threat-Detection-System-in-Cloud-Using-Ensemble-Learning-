"""
Train the 4 classifiers (RF, AdaBoost, GaussianNB, DecisionTree) on your prepared dataset
that already has the 15 selected features + the 'Label' column (binary 0/1).
Saves: models/rf.pkl, models/ada.pkl, models/gnb.pkl, models/dt.pkl and scaler.pkl (optional).
"""
import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(APP_DIR, "feature_list.json"), "r", encoding="utf-8") as f:
    FEATURE_LIST = json.load(f)["features"]

# === Load your dataset ===
# Expect a CSV with 16 columns: the 15 features + Label (0/1)
# Replace the path below with your dataset path
DATASET_CSV = os.path.join(APP_DIR, "CSE-CIC-IDS2018_15f.csv")
if not os.path.exists(DATASET_CSV):
    raise FileNotFoundError(f"Please create {DATASET_CSV} with 15 features + Label.")

df = pd.read_csv(DATASET_CSV)
X = df[FEATURE_LIST].copy()
y = df["Label"].astype(int).values

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 1) RandomForest
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODELS_DIR, "rf.pkl"))

# 2) AdaBoost (DecisionStump depth=2 as in your notebook)
base = DecisionTreeClassifier(max_depth=2, random_state=42)
ada = AdaBoostClassifier(estimator=base, n_estimators=300, learning_rate=0.5, random_state=42)
ada.fit(X_train, y_train)
joblib.dump(ada, os.path.join(MODELS_DIR, "ada.pkl"))

# 3) GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
joblib.dump(gnb, os.path.join(MODELS_DIR, "gnb.pkl"))

# 4) DecisionTree (plain)
dt = DecisionTreeClassifier(max_depth=None, random_state=42)
dt.fit(X_train, y_train)
joblib.dump(dt, os.path.join(MODELS_DIR, "dt.pkl"))

print("✅ Trained & saved rf.pkl, ada.pkl, gnb.pkl, dt.pkl, and scaler.pkl")
