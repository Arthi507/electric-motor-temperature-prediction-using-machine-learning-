import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("measures_v2.csv")

df = df.dropna()

# Features & Target
X = df.drop("pm", axis=1)
y = df["pm"]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully")