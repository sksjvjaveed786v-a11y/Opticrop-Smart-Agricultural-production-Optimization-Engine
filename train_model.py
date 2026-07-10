import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Features (input)
X = data.drop("label", axis=1)

# Target (output)
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")