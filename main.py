
import pandas as pd
from src.data_preprocessing import preprocess_sensor_data
from src.model_training import train_random_forest, train_xgboost
from src.evaluation import evaluate_model

# Load sensor data
df = pd.read_csv("data/sensor_data.csv", parse_dates=['timestamp'])

# Preprocess
X_train, X_test, y_train, y_test = preprocess_sensor_data(df)

# Train models
rf_model = train_random_forest(X_train, y_train)
xgb_model = train_xgboost(X_train, y_train)

# Evaluate
print("\nRandom Forest Performance:")
evaluate_model(rf_model, X_test, y_test)

print("\nXGBoost Performance:")
evaluate_model(xgb_model, X_test, y_test)
