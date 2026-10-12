import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_sensor_model():
    print("Loading dataset...")
    data_path = '../01_datasets/dummy_sensors.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run create_dummy_sensors.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    X = df[['temperature', 'humidity', 'gas', 'mq135_raw', 'mq2_digital', 'alarm']]
    y = df['risk_score']
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Evaluate score on training data for sanity check
    score = model.score(X, y)
    print(f"Model R^2 Score on training data: {score:.4f}")
    
    os.makedirs('../02_models', exist_ok=True)
    model_path = '../02_models/sensor_model.pkl'
    joblib.dump(model, model_path)
    print(f"Trained sensor model saved to '{model_path}'.")

if __name__ == '__main__':
    train_sensor_model()
