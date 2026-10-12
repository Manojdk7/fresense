import pandas as pd
import numpy as np
import os

def generate_sensor_data():
    print("Generating dummy sensor data...")
    np.random.seed(42)
    n_samples = 2000
    
    # Simulate data ranges
    # Temperature: 0 to 45 C
    temperature = np.random.uniform(0, 45, n_samples)
    # Humidity: 30 to 100 %
    humidity = np.random.uniform(30, 100, n_samples)
    # Gas (e.g., TVOC / Ammonia in ppm): 0 to 500
    gas = np.random.uniform(0, 500, n_samples)
    # Hardware-aligned features from the ESP32 payload.
    mq135_raw = np.clip((gas / 500.0 * 4095.0) + np.random.normal(0, 120, n_samples), 0, 4095).astype(int)
    mq2_digital = (np.random.random(n_samples) < (gas / 700.0)).astype(int)
    alarm = ((mq135_raw >= 500) | (mq2_digital == 0)).astype(int)
    
    # Calculate synthetic risk based on simple environmental logic
    # Higher temp, humidity, and gas -> higher spoilage risk
    temp_risk = temperature / 45.0
    hum_risk = humidity / 100.0
    gas_risk = gas / 500.0
    mq135_risk = mq135_raw / 4095.0
    mq2_risk = 1.0 - mq2_digital
    alarm_risk = alarm.astype(float)
    
    # Weighted combination
    risk = (temp_risk * 0.3) + (hum_risk * 0.2) + (gas_risk * 0.2)
    risk += (mq135_risk * 0.15) + (mq2_risk * 0.05) + (alarm_risk * 0.1)
    
    # Add random noise for realism
    risk += np.random.normal(0, 0.05, n_samples)
    risk = np.clip(risk, 0, 1) # Ensure risk stays between 0 and 1
    
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'gas': gas,
        'mq135_raw': mq135_raw,
        'mq2_digital': mq2_digital,
        'alarm': alarm,
        'risk_score': risk
    })
    
    os.makedirs('../01_datasets', exist_ok=True)
    output_path = '../01_datasets/dummy_sensors.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} rows of sensor data at '{output_path}'.")

if __name__ == '__main__':
    generate_sensor_data()
