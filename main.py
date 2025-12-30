# Initial data download and setup
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

# 1. Load the dataset
import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')

# 2. Select features
X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

# 3. Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# 5. Choose a model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train and test the accuracy
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0) # verbose=0 to suppress training output

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import accuracy_score
import numpy as np

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions

manual_accuracy = accuracy_score(y_test, y_pred)
print(f"Manual Accuracy Check: {manual_accuracy:.4f}")

# Update config.yaml
import yaml

selected_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
model_path = 'my_model.joblib'

config_data = {
    'selected_features': selected_features,
    'path': model_path
}

config_file_path = 'config.yaml'

with open(config_file_path, 'w') as file:
    yaml.dump(config_data, file, sort_keys=False)

# 7. Save the model
import joblib
joblib.dump(model, 'my_model.joblib')

