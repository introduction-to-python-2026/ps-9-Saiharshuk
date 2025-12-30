import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
# df.head() # Removed unnecessary head() call here

x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and test sets first
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(x_train) # Fit scaler only on training data
x_test_scaled = scaler.transform(x_test)     # Transform test data using the fitted scaler

# Train the K-Nearest Neighbors Classifier model
model = KNeighborsClassifier()
model.fit(x_train_scaled, y_train) # Train on scaled training data

# Make predictions on the scaled test data
y_pred = model.predict(x_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib
joblib.dump(model, 'my_model.joblib')
