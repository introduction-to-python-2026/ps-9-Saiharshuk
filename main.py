import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()
x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_scaled, y)
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
import joblib

joblib.dump(model, 'my_model.joblib')
