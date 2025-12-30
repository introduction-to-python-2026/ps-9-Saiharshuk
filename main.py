import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
display(df.head())
x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
import pandas as pd
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid') # Sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
import joblib

joblib.dump(model, 'my_model.joblib')
