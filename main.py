import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv("parkinsons.csv")
df = df.dropna()

# 2. Select features and output
X = df[["MDVP:Fo(Hz)", "MDVP:Jitter(%)"]]
y = df["status"]

# 4. Split the data  (שורה אחת, בלי סוגריים מוזרים)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Choose and train model
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

# 6. Test accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# 7. Save model
joblib.dump(model, "my_model.joblib")
