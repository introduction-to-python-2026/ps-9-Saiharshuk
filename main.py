import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# ודאי שהקובץ בתיקייה נקרא בדיוק כך
df = pd.read_csv("parkinsons.csv")

# 2. Select features
# בחרנו שני מאפיינים פופולריים מהמאמר
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]
X = df[features]
y = df["status"]

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Choose a model
# KNN עובד מצוין על הנתונים האלו ומגיע לאחוזים גבוהים
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 7. Save the model
joblib.dump(model, "my_model.joblib")
