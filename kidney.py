import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data = {
    'age': [45, 50, 60, 30, 70],
    'blood_pressure': [80, 85, 90, 75, 95],
    'specific_gravity': [1.010, 1.020, 1.015, 1.010, 1.025],
    'albumin': [1, 2, 3, 1, 2],
    'sugar': [0, 0, 1, 0, 1],
    'red_blood_cells': [0, 0, 1, 0, 1],
    'class': [1, 0, 1, 0, 1]  # 1 = ckd, 0 = notckd
}
df = pd.DataFrame(data)
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1)) 
