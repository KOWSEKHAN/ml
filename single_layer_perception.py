from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
data = {
    'feature1': [0, 1, 1, 0, 1, 0],
    'feature2': [0, 0, 1, 1, 1, 0],
    'class': [0, 1, 1, 0, 1, 0]
}
import pandas as pd
df = pd.DataFrame(data)
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42, solver='adam', learning_rate_init=0.001, warm_start=True)
for epoch in range(1, 6):
    mlp.max_iter = epoch
    mlp.fit(X_train, y_train)
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Epoch {epoch}/5")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {mlp.loss_:.4f}")
    print(f"===============================")
final_train_accuracy = accuracy_score(y_train, mlp.predict(X_train))
final_test_accuracy = accuracy_score(y_test, mlp.predict(X_test))
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
