from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Prediction: {y_pred}")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
