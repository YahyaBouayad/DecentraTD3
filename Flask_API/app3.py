from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)

predictions_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f"Logistic Regression Model Accuracy: {accuracy_rf}")

# Save the model
joblib.dump(model_rf, 'model_rf.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_rf():
    data = request.get_json()
    features = [data['features']]
    model_rf = joblib.load("model_rf.pkl")
    predictions = model_rf.predict(features)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host="localhost", port=5002, debug=True)
