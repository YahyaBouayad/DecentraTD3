from flask import Flask, request, jsonify
import joblib
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train, y_train)

# Evaluate the model
predictions_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, predictions_lr)
print(f"Logistic Regression Model Accuracy: {accuracy_lr}")

# Save the model
joblib.dump(model_lr, 'model_lr.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_lr():
    logging.info("Accessed the /predict_lr route")
    data = request.get_json()
    features = [data['features']]
    model_lr = joblib.load("model_lr.pkl")
    predictions = model_lr.predict(features)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
