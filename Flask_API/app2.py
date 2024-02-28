from flask import Flask, request, jsonify
import joblib
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model_svm = svm.SVC(probability=True)
model_svm.fit(X_train, y_train)

predictions_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f"Logistic Regression Model Accuracy: {accuracy_svm}")

# Save the model
joblib.dump(model_svm, 'model_svm.pkl')

app = Flask(__name__)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    data = request.get_json()
    features = [data['features']]
    model_svm = joblib.load("model_svm.pkl")
    predictions = model_svm.predict(features)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host="localhost", port=5001, debug=True)
