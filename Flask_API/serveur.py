import requests
import json
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights equally as we start with no prior knowledge about model performance
weights = {
    'model_lr': 1/3,
    'model_svm': 1/3,
    'model_rf': 1/3
}

# List of model API endpoints exposed by ngrok
model_urls = [
    'http://localhost:5000/predict',
    'http://localhost:5001/predict',
    'http://localhost:5002/predict'
]

# Function to send batch requests and collect responses
def send_batch_requests(batch_data, model_urls):
    responses = []
    for url in model_urls:
        response = requests.post(url, json={'features': batch_data}).json()
        responses.append(response['predictions'])
    return responses

# Function to get consensus prediction
def get_consensus_prediction(X_test, weights):
    all_predictions = []
    for features in X_test:
        # Collect predictions from all models
        model_predictions = send_batch_requests(features.tolist(), model_urls)
        # Weighted average of predictions
        weighted_predictions = np.average(model_predictions, axis=0, weights=list(weights.values()))
        # Consensus prediction is the one with the highest weighted average score
        consensus_prediction = np.argmax(weighted_predictions)
        all_predictions.append(consensus_prediction)
    return all_predictions

# Calculate consensus predictions
consensus_predictions = get_consensus_prediction(X_test, weights)
# Evaluate the consensus model
consensus_accuracy = accuracy_score(y_test, consensus_predictions)
print(f'Consensus Model Accuracy: {consensus_accuracy}')
