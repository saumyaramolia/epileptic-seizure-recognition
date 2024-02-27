from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import torch

# Load the trained model
model_filename = 'my_dnn_model.pkl'
with open(model_filename, 'rb') as f:
    loaded_model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Add CORS support to the app


# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        # Convert JSON data to numpy array
        data = torch.tensor(json_data['data'], dtype=torch.float32)

        # Make predictions using the loaded model
        with torch.no_grad():
            outputs = loaded_model(data)
            _, predicted = torch.max(outputs, 1)
            predictions = predicted.numpy().tolist()

        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
