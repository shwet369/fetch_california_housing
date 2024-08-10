

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load the regression model
regmodel = pickle.load(open('E:/fetch_california_housing/regmodel.pkl', 'rb'))

# If you have a scaler, make sure to load or define it here
# scaler = pickle.load(open('path_to_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if request.is_json:
        data = request.json.get('data', {})
        
        # Ensure data is a dictionary and has the correct number of features
        if not isinstance(data, dict) or len(data) != 8:
            return jsonify({"error": "Invalid data format"}), 400
        
        try:
            data_values = np.array(list(data.values())).reshape(1, -1)
            
            # If you have a scaler, transform the data
            # new_data = scaler.transform(data_values)
            # For now, assuming no scaling is needed
            new_data = data_values
            
            # Predict using the regression model
            output = regmodel.predict(new_data)
            
            # Return the prediction as JSON
            return jsonify({"prediction": output[0].tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == "__main__":
    app.run(debug=True)

print("Received data:", data)
print("Data values shape:", data_values.shape)

