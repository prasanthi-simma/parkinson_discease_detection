from flask import request
from flask import Flask, jsonify
import joblib

flask = Flask(__name__)

@flask.route('/')
def home():
    return "Welcome to Parkinson's Disease Detection API"
    print("Welcome to Parkinson's Disease Detection API")   

@flask.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Assuming the input data is in the correct format for prediction
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    model = joblib.load('parkinson_model.pkl')
    flask.run(debug=True)   
    
