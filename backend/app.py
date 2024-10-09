import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the machine learning model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get('news')

    # Prediction (Assume you have a vectorizer inside the pickle file)
    prediction = model.predict([news_text])[0]

    # Return result as JSON
    return jsonify({'prediction': 'fake' if prediction == 1 else 'real'})

if __name__ == '__main__':
    app.run(debug=True)
