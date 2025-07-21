from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Add this
from predict_mood import predict_mood

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    mood = predict_mood(text)
    return jsonify({'mood': mood})

if __name__ == '__main__':
    app.run(port=5000)
