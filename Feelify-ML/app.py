from flask import Flask, request, jsonify
from predict_mood import predict_mood  # ✅ FIXED: Use actual filename

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    mood = predict_mood(text)  # Returns: 'happy', 'sad', 'angry', etc.
    return jsonify({'mood': mood})

if __name__ == '__main__':
    app.run(port=5000)
