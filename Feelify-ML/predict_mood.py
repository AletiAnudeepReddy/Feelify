import joblib
from sklearn.datasets import load_files

# Load mood dataset to get target names
data = load_files('mood_data', encoding='utf-8', decode_error='replace')
target_names = data.target_names

# Load the trained model
model = joblib.load('mood_model.pkl')

def predict_mood(text):
    predicted_index = model.predict([text])[0]
    predicted_label = target_names[predicted_index]
    return predicted_label
