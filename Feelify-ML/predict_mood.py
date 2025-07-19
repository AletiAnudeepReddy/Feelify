import joblib
from sklearn.datasets import load_files

# Load mood dataset just to get target names
data = load_files('mood_data', encoding='utf-8', decode_error='replace')

# Load the trained model
model = joblib.load('mood_model.pkl')

# Get user input
text = input("Enter a sentence to detect mood: ")

# Predict mood index
predicted_index = model.predict([text])[0]

# Convert index to label
predicted_label = data.target_names[predicted_index]

# Show result
print(f"🧠 Detected mood: {predicted_label}")
