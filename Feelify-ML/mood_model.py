# mood_model.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
import joblib
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
data = load_files('mood_data', encoding='utf-8', decode_error='replace')
X, y = data.data, data.target

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, 'mood_model.pkl')

print("✅ Model trained and saved as mood_model.pkl")
