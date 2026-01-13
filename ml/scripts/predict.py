# server/predict.py

import sys
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Define correct paths to the model files ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from 'ml' to the root, then into 'ml/models'
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml', 'models', 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'ml', 'models', 'vectorizer.pkl')

try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords', quiet=True)

# --- Global variables for efficiency ---
# Load the model and vectorizer once when the script starts
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    # If files are not found, print an error to stderr and exit
    print(f"Error: Model or vectorizer not found. Searched at {MODEL_PATH} and {VECTORIZER_PATH}", file=sys.stderr)
    sys.exit(1)

# Pre-compile regex and load stop words for speed
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
# Regex to remove noise: URLs, @mentions, and non-alphanumeric characters
noise_regex = re.compile(r'(@[A-Za-z0-9]+)|(https?://[A-Za-z0-9./]+ )|([^A-Za-z\s])')

def preprocess_text(text):
    """
    The same preprocessing function used during model training.
    This is crucial for getting accurate predictions.
    """
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove noise
    text = noise_regex.sub(' ', text)
    # 3. Tokenize (split into words)
    words = text.split()
    # 4. Remove stop words and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(processed_words)

def predict_sentiment(text):
    """
    Takes raw text, preprocesses it, and returns the model's prediction.
    """
    # Preprocess the new text
    processed_text = preprocess_text(text)
    
    # Transform the text using the loaded TF-IDF vectorizer
    # The vectorizer expects a list of documents, so we pass [processed_text]
    text_vector = vectorizer.transform([processed_text])
    
    # Predict using the loaded model
    prediction = model.predict(text_vector)
    
    # The model returns an array (e.g., ['positive']), so we return the first element
    return prediction[0]

# --- Main execution block ---
if __name__ == '__main__':
    # The script expects exactly one command-line argument (the text)
    if len(sys.argv) > 1:
        # sys.argv[0] is the script name, sys.argv[1] is the first argument
        input_text = sys.argv[1]
        
        # Make the prediction
        sentiment = predict_sentiment(input_text)
        
        # Print the result to standard output.
        # This is how Node.js will get the result.
        print(sentiment)
    else:
        # If no text is provided, print an error to stderr
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

