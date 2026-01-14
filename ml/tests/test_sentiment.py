"""
Sentiment Analysis Test Script
Tests the sentiment prediction model on various test cases and generates a report.
"""

import sys
import os
import re
import joblib
import nltk
from datetime import datetime
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Set up paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / 'models'
MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
VECTORIZER_PATH = MODELS_DIR / 'vectorizer.pkl'

# Load model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Loaded vectorizer from: {VECTORIZER_PATH}")
except FileNotFoundError as e:
    print(f"Error: Model or vectorizer not found. {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
    sys.exit(1)

# Preprocessing function (matching the training notebook)
def clean_text(text):
    """Preprocess text using the same function as in the training notebook."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    tokens = nltk.word_tokenize(text)
    
    # Refined Stopwords - keep negation words
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', "n't", 'cannot', 
                     "don't", "doesn't", "didn't", "won't", "shouldn't", "couldn't", 
                     "wasn't", "weren't", "isn't", "aren't"}
    final_stopwords = stop_words - negation_words
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word in negation_words or word not in final_stopwords]
    
    return ' '.join(tokens)

def predict_sentiment(text):
    """Predict sentiment using the loaded model and vectorizer."""
    # Preprocess the text
    processed_text = clean_text(text)
    
    # Handle empty text after preprocessing
    if not processed_text or processed_text.strip() == '':
        return "ERROR: Empty text after preprocessing"
    
    # Transform the text using the loaded TF-IDF vectorizer
    text_vector = vectorizer.transform([processed_text])
    
    # Predict using the loaded model
    prediction = model.predict(text_vector)
    
    # The model returns an array (e.g., ['positive']), so we return the first element
    return prediction[0]

def test_sentiment(text, expected, category=""):
    """Test a single text and return the result."""
    try:
        prediction = predict_sentiment(text)
        is_correct = prediction.lower() == expected.lower() if not prediction.startswith("ERROR") else False
    except Exception as e:
        prediction = f"ERROR: {str(e)}"
        is_correct = False
    
    return {
        'text': text,
        'expected': expected,
        'predicted': prediction,
        'correct': is_correct,
        'category': category
    }

def generate_report(results):
    """Generate a formatted report and save it to report.txt"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SENTIMENT ANALYSIS TEST REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Group results by category
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    total_tests = len(results)
    total_correct = sum(1 for r in results if r['correct'])
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Tests: {total_tests}")
    report_lines.append(f"Correct Predictions: {total_correct}")
    report_lines.append(f"Accuracy: {accuracy:.2f}%")
    report_lines.append("")
    
    # Detailed results by category
    for category, cat_results in categories.items():
        cat_correct = sum(1 for r in cat_results if r['correct'])
        cat_accuracy = (cat_correct / len(cat_results) * 100) if cat_results else 0
        
        report_lines.append("=" * 80)
        report_lines.append(f"CATEGORY: {category.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Tests: {len(cat_results)} | Correct: {cat_correct} | Accuracy: {cat_accuracy:.2f}%")
        report_lines.append("")
        
        for i, result in enumerate(cat_results, 1):
            status = "[PASS]" if result['correct'] else "[FAIL]"
            report_lines.append(f"Test {i}: {status}")
            report_lines.append(f"  Text: {result['text']}")
            report_lines.append(f"  Expected: {result['expected']}")
            report_lines.append(f"  Predicted: {result['predicted']}")
            if not result['correct']:
                report_lines.append(f"  [MISMATCH] Expected '{result['expected']}' but got '{result['predicted']}'")
            report_lines.append("")
    
    # Overall statistics
    report_lines.append("=" * 80)
    report_lines.append("DETAILED STATISTICS BY CATEGORY")
    report_lines.append("=" * 80)
    for category, cat_results in categories.items():
        cat_correct = sum(1 for r in cat_results if r['correct'])
        cat_total = len(cat_results)
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        report_lines.append(f"{category:30s}: {cat_correct}/{cat_total} ({cat_accuracy:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write to file
    report_path = SCRIPT_DIR / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")
    
    return report_path

def main():
    """Run all test cases."""
    print("Running sentiment analysis tests...")
    print("=" * 80)
    
    results = []
    
    # 1. The "Easy Wins" (Baseline Test)
    print("\n1. Testing 'Easy Wins' (Baseline Test)...")
    results.append(test_sentiment(
        "Absolutely loved this! The quality is top-notch and shipping was incredibly fast. Highly recommended.",
        "positive",
        "Easy Wins - Baseline"
    ))
    results.append(test_sentiment(
        "Five stars. Best investment I've made all year. Works exactly as described.",
        "positive",
        "Easy Wins - Baseline"
    ))
    results.append(test_sentiment(
        "Terrible experience. The item arrived broken and customer service was completely useless.",
        "negative",
        "Easy Wins - Baseline"
    ))
    results.append(test_sentiment(
        "Waste of money. Do not buy this. It stopped working after two days.",
        "negative",
        "Easy Wins - Baseline"
    ))
    
    # 2. The "Negation" Trap (Logic Test)
    print("\n2. Testing 'Negation' Trap (Logic Test)...")
    results.append(test_sentiment(
        "The plot was not good at all, despite the hype.",
        "negative",
        "Negation Trap"
    ))
    results.append(test_sentiment(
        "I had no issues with the installation process.",
        "positive",
        "Negation Trap"
    ))
    
    # 3. Mixed Sentiment (The 77% Zone)
    print("\n3. Testing 'Mixed Sentiment' (The 77% Zone)...")
    results.append(test_sentiment(
        "The screen is amazing and the battery life is great, but the software is buggy and crashes constantly.",
        "negative",  # Likely negative due to "buggy" and "crashes"
        "Mixed Sentiment"
    ))
    results.append(test_sentiment(
        "It's a bit expensive for what you get, but the build quality makes up for it eventually.",
        "positive",  # Leaning positive
        "Mixed Sentiment"
    ))
    results.append(test_sentiment(
        "I really wanted to like this because the design is beautiful, but it is just too uncomfortable to use for long periods.",
        "negative",  # Leaning negative
        "Mixed Sentiment"
    ))
    
    # 4. Sarcasm (The Stress Test)
    print("\n4. Testing 'Sarcasm' (The Stress Test)...")
    results.append(test_sentiment(
        "Great, another update that breaks everything. Just what I needed today.",
        "negative",  # Sarcastic
        "Sarcasm"
    ))
    results.append(test_sentiment(
        "Oh brilliant. The battery lasts a whole 20 minutes. Groundbreaking technology.",
        "negative",  # Sarcastic
        "Sarcasm"
    ))
    
    # 5. Short & Context-Dependent
    print("\n5. Testing 'Short & Context-Dependent'...")
    results.append(test_sentiment(
        "Meh.",
        "negative",
        "Short & Context-Dependent"
    ))
    results.append(test_sentiment(
        "Solid.",
        "positive",
        "Short & Context-Dependent"
    ))
    results.append(test_sentiment(
        "Refunded.",
        "negative",
        "Short & Context-Dependent"
    ))
    results.append(test_sentiment(
        "Steal.",  # Context: "It was a steal"
        "positive",
        "Short & Context-Dependent"
    ))
    
    # Generate report
    print("\n" + "=" * 80)
    print("Generating report...")
    report_path = generate_report(results)
    
    print(f"\n[SUCCESS] Testing complete! Report saved to: {report_path}")

if __name__ == '__main__':
    main()

