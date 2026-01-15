"""
Sentiment Analysis Test Suite

This module provides comprehensive testing for the sentiment analysis model.
It tests the model's accuracy across various categories including:
- Baseline positive/negative cases
- Negation handling (e.g., "not good")
- Mixed sentiment texts
- Sarcasm detection
- Short and context-dependent phrases

The test suite generates detailed reports showing accuracy by category and
individual test results, helping identify model strengths and weaknesses.

Author: [Your Name]
Date: [Date]
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

# ============================================================================
# NLTK DATA SETUP
# ============================================================================
# Download NLTK data if needed (stopwords, tokenizers, etc.)
# This ensures the required NLTK resources are available for text processing
try:
    # Check if stopwords are already downloaded
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    # If not found, download required NLTK data
    nltk.download('stopwords', quiet=True)  # Common English stopwords
    nltk.download('punkt', quiet=True)      # Tokenizer data
    nltk.download('punkt_tab', quiet=True) # Additional tokenizer data

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Get the directory of the current script (ml/tests/)
SCRIPT_DIR = Path(__file__).parent
# Get the directory of the models (ml/models/)
# Go up one level from tests/ to ml/, then into models/
MODELS_DIR = SCRIPT_DIR.parent / 'models'
# Get the path to the sentiment model (trained classifier)
MODEL_PATH = MODELS_DIR / 'sentiment_model.pkl'
# Get the path to the vectorizer (TF-IDF vectorizer used during training)
VECTORIZER_PATH = MODELS_DIR / 'vectorizer.pkl'

# ============================================================================
# MODEL LOADING
# ============================================================================
# Load the pre-trained sentiment analysis model and TF-IDF vectorizer
# These files must exist in the ml/models/ directory
# The model was trained using the notebooks in ml/notebooks/
try:
    # Load the trained classifier (e.g., LogisticRegression)
    model = joblib.load(MODEL_PATH)
    # Load the TF-IDF vectorizer used during training
    # This must be the same vectorizer to ensure feature consistency
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Loaded vectorizer from: {VECTORIZER_PATH}")
# If the model or vectorizer file is not found, exit with error
except FileNotFoundError as e:
    print(f"Error: Model or vectorizer not found. {e}", file=sys.stderr)
    print("Please ensure the model files exist in ml/models/", file=sys.stderr)
    sys.exit(1)
# If there is any other error loading the model or vectorizer, exit with error
except Exception as e:
    print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================================
def clean_text(text):
    """
    Preprocess text using the same function as in the training notebook.
    
    This function MUST match the preprocessing used during model training.
    Any differences will cause incorrect predictions.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs, mentions, hashtags, special characters, numbers
    3. Tokenize the text
    4. Remove stopwords (but preserve negation words)
    5. Apply stemming to reduce words to root forms
    
    Args:
        text (str): Raw input text to preprocess
        
    Returns:
        str: Preprocessed text ready for vectorization
    """
    # Step 1: Normalize case - convert all text to lowercase
    # This ensures "Good" and "good" are treated the same
    text = text.lower()
    
    # Step 2: Remove URLs (http, https, www links)
    # Regex matches: http://..., https://..., www.example.com
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove Twitter-style mentions (@username)
    # Regex matches: @username
    text = re.sub(r'@\w+', '', text)
    
    # Step 4: Remove hashtag symbols but keep the word
    # "#awesome" becomes "awesome"
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Step 5: Remove all special characters (punctuation, symbols)
    # Keep only alphanumeric characters and whitespace
    text = re.sub(r'[^\w\s]', '', text)
    
    # Step 6: Remove all numbers
    # Numbers typically don't contribute to sentiment
    text = re.sub(r'\d+', '', text)
    
    # Step 7: Clean up whitespace
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space

    # Step 8: Tokenize - split text into individual words
    # Uses NLTK's word tokenizer which handles contractions properly
    tokens = nltk.word_tokenize(text)
    
    # Step 9: Stopword removal with negation preservation
    # Get standard English stopwords (common words like "the", "is", "a")
    stop_words = set(stopwords.words('english'))
    
    # Define negation words that MUST be preserved
    # These words reverse sentiment (e.g., "not good" vs "good")
    # Removing them would cause "not happy" to be classified as positive!
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', "n't", 'cannot', 
                     "don't", "doesn't", "didn't", "won't", "shouldn't", "couldn't", 
                     "wasn't", "weren't", "isn't", "aren't"}
    
    # Remove negation words from stopwords list
    # This ensures negation words are kept in the processed text
    final_stopwords = stop_words - negation_words
    
    # Step 10: Apply stemming and filter stopwords
    # Stemming reduces words to root form (e.g., "running" -> "run")
    stemmer = PorterStemmer()
    # Keep words that are either:
    # - Negation words (important for sentiment)
    # - Not in the final stopwords list
    tokens = [stemmer.stem(word) for word in tokens if word in negation_words or word not in final_stopwords]
    
    # Step 11: Join tokens back into a single string
    return ' '.join(tokens)

# ============================================================================
# SENTIMENT PREDICTION FUNCTION
# ============================================================================
def predict_sentiment(text):
    """
    Predict sentiment using the loaded model and vectorizer.
    
    This function replicates the prediction pipeline:
    1. Preprocess the input text (must match training preprocessing)
    2. Transform text to TF-IDF features using the trained vectorizer
    3. Predict sentiment using the trained model
    
    Args:
        text (str): Raw input text to analyze
        
    Returns:
        str: Predicted sentiment ("positive" or "negative")
             Returns "ERROR: ..." if preprocessing fails or text is empty
    """
    # Step 1: Preprocess the text using the same function as training
    # This is critical - preprocessing must match exactly
    processed_text = clean_text(text)
    
    # Step 2: Handle edge case - empty text after preprocessing
    # This can happen if text contains only URLs, mentions, or stopwords
    if not processed_text or processed_text.strip() == '':
        return "ERROR: Empty text after preprocessing"
    
    # Step 3: Transform text to numerical features using TF-IDF vectorizer
    # The vectorizer converts text to a sparse matrix of TF-IDF scores
    # Note: transform() expects a list, so we pass [processed_text]
    # This creates a feature vector matching the training data format
    text_vector = vectorizer.transform([processed_text])
    
    # Step 4: Predict sentiment using the trained classifier
    # The model outputs a prediction based on the feature vector
    prediction = model.predict(text_vector)
    
    # Step 5: Extract the prediction
    # model.predict() returns a numpy array (e.g., ['positive'] or ['negative'])
    # We return the first (and only) element as a string
    return prediction[0]

# ============================================================================
# TEST FUNCTION
# ============================================================================
def test_sentiment(text, expected, category=""):
    """
    Test a single text and return the result.
    
    This function runs a single test case by:
    1. Getting the model's prediction for the input text
    2. Comparing it to the expected sentiment
    3. Returning a result dictionary with all relevant information
    
    Args:
        text (str): The text to test
        expected (str): The expected sentiment ("positive" or "negative")
        category (str): Optional category name for grouping tests
        
    Returns:
        dict: A dictionary containing:
            - text: Original input text
            - expected: Expected sentiment
            - predicted: Model's prediction
            - correct: Boolean indicating if prediction matches expected
            - category: Test category name
    """
    try:
        # Get the model's prediction for this text
        prediction = predict_sentiment(text)
        
        # Check if prediction is correct
        # Compare lowercase versions to handle case differences
        # If prediction starts with "ERROR", it's automatically incorrect
        is_correct = prediction.lower() == expected.lower() if not prediction.startswith("ERROR") else False
    except Exception as e:
        # If any error occurs during prediction, mark as error
        prediction = f"ERROR: {str(e)}"
        is_correct = False
    
    # Return structured result for reporting
    return {
        'text': text,              # Original input text
        'expected': expected,      # Expected sentiment label
        'predicted': prediction,   # Model's actual prediction
        'correct': is_correct,     # Whether prediction matches expected
        'category': category       # Test category for grouping
    }

# ============================================================================
# REPORT GENERATION FUNCTION
# ============================================================================
def generate_report(results):
    """
    Generate a formatted test report and save it to report.txt.
    
    Creates a comprehensive report including:
    - Overall summary statistics
    - Detailed results grouped by category
    - Individual test pass/fail status
    - Category-wise accuracy breakdown
    
    Args:
        results (list): List of test result dictionaries from test_sentiment()
        
    Returns:
        Path: Path to the generated report file
    """
    # Initialize list to store report lines
    report_lines = []
    
    # Report header
    report_lines.append("=" * 80)
    report_lines.append("SENTIMENT ANALYSIS TEST REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Group results by category for organized reporting
    # This allows us to see which categories perform better/worse
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    # Calculate overall statistics
    total_tests = len(results)  # Total number of test cases
    total_correct = sum(1 for r in results if r['correct'])  # Count of correct predictions
    # Calculate accuracy percentage (avoid division by zero)
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    # ========================================================================
    # SUMMARY SECTION
    # ========================================================================
    # Display overall test summary at the top of the report
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Tests: {total_tests}")
    report_lines.append(f"Correct Predictions: {total_correct}")
    report_lines.append(f"Accuracy: {accuracy:.2f}%")
    report_lines.append("")
    
    # ========================================================================
    # DETAILED RESULTS BY CATEGORY
    # ========================================================================
    # For each category, show individual test results
    for category, cat_results in categories.items():
        # Calculate category-specific statistics
        cat_correct = sum(1 for r in cat_results if r['correct'])
        cat_accuracy = (cat_correct / len(cat_results) * 100) if cat_results else 0
        
        # Category header
        report_lines.append("=" * 80)
        report_lines.append(f"CATEGORY: {category.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Tests: {len(cat_results)} | Correct: {cat_correct} | Accuracy: {cat_accuracy:.2f}%")
        report_lines.append("")
        
        # List each test in this category
        for i, result in enumerate(cat_results, 1):
            # Mark test as PASS or FAIL
            status = "[PASS]" if result['correct'] else "[FAIL]"
            report_lines.append(f"Test {i}: {status}")
            report_lines.append(f"  Text: {result['text']}")
            report_lines.append(f"  Expected: {result['expected']}")
            report_lines.append(f"  Predicted: {result['predicted']}")
            # If test failed, highlight the mismatch
            if not result['correct']:
                report_lines.append(f"  [MISMATCH] Expected '{result['expected']}' but got '{result['predicted']}'")
            report_lines.append("")
    
    # ========================================================================
    # STATISTICS SUMMARY BY CATEGORY
    # ========================================================================
    # Quick reference table showing accuracy per category
    report_lines.append("=" * 80)
    report_lines.append("DETAILED STATISTICS BY CATEGORY")
    report_lines.append("=" * 80)
    for category, cat_results in categories.items():
        cat_correct = sum(1 for r in cat_results if r['correct'])
        cat_total = len(cat_results)
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        # Format: "Category Name              : 8/10 (80.0%)"
        report_lines.append(f"{category:30s}: {cat_correct}/{cat_total} ({cat_accuracy:.1f}%)")
    
    # Report footer
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # ========================================================================
    # FILE OUTPUT
    # ========================================================================
    # Save report to file in the same directory as this script
    report_path = SCRIPT_DIR / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console for immediate feedback
    print('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")
    
    return report_path

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================
def main():
    """
    Run all test cases across multiple categories.
    
    This function executes a comprehensive test suite covering:
    1. Baseline cases (obviously positive/negative)
    2. Negation handling (critical for accuracy)
    3. Mixed sentiment texts
    4. Sarcasm detection (challenging for ML models)
    5. Short and context-dependent phrases
    
    After running all tests, generates a detailed report.
    """
    print("Running sentiment analysis tests...")
    print("=" * 80)
    
    # List to store all test results
    results = []
    
    # ========================================================================
    # TEST CATEGORY 1: BASELINE TESTS ("Easy Wins")
    # ========================================================================
    # These are straightforward cases that should be easy for the model
    # They contain clear positive or negative language
    # Purpose: Verify the model works correctly on obvious cases
    print("\n1. Testing 'Easy Wins' (Baseline Test)...")
    # Test 1.1: Strongly positive review with multiple positive indicators
    results.append(test_sentiment(
        "Absolutely loved this! The quality is top-notch and shipping was incredibly fast. Highly recommended.",
        "positive",
        "Easy Wins - Baseline"
    ))
    # Test 1.2: Positive review with explicit rating and satisfaction
    results.append(test_sentiment(
        "Five stars. Best investment I've made all year. Works exactly as described.",
        "positive",
        "Easy Wins - Baseline"
    ))
    # Test 1.3: Strongly negative review with multiple problems
    results.append(test_sentiment(
        "Terrible experience. The item arrived broken and customer service was completely useless.",
        "negative",
        "Easy Wins - Baseline"
    ))
    # Test 1.4: Negative review with warning and specific complaint
    results.append(test_sentiment(
        "Waste of money. Do not buy this. It stopped working after two days.",
        "negative",
        "Easy Wins - Baseline"
    ))
    
    # ========================================================================
    # TEST CATEGORY 2: NEGATION HANDLING
    # ========================================================================
    # These tests verify the model correctly handles negation words
    # Critical because "not good" should be negative, not positive
    # This is where preserving negation words during preprocessing matters
    print("\n2. Testing 'Negation' Trap (Logic Test)...")
    # Test 2.1: "not good" should be negative (not positive)
    results.append(test_sentiment(
        "The plot was not good at all, despite the hype.",
        "negative",
        "Negation Trap"
    ))
    # Test 2.2: "no issues" is positive (double negative = positive)
    # This tests if the model understands that "no problems" = good
    results.append(test_sentiment(
        "I had no issues with the installation process.",
        "positive",
        "Negation Trap"
    ))
    
    # ========================================================================
    # TEST CATEGORY 3: MIXED SENTIMENT
    # ========================================================================
    # These texts contain both positive and negative elements
    # The model must determine the overall sentiment
    # These are challenging because they require understanding context and weighting
    print("\n3. Testing 'Mixed Sentiment' (The 77% Zone)...")
    # Test 3.1: Positive features but negative conclusion (software issues)
    # The negative aspects (buggy, crashes) should outweigh positives
    results.append(test_sentiment(
        "The screen is amazing and the battery life is great, but the software is buggy and crashes constantly.",
        "negative",  # Likely negative due to "buggy" and "crashes"
        "Mixed Sentiment"
    ))
    # Test 3.2: Minor negative but overall positive
    # The positive conclusion should outweigh the minor complaint
    results.append(test_sentiment(
        "It's a bit expensive for what you get, but the build quality makes up for it eventually.",
        "positive",  # Leaning positive
        "Mixed Sentiment"
    ))
    # Test 3.3: Positive start but negative conclusion
    # The negative ending should determine the overall sentiment
    results.append(test_sentiment(
        "I really wanted to like this because the design is beautiful, but it is just too uncomfortable to use for long periods.",
        "negative",  # Leaning negative
        "Mixed Sentiment"
    ))
    
    # ========================================================================
    # TEST CATEGORY 4: SARCASM DETECTION
    # ========================================================================
    # Sarcasm is extremely difficult for ML models because:
    # - Words are positive but meaning is negative
    # - Requires understanding context and tone
    # - Often uses positive words ironically
    print("\n4. Testing 'Sarcasm' (The Stress Test)...")
    # Test 4.1: "Great" used sarcastically - should be negative
    results.append(test_sentiment(
        "Great, another update that breaks everything. Just what I needed today.",
        "negative",  # Sarcastic - "Great" means the opposite here
        "Sarcasm"
    ))
    # Test 4.2: "Brilliant" and "Groundbreaking" used sarcastically
    # The context (20 minutes battery) makes it clearly negative
    results.append(test_sentiment(
        "Oh brilliant. The battery lasts a whole 20 minutes. Groundbreaking technology.",
        "negative",  # Sarcastic - positive words used ironically
        "Sarcasm"
    ))
    
    # ========================================================================
    # TEST CATEGORY 5: SHORT & CONTEXT-DEPENDENT
    # ========================================================================
    # Very short texts are challenging because:
    # - Limited context for the model to analyze
    # - May require external knowledge
    # - Ambiguous without surrounding context
    print("\n5. Testing 'Short & Context-Dependent'...")
    # Test 5.1: Single word expressing indifference/negativity
    results.append(test_sentiment(
        "Meh.",
        "negative",
        "Short & Context-Dependent"
    ))
    # Test 5.2: Single positive word
    results.append(test_sentiment(
        "Solid.",
        "positive",
        "Short & Context-Dependent"
    ))
    # Test 5.3: Action word implying negative experience
    results.append(test_sentiment(
        "Refunded.",
        "negative",
        "Short & Context-Dependent"
    ))
    # Test 5.4: Ambiguous word - "steal" can mean good (bargain) or bad (theft)
    # In product review context, "steal" usually means "great deal" (positive)
    results.append(test_sentiment(
        "Steal.",  # Context: "It was a steal" = great deal
        "positive",
        "Short & Context-Dependent"
    ))
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    # After all tests complete, generate comprehensive report
    print("\n" + "=" * 80)
    print("Generating report...")
    report_path = generate_report(results)
    
    print(f"\n[SUCCESS] Testing complete! Report saved to: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
# Run the test suite when this script is executed directly
# Usage: python test_sentiment.py
if __name__ == '__main__':
    main()

