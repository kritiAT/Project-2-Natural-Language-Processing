"""
Reusable utilities for the NLP text classification project
"""

import os
import re
import json
import pickle
from datetime import datetime

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_FILE = "results_log.jsonl"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Text cleaning + preprocessing (combined)
# ---------------------------------------------------------------------------
def clean_and_preprocess_text(text):
    """
    Full text cleaning + preprocessing pipeline, combining:
    - HTML/JS/CSS/comment stripping
    - URL removal
    - special character / number / single-character removal
    - lowercasing
    - tokenization
    - stopword removal (optional)
    - lemmatization (optional)

    Parameters
    ----------
    text : str
        Raw input text.
    remove_stopwords : bool, default True
        Whether to filter out English stopwords.
    lemmatize : bool, default True
        Whether to lemmatize remaining tokens.

    Returns
    -------
    str
        Cleaned, preprocessed text (space-joined tokens).
    """
    text = str(text)

    # --- Cleaning ---
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)  # inline JS
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)     # inline CSS
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)                                # HTML comments
    text = re.sub(r'<[^>]+>', '', text)                                                    # remaining HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)                                             # URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)                                            # special characters
    text = re.sub(r'\d+', '', text)                                                        # numbers
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)                                              # single characters
    text = re.sub(r'^\s*[a-zA-Z]\s+', '', text)                                            # leading single char
    text = re.sub(r'\s+', ' ', text)                                                       # collapse whitespace
    text = re.sub(r'^b\s+', '', text)                                                      # stray prefixed 'b'
    text = text.lower()

    # --- Preprocessing (tokenize -> stopwords -> lemmatize) ---
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.lower() not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]

    return " ".join(tokens)


def preprocess_text_columns(X, text_columns):
    """
    Apply clean_and_preprocess_text() to each text column in a dataframe copy.
    Call this SEPARATELY for train and test sets (fit-free, so no leakage).

    Adds a new column '{col}_clean' for each column in text_columns.
    """
    X = X.copy()
    for col in text_columns:
        X[f'{col}_clean'] = X[col].apply(lambda t: clean_and_preprocess_text(t))
    return X


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Predict on train and test data and compute evaluation metrics.
    Returns a dict of metrics + the confusion matrix (test-based).
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'y_pred': y_test_pred,
    }
    return metrics


# ---------------------------------------------------------------------------
# Persistence: results log (JSONL) + model files (pickle)
# ---------------------------------------------------------------------------
def save_result_to_file(result, filepath=RESULTS_FILE):
    """
    Append a single result to a JSONL file (one JSON object per line).
    Excludes non-serializable objects (fitted model, y_pred array).
    """
    serializable_result = {
        'timestamp': result['timestamp'],
        'model_name': result['model_name'],
        'method': result['method'],
        'comments': result['comments'],
        'train_accuracy': result['train_accuracy'],
        'test_accuracy': result['test_accuracy'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1_score': result['f1_score'],
        'confusion_matrix': result['confusion_matrix'].tolist(),
        'model_path': result.get('model_path'),
    }

    with open(filepath, 'a') as f:
        f.write(json.dumps(serializable_result) + "\n")


def load_results_from_file(filepath=RESULTS_FILE):
    """Load all saved results from the JSONL file into a DataFrame."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def save_model_to_file(model, model_name, method, timestamp, models_dir=MODELS_DIR):
    """
    Save the fitted model to disk using pickle.
    Filename includes model name, feature method, and timestamp to avoid overwriting.
    Returns the path the model was saved to.
    """
    safe_name = f"{model_name}_{method}_{timestamp}".replace(" ", "_").replace(":", "-")
    filepath = os.path.join(models_dir, f"{safe_name}.pkl")

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    return filepath


def load_model_from_file(filepath):
    """Load a previously saved model back from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Train + evaluate + persist (the "super function")
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train):
    """Fit the given model on training data. Returns the fitted model."""
    model.fit(X_train, y_train)
    return model


def train_evaluate_model(model, method, model_name, X_train, y_train, X_test, y_test,
                          comments="", results_container=None,
                          save_to_file=True, filepath=RESULTS_FILE,
                          save_model=True, models_dir=MODELS_DIR):
    """
    Super-function: trains, predicts, evaluates, and optionally stores results
    (in-memory + on disk) and the fitted model itself (on disk), so everything
    persists across notebook restarts.

    Parameters
    ----------
    model : sklearn-like estimator (must implement .fit and .predict)
    method : str, feature-extraction method used (e.g. "TF-IDF")
    model_name : str, name/label for this run (e.g. "Logistic Regression")
    comments : str, free-text notes (e.g. hyperparameters, feature set used)
    results_container : list, optional. If provided, the result dict is appended to it.
    save_to_file : bool, whether to append this result to disk immediately (default True)
    filepath : str, path to the JSONL results log file
    save_model : bool, whether to pickle the fitted model to disk (default True)
    models_dir : str, directory to save models into

    Returns
    -------
    result : dict with keys:
        timestamp, model_name, method, comments, train_accuracy, test_accuracy,
        precision, recall, f1_score, confusion_matrix, model (fitted object), model_path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fitted_model = train_model(model, X_train, y_train)
    metrics = evaluate_model(fitted_model, X_train, y_train, X_test, y_test)

    result = {
        'timestamp': timestamp,
        'model_name': model_name,
        'method': method,
        'comments': comments,
        'train_accuracy': metrics['train_accuracy'],
        'test_accuracy': metrics['test_accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'confusion_matrix': metrics['confusion_matrix'],
        'model': fitted_model,
        'model_path': None,
    }

    print(f"--- {model_name} | {method} ({comments}) ---")
    print(f"Train Accuracy: {result['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {result['test_accuracy']:.4f}")
    print(f"Precision:      {result['precision']:.4f}")
    print(f"Recall:         {result['recall']:.4f}")
    print(f"F1 Score:       {result['f1_score']:.4f}")
    print("Confusion Matrix:\n", result['confusion_matrix'])

    if save_model:
        model_path = save_model_to_file(fitted_model, model_name, method, timestamp, models_dir=models_dir)
        result['model_path'] = model_path
        print(f"Model saved to: {model_path}")

    print()

    if results_container is not None:
        results_container.append(result)

    if save_to_file:
        save_result_to_file(result, filepath=filepath)

    return result