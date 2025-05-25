# scripts/utils.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import spacy
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# Load SpaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    raise

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for text preprocessing to be used in scikit-learn pipelines.
    Applies a given preprocessing function to each text entry.
    """
    def __init__(self, preprocess_func):
        self.preprocess_func = preprocess_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a Series or DataFrame column for apply method
        if isinstance(X, pd.Series):
            return X.apply(self.preprocess_func)
        elif isinstance(X, np.ndarray): # Also added np.ndarray case for robustness
            # If it's a numpy array (e.g., from a single column selection), convert to Series
            return pd.Series(X.flatten()).apply(self.preprocess_func)
        else:
            raise TypeError("Input must be a pandas Series or numpy array.")


def preprocess_text_nlp(text):
    """
    Basic text preprocessing for descriptions: lowercasing, removing punctuation.
    More advanced steps like stop word removal or lemmatization can be added here
    if they improved your model performance.
    """
    if pd.isna(text) or not isinstance(text, str):
        return "" # Handle NaN or non-string descriptions gracefully
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def get_word_embedding(text):
    """
    Generates an averaged word embedding for a given text using SpaCy.
    Used for the embedding-based pipeline if chosen.
    """
    if pd.isna(text) or not isinstance(text, str):
        return np.zeros(nlp.vocab.vectors.shape[1])

    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector and not token.is_punct and not token.is_space]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp.vocab.vectors.shape[1])

def load_data(filepath='data/transactions.xlsx'):
    """Loads the raw transaction data."""
    try:
        df = pd.read_excel(filepath)
        # Ensure necessary columns exist
        required_cols = ['Date', 'Amount', 'Merchant', 'Category', 'Description']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {filepath}. Expected: {required_cols}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_model(model, filepath='models/xgb_categorizer_pipeline.pkl'):
    """Saves the trained model pipeline."""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filepath='models/xgb_categorizer_pipeline.pkl'):
    """Loads a trained model pipeline."""
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Placeholder for LabelEncoder - will be saved/loaded with the pipeline if in it
# Or saved separately if target encoding is done outside the pipeline
def save_label_encoder(le, filepath='models/label_encoder.pkl'):
    """Saves the LabelEncoder for inverse transformation."""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(le, file)
        print(f"LabelEncoder saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving LabelEncoder: {e}")

def load_label_encoder(filepath='models/label_encoder.pkl'):
    """Loads the LabelEncoder."""
    try:
        with open(filepath, 'rb') as file:
            le = pickle.load(file)
        print(f"LabelEncoder loaded successfully from {filepath}")
        return le
    except FileNotFoundError:
        print(f"Error: LabelEncoder file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading LabelEncoder: {e}")