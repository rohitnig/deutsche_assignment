# scripts/predict_category.py

import pandas as pd
import numpy as np # Ensure this is imported
import utils # For preprocess_text_nlp
import os
_unsupervised_tfidf_vectorizer = None
_unsupervised_svd_transformer = None # NEW: For TruncatedSVD
_kmeans_model = None
_model = None
_label_encoder = None

def _load_resources():
    """Helper to load all models and label encoder if not already loaded."""
    global _model, _label_encoder, _unsupervised_tfidf_vectorizer, _unsupervised_svd_transformer, _kmeans_model

    # Supervised model loading
    if _model is None:
        _model = utils.load_model('models/xgb_categorizer_pipeline.pkl')
    if _label_encoder is None:
        _label_encoder = utils.load_label_encoder('models/label_encoder.pkl')

    # Unsupervised model loading
    if _unsupervised_tfidf_vectorizer is None:
        _unsupervised_tfidf_vectorizer = utils.load_model('models/unsupervised_tfidf_vectorizer.pkl')
    if _unsupervised_svd_transformer is None: # NEW: Load SVD
        _unsupervised_svd_transformer = utils.load_model('models/unsupervised_svd_transformer.pkl')
    if _kmeans_model is None:
        _kmeans_model = utils.load_model('models/kmeans_model.pkl')

    # Ensure all models are loaded
    return (_model is not None and _label_encoder is not None and
            _unsupervised_tfidf_vectorizer is not None and
            _unsupervised_svd_transformer is not None and # NEW: Check SVD
            _kmeans_model is not None)

# Modify predict_single_transaction_api to include cluster prediction
def predict_single_transaction_api(description, amount, merchant):
    """
    Predicts the category and cluster for a single new transaction.
    Returns a tuple (predicted_category, confidence, cluster_id) or (None, None, None) on failure.
    """
    if not _load_resources():
        print("Error: One or more models/encoders not loaded. Cannot make predictions.")
        return None, None, None

    # Create a DataFrame for the new transaction (for supervised model)
    new_data_df = pd.DataFrame([{
        'Description': description,
        'Amount': amount,
        'Merchant': merchant
    }])

    # Preprocess description for unsupervised model
    unsupervised_text_processed = utils.preprocess_text_nlp(description)

    try:
        # Supervised prediction
        predicted_encoded_probs = _model.predict_proba(new_data_df)[0]
        predicted_encoded = _model.predict(new_data_df)[0]
        predicted_category = _label_encoder.inverse_transform([predicted_encoded])[0]
        confidence = float(np.max(predicted_encoded_probs))

        # Unsupervised prediction
        cluster_id = -1 # Default if clustering fails
        if unsupervised_text_processed and _unsupervised_tfidf_vectorizer and _kmeans_model:
            unsupervised_tfidf_features = _unsupervised_tfidf_vectorizer.transform([unsupervised_text_processed])

            if unsupervised_tfidf_features.shape[1] > 0 and unsupervised_tfidf_features.sum() > 0:
                # Apply SVD if it was used during training
                if _unsupervised_svd_transformer:
                    unsupervised_features_reduced = _unsupervised_svd_transformer.transform(unsupervised_tfidf_features)
                else:
                    unsupervised_features_reduced = unsupervised_tfidf_features # No SVD used

                if unsupervised_features_reduced.shape[1] > 0: # Ensure features exist after SVD
                    cluster_id = int(_kmeans_model.predict(unsupervised_features_reduced)[0])
                else:
                    print("Warning: Unsupervised features became empty after SVD. Cannot cluster.")
            else:
                print("Warning: Unsupervised TF-IDF features empty for description. Cannot cluster.")
        else:
            print("Warning: Unsupervised models not available or description empty. Cannot cluster.")


        return predicted_category, confidence, cluster_id
    except Exception as e:
        print(f"Error during single prediction: {e}")
        return None, None, None

# Modify predict_batch_transactions_api to include cluster prediction
def predict_batch_transactions_api(filepath):
    """
    Loads new transactions from a CSV/Excel file and predicts categories and clusters for all.
    Returns a DataFrame with predictions or None on failure.
    """
    if not _load_resources():
        print("Error: Models or LabelEncoder not loaded. Cannot make predictions.")
        return None

    try:
        # ... (existing code for loading new_transactions_df) ...
        if filepath.lower().endswith('.csv'):
            new_transactions_df = pd.read_csv(filepath)
        elif filepath.lower().endswith('.xlsx'):
            new_transactions_df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

        required_cols = ['Description', 'Amount', 'Merchant']
        if not all(col in new_transactions_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {filepath}. Expected: {required_cols}")

        # Supervised predictions
        predicted_encoded = _model.predict(new_transactions_df[required_cols])
        predicted_probabilities = _model.predict_proba(new_transactions_df[required_cols])
        predicted_categories = _label_encoder.inverse_transform(predicted_encoded)
        confidences = np.max(predicted_probabilities, axis=1)

        # Unsupervised predictions
        cluster_ids = []
        if _unsupervised_tfidf_vectorizer and _kmeans_model:
            unsupervised_descriptions = new_transactions_df['Description'].apply(utils.preprocess_text_nlp)
            unsupervised_tfidf_features = _unsupervised_tfidf_vectorizer.transform(unsupervised_descriptions)

            for i in range(unsupervised_tfidf_features.shape[0]):
                single_desc_features = unsupervised_tfidf_features[i]
                current_cluster_id = -1

                if single_desc_features.shape[1] > 0 and single_desc_features.sum() > 0:
                    # Apply SVD if it was used during training
                    if _unsupervised_svd_transformer:
                        single_desc_features_reduced = _unsupervised_svd_transformer.transform(single_desc_features)
                    else:
                        single_desc_features_reduced = single_desc_features

                    if single_desc_features_reduced.shape[1] > 0:
                        current_cluster_id = int(_kmeans_model.predict(single_desc_features_reduced)[0])
                cluster_ids.append(current_cluster_id)
        else:
            print("Warning: Unsupervised models not fully loaded. Skipping cluster prediction for batch.")
            cluster_ids = [-1] * len(new_transactions_df) # Assign -1 for all if models not available


        new_transactions_df['Predicted_Category'] = predicted_categories
        new_transactions_df['Confidence'] = confidences
        new_transactions_df['Cluster_ID'] = cluster_ids # Add cluster ID

        return new_transactions_df

    except FileNotFoundError:
        print(f"Error: Prediction data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        return None