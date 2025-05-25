# scripts/train_model.py (Updated for UI integration)

import pandas as pd
from sklearn.decomposition import TruncatedSVD 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import utils # Import our utility functions
import os

from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer  
import unsupervised_analysis

def train_categorization_model(data_filepath='data/transactions.xls'):
    """
    Trains the XGBoost model pipeline and saves it along with the LabelEncoder.
    Also trains and saves unsupervised clustering models.
    Returns True on success, False otherwise.
    """
    print("Attempting to load data from:", data_filepath) # Debug print
    df = utils.load_data(data_filepath)
    if df is None:
        print("Failed to load data. Aborting training.")
        return False

    os.makedirs('models', exist_ok=True) # Ensure models directory exists

    # Preprocessing: Encode the target variable for SUPERVISED model
    le = LabelEncoder()
    if 'Category' not in df.columns:
        print("Error: 'Category' column not found in training data. Supervised training aborted.")
        # We can still proceed with unsupervised if supervised fails due to missing category
        supervised_success = False
    else:
        df['Category_Encoded'] = le.fit_transform(df['Category'])
        utils.save_label_encoder(le, 'models/label_encoder.pkl')

        X = df[['Description', 'Amount', 'Merchant']]
        y = df['Category_Encoded']

        # Split the data
        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            print("Warning: Only one category present. Skipping stratified split for supervised training.")
            X_train, X_test, y_train, y_test = X, y, X, y # Use full data as train and test for simplicity

        print("--- Starting Supervised Model Training ---")

        preprocessor = ColumnTransformer(
            transformers=[
                ('description_tfidf', Pipeline([
                    ('text_preprocess', utils.TextPreprocessor(utils.preprocess_text_nlp)),
                    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=1000))
                ]), 'Description'),
                ('amount_scaler', StandardScaler(), ['Amount']),
                ('merchant_ohe', OneHotEncoder(handle_unknown='ignore'), ['Merchant'])
            ],
            remainder='drop'
        )

        xgb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(objective='multi:softmax',
                                             num_class=len(le.classes_),
                                             eval_metric='mlogloss',
                                             n_estimators=100,
                                             learning_rate=0.1,
                                             max_depth=5,
                                             subsample=0.8,
                                             colsample_bytree=0.8,
                                             random_state=42))
        ])

        try:
            print("Fitting the supervised model pipeline...")
            xgb_pipeline.fit(X_train, y_train)
            print("Supervised model fitting complete.")

            if len(X_test) > 0 and len(y_test.unique()) > 1:
                y_pred_test = xgb_pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_test)
                print(f"\nSupervised Model Test Accuracy: {accuracy:.4f}")
                print(f"Classification Report:\n{classification_report(y_test, y_pred_test, target_names=le.classes_)}")
            else:
                print("Skipping supervised evaluation due to insufficient test data.")

            utils.save_model(xgb_pipeline, 'models/xgb_categorizer_pipeline.pkl')
            print("--- Supervised Model Training Complete ---")
            supervised_success = True
        except Exception as e:
            print(f"An error occurred during supervised model training: {e}")
            supervised_success = False

    unsupervised_success = perform_clustering_and_save_models(data_filepath) 

    return supervised_success and unsupervised_success 


def perform_clustering_and_save_models(data_filepath):
    """
    Wrapper to call perform_clustering from unsupervised_analysis and save its components.
    """
    print("\n--- Starting Unsupervised Model (TF-IDF + K-Means) Training and Saving ---")
    df = utils.load_data(data_filepath)
    if df is None:
        print("Failed to load data for unsupervised training.")
        return False

    descriptions = df['Description'].apply(utils.preprocess_text_nlp)
    if descriptions.isnull().all() or descriptions.empty or descriptions.str.strip().eq('').all():
        print("No valid descriptions for unsupervised model training. Skipping.")
        return False

    # Train and save TF-IDF Vectorizer
    unsupervised_tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    try:
        unsupervised_tfidf_vectorizer.fit(descriptions)
        utils.save_model(unsupervised_tfidf_vectorizer, 'models/unsupervised_tfidf_vectorizer.pkl')
        print("Unsupervised TF-IDF Vectorizer saved.")
    except Exception as e:
        print(f"Error training/saving unsupervised TF-IDF Vectorizer: {e}")
        return False

    # Transform data for K-Means
    X_unsupervised_tfidf = unsupervised_tfidf_vectorizer.transform(descriptions)

    # Handle cases where TF-IDF might produce zero features or too few features
    if X_unsupervised_tfidf.shape[1] == 0:
        print("TF-IDF produced zero features. Cannot train K-Means.")
        return False

    # Dimensionality reduction (important for K-Means on sparse data)
    n_components_svd = min(X_unsupervised_tfidf.shape[0] - 1, X_unsupervised_tfidf.shape[1] - 1, 50)
    if n_components_svd < 1:
        print(f"Not enough data points or features for meaningful SVD ({n_components_svd}). K-Means will run on raw TF-IDF.")
        X_reduced_for_kmeans = X_unsupervised_tfidf
    else:
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X_reduced_for_kmeans = svd.fit_transform(X_unsupervised_tfidf)
        utils.save_model(svd, 'models/unsupervised_svd_transformer.pkl') # Save SVD transformer
        print("Unsupervised SVD transformer saved.")

    # Train and save K-Means model
    n_clusters = 5 # Default number of clusters. Can be made configurable.
    if X_reduced_for_kmeans.shape[0] < n_clusters:
        actual_n_clusters = X_reduced_for_kmeans.shape[0]
        if actual_n_clusters < 2:
            print("Not enough data points to form clusters for K-Means (less than 2). Skipping K-Means.")
            return False
        print(f"Adjusting K-Means n_clusters from {n_clusters} to {actual_n_clusters} due to small dataset size.")
        n_clusters = actual_n_clusters

    try:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(X_reduced_for_kmeans)
        utils.save_model(kmeans_model, 'models/kmeans_model.pkl')
        print(f"Unsupervised K-Means model trained with {n_clusters} clusters and saved.")
    except Exception as e:
        print(f"Error training/saving K-Means model: {e}")
        return False

    print("--- Unsupervised Model Training and Saving Complete ---")
    return True

if __name__ == "__main__":
    # This block remains for direct execution/testing, but won't be called by the UI
    train_categorization_model()