# scripts/unsupervised_analysis.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD # For dimensionality reduction before K-Means
import utils # For preprocess_text_nlp and load_data
import numpy as np
import os # Added for path handling

def perform_clustering(data_filepath, n_clusters=5):
    """
    Performs K-Means clustering on transaction descriptions (TF-IDF features).
    Returns a DataFrame with original data and assigned cluster IDs,
    and a dictionary of top terms per cluster.
    """
    print(f"Loading data for unsupervised analysis from: {data_filepath}")
    df = utils.load_data(data_filepath)
    if df is None:
        print("Failed to load data for clustering.")
        return None, "Failed to load data."

    if 'Description' not in df.columns or df['Description'].empty:
        print("No 'Description' column or descriptions are empty for clustering.")
        return None, "No valid descriptions found for clustering."

    descriptions = df['Description'].apply(utils.preprocess_text_nlp)

    if descriptions.isnull().all() or descriptions.empty or descriptions.str.strip().eq('').all():
        print("All descriptions are empty or whitespace after preprocessing.")
        return None, "All descriptions are empty or whitespace after preprocessing. Cannot cluster."

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
    try:
        X_tfidf = tfidf_vectorizer.fit_transform(descriptions)
    except ValueError as e:
        print(f"Error during TF-IDF vectorization: {e}. Check if descriptions are all empty/invalid.")
        return None, f"Error during TF-IDF vectorization: {e}"

    # Handle cases where TF-IDF might produce zero features or too few features
    if X_tfidf.shape[1] == 0:
        print("TF-IDF produced zero features. Cannot cluster.")
        return None, "TF-IDF produced zero features. Cannot cluster."

    # Dimensionality reduction (important for K-Means on sparse data like TF-IDF)
    # Ensure n_components is valid: min(num_samples - 1, num_features - 1, desired_max)
    n_components_svd = min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1, 50)
    if n_components_svd < 1: # Must be at least 1 for SVD
        print(f"Not enough data points or features for meaningful dimensionality reduction (n_components={n_components_svd}). Skipping SVD.")
        X_reduced = X_tfidf # Use original TF-IDF directly if SVD not possible
    else:
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X_reduced = svd.fit_transform(X_tfidf)


    # K-Means Clustering
    # Ensure n_clusters is valid given the data size
    if X_reduced.shape[0] < n_clusters:
        actual_n_clusters = X_reduced.shape[0] # Can't have more clusters than samples
        if actual_n_clusters < 2:
            print("Not enough data points to form clusters (less than 2).")
            return None, "Not enough data points to form clusters."
        print(f"Adjusting n_clusters from {n_clusters} to {actual_n_clusters} due to small dataset size.")
        n_clusters = actual_n_clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)

    df['Cluster_ID'] = clusters

    # Get top terms for each cluster for interpretation
    top_terms_per_cluster = {}
    feature_names = tfidf_vectorizer.get_feature_names_out()

    for i in range(n_clusters):
        cluster_docs_indices = np.where(clusters == i)[0]
        if len(cluster_docs_indices) == 0:
            top_terms_per_cluster[i] = []
            continue

        # Get the average TF-IDF scores for terms within this cluster
        cluster_tfidf_matrix_original = X_tfidf[cluster_docs_indices]
        avg_tfidf_scores = cluster_tfidf_matrix_original.mean(axis=0).A1 # .A1 converts sparse matrix row to numpy array

        top_indices = avg_tfidf_scores.argsort()[::-1][:10] # Top 10 terms
        top_terms = [feature_names[idx] for idx in top_indices]
        top_terms_per_cluster[i] = top_terms

    return df, top_terms_per_cluster

if __name__ == "__main__":
    # Example usage for direct testing (adjust path as needed)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'transactions.xlsx')

    print(f"Attempting direct clustering test from: {data_path}")
    clustered_df, top_terms = perform_clustering(data_path, n_clusters=3)
    if clustered_df is not None:
        print("\nClustering Results (first 5 rows):")
        print(clustered_df[['Description', 'Cluster_ID']].head())
        print("\nTop Terms per Cluster:")
        for cluster_id, terms in top_terms.items():
            print(f"Cluster {cluster_id}: {', '.join(terms)}")
    else:
        print(f"Clustering failed: {top_terms}") # top_terms contains error message here