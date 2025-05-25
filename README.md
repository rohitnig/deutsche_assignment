# Transaction Categorizer ML Solution

This project provides a machine learning solution for categorizing financial transactions based on their description, amount, and merchant. It incorporates both supervised and unsupervised learning approaches, wrapped in a user-friendly Flask web interface, allowing for predictions, batch processing, and continuous model improvement through human feedback and retraining.

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Setup Instructions](#setup-instructions)
    * [Prerequisites](#prerequisites)
    * [Clone the Repository](#clone-the-repository)
    * [Initial Data Setup](#initial-data-setup)
    * [Python Environment Setup](#python-environment-setup)
4.  [How to Run](#how-to-run)
    * [Option 1: Run Directly with Python (Development/Testing)](#option-1-run-directly-with-python-developmenttesting)
    * [Option 2: Run with Docker (Recommended for Deployment)](#option-2-run-with-docker-recommended-for-deployment)
5.  [Usage Guide (Web UI)](#usage-guide-web-ui)
    * [Predict Single Transaction](#predict-single-transaction)
    * [Upload Transactions for Batch Prediction](#upload-transactions-for-batch-prediction)
    * [Upload New Training Data & Retrain](#upload-new-training-data--retrain)
    * [Explore Unsupervised Clusters](#explore-unsupervised-clusters)
    * [Human-in-the-Loop (Add/Correct Data)](#human-in-the-loop-addcorrect-data)
6.  [Key Technologies Used](#key-technologies-used)
7.  [Future Enhancements](#future-enhancements)

---

## Features

* **Supervised Learning:** Trains an XGBoost classifier to predict transaction categories based on provided labels.
* **Unsupervised Learning:** Utilizes TF-IDF and K-Means clustering to discover natural groupings within transaction descriptions, without relying on predefined categories.
* **Web User Interface (Flask):**
    * Predict categories for single transactions.
    * Upload CSV/Excel files for batch prediction.
    * Upload new training data files to retrain the model.
    * Add/correct individual transaction categories to improve the training dataset (Human-in-the-Loop).
    * Explore unsupervised clusters to gain insights into data patterns.
* **Modular Design:** Separated logic for utilities, training, prediction, and unsupervised analysis.
* **Reproducible Environment:** `requirements.txt` for Python dependencies and `Dockerfile` for containerized deployment.

## Project Structure
```
transaction_categorizer/
├── data/
│   └── transactions.xls      
│   └── dummy_new_transactions.csv 
├── models/                   # Stores trained ML models (.pkl files)
│   └── xgb_categorizer_pipeline.pkl
│   └── label_encoder.pkl
│   └── unsupervised_tfidf_vectorizer.pkl
│   └── unsupervised_svd_transformer.pkl
│   └── kmeans_model.pkl
├── scripts/
│   ├── train_model.py        # Script for supervised and unsupervised model training
│   ├── predict_category.py   # Script for making predictions
│   ├── unsupervised_analysis.py # Script for unsupervised clustering logic
│   └── utils.py              # Helper functions (data loading, preprocessing, model saving/loading)
├── app/
│   ├── app.py                # Flask web application backend
│   ├── templates/            # HTML templates for the UI
│   │   ├── index.html
│   │   ├── results.html
│   │   └── unsupervised_results.html
│   └── static/               # Static files (CSS)
│       └── style.css
├── .dockerignore             # Files/folders to ignore when building Docker image
├── Dockerfile                # Instructions for building the Docker image
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### Prerequisites

* **Python 3.8+** (Recommended to use a virtual environment)
* **pip** (Python package installer)
* **Git** (for cloning the repository)
* **Docker Desktop** (if you choose to run with Docker)

### Clone the Repository

```bash
git clone https://github.com/rohitnig/deutsche_assignment
cd deutsche_assignment
```
Initial Data Setup

Place your initial training data file, named transactions.xls, in the data/ directory.
Ensure it has the following columns: Date, Amount, Merchant, Category, Description.

You can also create a dummy_new_transactions.csv in the data/ folder for testing batch predictions.

Example data/transactions.xls content:
```
Date,Amount,Merchant,Category,Description
2023-01-01,25.50,Starbucks,Food,Coffee and sandwich
2023-01-02,100.00,Amazon,Shopping,Books and electronics
2023-01-03,15.75,Local Cafe,Food,Lunch at downtown cafe
```

Python Environment Setup

It's highly recommended to use a virtual environment to manage dependencies.
```bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Download the SpaCy English language model (Crucial for NLP)
python -m spacy download en_core_web_sm
```

---

## How to Run

### Option 1: Run Directly with Python (Development/Testing)

This method runs the Flask application directly using your local Python environment.

1.  **Activate your virtual environment** (if not already active):
    * Windows: `.\venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
2.  **Navigate to the `app` directory:**
    ```bash
    cd app
    ```
3.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will start, typically accessible at `http://127.0.0.1:5000/`.
    * If models are not found (`models/` folder is empty or files are missing), the app will attempt an initial training using `data/transactions.xls`.

### Option 2: Run with Docker (Recommended for Deployment)

This method builds a self-contained Docker image of your application, ensuring consistent execution across different environments.

1.  **Ensure Docker Desktop is running** on your machine.
2.  **Navigate to the project root directory** (where `Dockerfile` is located):
    ```bash
    cd transaction_categorizer
    ```
3.  **Build the Docker image:**
    ```bash
    docker build -t transaction-categorizer .
    ```
    This might take a few minutes the first time.
4.  **Run the Docker container:**
    ```bash
    docker run -p 5000:5000 transaction-categorizer
    ```
    The application will start inside the Docker container, accessible at `http://127.0.0.1:5000/` in your web browser.
    * The Docker container will automatically perform initial model training if the models are not present within the image (first run).

## Usage Guide (Web UI)

Access the application in your browser at `http://127.0.0.1:5000/`.

### Predict Single Transaction

* Fill in the `Description`, `Amount`, and `Merchant` fields.
* Click "Predict Category".
* The predicted category, confidence, and unsupervised cluster ID will be displayed.

### Upload Transactions for Batch Prediction

* Click "Choose File" under "Upload Transactions for Batch Prediction".
* Select a `.csv` or `.xlsx` file containing `Description`, `Amount`, and `Merchant` columns.
* Click "Upload & Predict Batch".
* You will be redirected to a results page showing predictions for all transactions in the uploaded file.

### Upload New Training Data & Retrain

* This feature allows you to replace your entire `transactions.xls` file with an updated version and trigger retraining.
* Click "Choose File" under "Upload New Training Data & Retrain".
* Select a `.csv` or `.xlsx` file (it should have `Date`, `Amount`, `Merchant`, `Category`, `Description` columns).
* Click "Upload & Retrain Model".
* The uploaded file will replace `data/transactions.xls`, and both the supervised and unsupervised models will be retrained using this new dataset.

### Explore Unsupervised Clusters

* Enter the desired number of clusters (K) in the input field.
* Click "Analyze & Explore Clusters".
* The application will perform K-Means clustering on the descriptions in your current `transactions.xls` file.
* A new page will display each cluster, its top associated terms, and the transactions belonging to that cluster. This helps in understanding natural groupings in your data.

### Human-in-the-Loop (Add/Correct Data)

* After a single transaction prediction, if the prediction is incorrect or you want to add a new labeled transaction to your training data:
    * The `Description`, `Amount`, and `Merchant` fields will be pre-filled.
    * Enter the `Correct Category` (or a new category if it's a new type of transaction).
    * Click "Add to Training Data".
    * This will append the new labeled transaction to your `data/transactions.xls` file. Remember to then **"Upload New Training Data & Retrain"** to incorporate these changes into your model.
 
## Key Technologies Used

* **Python:** Core programming language.
* **Flask:** Lightweight web framework for the UI.
* **Pandas:** Data manipulation and analysis.
* **scikit-learn:** Machine learning utilities, preprocessing, and pipeline management.
* **XGBoost:** Gradient Boosting model for supervised classification.
* **SpaCy:** Natural Language Processing library for text preprocessing and embeddings.
* **NumPy:** Fundamental package for numerical computing.
* **Docker:** For containerization and reproducible deployment.
* **xlwt:** Python library for writing to .xls Excel files.

## Future Enhancements

* **Model Monitoring:** Implement logging and dashboards to track model performance over time (e.g., accuracy, drift detection).
* **API Endpoints:** Create dedicated REST API endpoints for programmatic access to prediction and retraining.
* **Advanced Unsupervised Visualization:** Implement interactive visualizations (e.g., t-SNE plots) for clusters.
* **User Authentication:** For multi-user environments.
* **Database Integration:** Replace `transactions.xls` with a proper database (e.g., SQLite, PostgreSQL) for more robust data management.
* **Scheduled Retraining:** Automate the retraining process on a schedule.
* **More Robust Error Handling:** Provide more specific user-facing error messages for various failure scenarios.
* **Configuration Management:** Use a separate config file for paths, hyperparameters, etc.

  
