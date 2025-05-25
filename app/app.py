# app/app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from werkzeug.utils import secure_filename
import sys
import shutil # For moving files

# Add the scripts directory to the Python path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import predict as predict_category
import train as train_model
import utils 
import unsupervised_analysis 

app = Flask(__name__)
app.secret_key = 'supersecretkey_for_flash_messages' # Needed for flash messages
app.config['UPLOAD_FOLDER'] = 'data' # Directory to temporarily store uploaded files
app.config['UPLOAD_TEMP_FOLDER'] = os.path.join(os.path.dirname(__file__), 'temp_uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

ALLOWED_EXTENSIONS = {'csv', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        # --- Handle Single Prediction ---
        if 'description' in request.form:
            description = request.form['description']
            try:
                amount = float(request.form['amount'])
            except ValueError:
                flash("Invalid amount. Please enter a number.", 'error')
                return render_template('index.html', prediction_result=None)
            merchant = request.form['merchant']

            category, confidence, cluster_id  = predict_category.predict_single_transaction_api(description, amount, merchant)

            if category:
                prediction_result = {
                    'description': description,
                    'amount': amount,
                    'merchant': merchant,
                    'category': category,
                    'confidence': f"{confidence*100:.2f}%",
                    'cluster_id': cluster_id
                }
                flash(f"Predicted: {category} (Confidence: {confidence*100:.2f}%)", 'success')
            else:
                flash("Error predicting category. Model might not be trained or data invalid.", 'error')

        # --- Handle Batch Upload ---
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                flash('File uploaded successfully. Processing...', 'success')

                # Predict batch categories
                batch_predictions_df = predict_category.predict_batch_transactions_api(filepath)
                os.remove(filepath) # Clean up uploaded file

                if batch_predictions_df is not None:
                    # Convert DataFrame to list of dicts for rendering in HTML
                    batch_results = batch_predictions_df.to_dict(orient='records')
                    return render_template('results.html', predictions=batch_results)
                else:
                    flash("Error processing batch file. Check file format and content.", 'error')
            else:
                flash('Allowed file types are csv, xls', 'error')

    return render_template('index.html', prediction_result=prediction_result)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to trigger model retraining."""
    data_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.xls')    
    # A more robust solution would allow uploading a new training dataset

    if train_model.train_categorization_model(data_filepath):
        flash("Model retraining initiated successfully!", 'success')
    else:
        flash("Model retraining failed. Check server logs.", 'error')
    return redirect(url_for('index'))

@app.route('/add_data', methods=['POST'])
def add_data():
    """Endpoint to add new labeled data for future retraining."""
    description = request.form['description']
    amount = float(request.form['amount'])
    merchant = request.form['merchant']
    category = request.form['category'] # The manually entered/corrected category

    new_transaction = pd.DataFrame([{
        'Date': pd.Timestamp.now(), # Add current date
        'Amount': amount,
        'Merchant': merchant,
        'Category': category,
        'Description': description
    }])

    data_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.xls')
    try:
        # Load existing data
        if os.path.exists(data_filepath):
            existing_df = pd.read_excel(data_filepath)
        else:
            existing_df = pd.DataFrame(columns=['Date', 'Amount', 'Merchant', 'Category', 'Description'])

        # Append new data
        updated_df = pd.concat([existing_df, new_transaction], ignore_index=True)
        updated_df.to_excel(data_filepath, index=False, engine='xlwt')
        flash("New data added to training set. Retrain model to incorporate changes!", 'success')
    except Exception as e:
        flash(f"Error adding data: {e}", 'error')

    return redirect(url_for('index'))

@app.route('/upload_train_data', methods=['POST'])
def upload_train_data():
    """Endpoint to upload a new training data file and trigger retraining."""
    if 'train_data_file' not in request.files:
        flash('No file part in the request', 'error')
        return redirect(request.url)

    file = request.files['train_data_file']

    if file.filename == '':
        flash('No selected file for training data', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_TEMP_FOLDER'], filename)
        file.save(temp_filepath)
        flash(f'Training data file "{filename}" uploaded successfully. Initiating retraining...', 'success')

        # Define the target path for the "master" training data file in the data folder
        # Overwrite the existing transactions.xls with the new uploaded one
        main_training_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.xls')

        try:
            # Move/overwrite the main training data file
            shutil.move(temp_filepath, main_training_data_path)
            print(f"Moved uploaded training data to: {main_training_data_path}")

            # Trigger model retraining with the new data
            if train_model.train_categorization_model(main_training_data_path):
                flash("Model successfully retrained with new data!", 'success')
            else:
                flash("Model retraining failed after data upload. Check server logs.", 'error')
        except Exception as e:
            flash(f"Error processing training data file: {e}", 'error')
            print(f"Error during training data upload/move/retrain: {e}")
        finally:
            # Ensure temp file is removed even if move failed, if it still exists
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    else:
        flash('Allowed file types for training data are csv, xls', 'error')

    return redirect(url_for('index'))

@app.route('/unsupervised_explore', methods=['POST'])
def unsupervised_explore():
    """
    Endpoint to trigger unsupervised clustering on the current training data
    and display detailed results on a new page.
    """
    num_clusters = request.form.get('num_clusters_explore', type=int, default=5)
    data_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.xls')

    print(f"Initiating unsupervised clustering for exploration with {num_clusters} clusters...")
    clustered_df, top_terms_per_cluster = unsupervised_analysis.perform_clustering(data_filepath, n_clusters=num_clusters)

    if clustered_df is not None:
        # Convert DataFrame to list of dicts for rendering in HTML
        # Ensure 'Category' is included for context in the unsupervised results page
        results_list = clustered_df[['Description', 'Amount', 'Merchant', 'Category', 'Cluster_ID']].to_dict(orient='records')
        flash("Unsupervised clustering completed successfully!", 'success')
        return render_template('unsupervised_results.html',
                               results_data=results_list, # Changed variable name to avoid conflict
                               top_terms_display=top_terms_per_cluster, # Changed variable name
                               num_clusters_explore=num_clusters) # Pass for display
    else:
        flash(f"Unsupervised clustering failed: {top_terms_per_cluster}", 'error') # top_terms_per_cluster now holds the error message
        return redirect(url_for('index'))

if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..')
    os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    os.makedirs(app.config['UPLOAD_TEMP_FOLDER'], exist_ok=True)

    # All models needed for the application to function correctly
    all_models_exist = (
        os.path.exists(os.path.join(project_root, 'models', 'xgb_categorizer_pipeline.pkl')) and
        os.path.exists(os.path.join(project_root, 'models', 'label_encoder.pkl')) and
        os.path.exists(os.path.join(project_root, 'models', 'unsupervised_tfidf_vectorizer.pkl')) and
        os.path.exists(os.path.join(project_root, 'models', 'unsupervised_svd_transformer.pkl')) and # NEW: Check SVD model
        os.path.exists(os.path.join(project_root, 'models', 'kmeans_model.pkl'))
    )

    print (f"The current project_root is {project_root}")
    initial_training_data_path = os.path.join(project_root, 'data', 'transactions.xls')

    if not all_models_exist:
        print("One or more models/encoders not found. Attempting initial training...")
        if os.path.exists(initial_training_data_path):
            # This call will now train both supervised and unsupervised models
            train_model.train_categorization_model(initial_training_data_path)
        else:
            print("Initial training data 'transactions.xls' not found. Please upload it via UI or place it in data/.")
            flash("Initial training data 'transactions.xls' not found. Please upload it via UI or place it in data/.", 'error')
    else:
        print("All models and LabelEncoder found. Ready to serve.")

    app.run(debug=True)