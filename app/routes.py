import os
import json
import uuid
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, render_template, request, jsonify, current_app,
    session, send_from_directory, abort, redirect,url_for
)

from .file_processing import DataProcessor
from .ai_integration import get_ai_instance

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a Blueprint for routes
main = Blueprint('main', __name__)

# Initialize data processor
data_processor = DataProcessor()

# Helper function to load a DataFrame dynamically
def load_dataframe(file_path):
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.csv':
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            try:
                df = pd.read_json(file_path)
            except ValueError:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    df = pd.DataFrame.from_dict(data, orient='index')
                else:
                    df = pd.DataFrame(data)
        elif ext in ['.txt', '.dat']:
            df = pd.read_csv(file_path, sep=None, engine='python')
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
        logger.info(f"Loaded dataframe from {file_path} with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@main.route('/')
def index():
    """Render the a.html page first."""
    logger.info("Rendering a.html page")
    return render_template('a.html')

@main.route('/login')
def login():
    """Render the login page."""
    logger.info("Rendering login page")
    return render_template('login.html')

@main.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """Handle login and render the main dashboard page."""
    logger.info(f"Dashboard route called with method: {request.method}")
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        logger.debug(f"Login attempt with username: {username}")

        # Here you would typically validate the credentials
        # For this example, we'll just check if they're not empty
        if username and password:
            # You could set a session variable to track login status
            session['logged_in'] = True
            session['username'] = username
            logger.info(f"User {username} logged in successfully")
            return render_template('index.html', username=username)
        else:
            # If validation fails, go back to login with an error
            logger.warning("Invalid login credentials")
            return render_template('login.html', error="Invalid credentials")

    # If it's a GET request, check if user is already logged in
    if session.get('logged_in'):
        logger.info(f"User {session.get('username')} already logged in")
        return render_template('index.html', username=session.get('username'))
    else:
        # Redirect to login if not logged in
        logger.info("User not logged in, redirecting to login page")
        return redirect(url_for('main.login'))

@main.route('/logout')
def logout():
    """Log out the user and redirect to the login page."""
    logger.info(f"Logging out user: {session.get('username')}")
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('main.login'))

@main.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for login."""
    try:
        data = request.get_json()
        username = data.get('username') or data.get('email')
        password = data.get('password')
        
        logger.info(f"API login attempt for user: {username}")

        # Here you would validate credentials against your database
        # For this example, we'll use a simple check
        if username and password:  # Replace with actual validation
            # Generate a simple token (in production, use a proper JWT)
            import secrets
            token = secrets.token_hex(16)

            # In a real app, you'd store this token in a database
            # For this example, we'll store in session
            session['token'] = token
            session['logged_in'] = True
            session['user_id'] = 1  # Example user ID
            session['username'] = username.split('@')[0] if '@' in username else username
            
            logger.info(f"API login successful for user: {session['username']}")

            return jsonify({
                'success': True,
                'token': token,
                'user_id': 1,
                'username': session['username']
            })
        else:
            logger.warning("API login failed: Invalid credentials")
            return jsonify({
                'success': False,
                'message': 'Invalid credentials'
            }), 401

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during login'
        }), 500

@main.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads, save them, and return a response.
    """
    logger.info("Handling file upload request")
    logger.debug(f"Request files: {request.files}")

    try:
        # Check if any files were uploaded
        uploaded_files = []
        if 'files[]' in request.files:
            uploaded_files = request.files.getlist('files[]')
            logger.debug(f"Found files under 'files[]': {[f.filename for f in uploaded_files]}")
        else:
            for key in request.files:
                if request.files.getlist(key):
                    uploaded_files = request.files.getlist(key)
                    logger.debug(f"Found files under '{key}': {[f.filename for f in uploaded_files]}")
                    break

        if not uploaded_files or uploaded_files[0].filename == '':
            logger.warning("No files found in request")
            return jsonify({"success": False, "error": "No files selected"}), 400

        # Get or create a session ID for this upload
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            logger.info(f"Created new session_id: {session['session_id']}")

        # List to store saved files
        saved_files = []
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        session_folder = os.path.join(upload_folder, session['session_id'])
        os.makedirs(session_folder, exist_ok=True)

        for file in uploaded_files:
            if file and file.filename:
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(session_folder, filename)
                    file.save(file_path)
                    saved_files.append(file_path)
                    logger.info(f"Successfully saved file: {filename} to {file_path}")
                except Exception as e:
                    logger.error(f"Error saving file {file.filename}: {str(e)}")
                    logger.error(traceback.format_exc())

        if not saved_files:
            logger.warning("No files were successfully saved")
            return jsonify({"success": False, "error": "Failed to save any files"}), 500

        return jsonify({
            "success": True,
            "message": f"Successfully uploaded {len(saved_files)} file(s)",
            "files": [os.path.basename(f) for f in saved_files],
            "session_id": session['session_id']
        })

    except Exception as e:
        logger.error(f"Exception in upload_files: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@main.route('/analyze', methods=['GET'])
def analyze_data():
    """
    Analyze the uploaded files and generate visualizations.
    """
    logger.info("Handling analyze data request")
    try:
        if 'session_id' not in session:
            logger.warning("No session ID found")
            return jsonify({"success": False, "error": "No active session"}), 400

        session_id = session['session_id']
        logger.debug(f"Using session_id: {session_id}")
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        session_folder = os.path.join(upload_folder, session_id)

        if not os.path.exists(session_folder):
            logger.warning(f"Session folder does not exist: {session_folder}")
            return jsonify({"success": False, "error": "No files found for session"}), 404

        files = [os.path.join(session_folder, f) for f in os.listdir(session_folder)
                if os.path.isfile(os.path.join(session_folder, f))]

        if not files:
            logger.warning(f"No files found in session folder: {session_folder}")
            return jsonify({"success": False, "error": "No files found"}), 404

        logger.debug(f"Found files: {files}")
        logger.info(f"Processing {len(files)} files")
        processed_data = process_files_directly(files)

        if not processed_data.get("success", False):
            logger.warning("File processing failed")
            return jsonify({"success": False, "error": "File processing failed", "details": processed_data}), 500

        dashboard_data = generate_dashboard_data_from_files(processed_data, files, session_id)
        dashboard_data["files"] = [os.path.basename(f) for f in files]
        dashboard_data["success"] = True

        try:
            ai_client = get_ai_instance()
            if ai_client:
                data_summary = {
                    "files": [os.path.basename(f) for f in files],
                    "metrics": dashboard_data.get("metrics", {}),
                    "data_overview": processed_data
                }
                ai_insights = ai_client.analyze_data_initial(data_summary)
                dashboard_data["ai_insights"] = ai_insights
            else:
                dashboard_data["ai_insights"] = "AI analysis is not available at the moment."
        except Exception as ai_error:
            logger.error(f"Error generating AI insights: {str(ai_error)}")
            dashboard_data["ai_insights"] = "Unable to generate AI insights at this time."

        return jsonify(dashboard_data)

    except Exception as e:
        logger.error(f"Exception in analyze_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

def process_files_directly(file_paths):
    """
    Direct file processing to extract summary data from files.
    """
    results = {"success": False}
    file_data = {}
    try:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            try:
                if ext == '.csv':
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif ext == '.json':
                    df = pd.read_json(file_path)
                else:
                    df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')

                logger.info(f"Successfully loaded file {file_name} with shape {df.shape}")

                summary = {
                    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "missing_data": {col: int(df[col].isnull().sum()) for col in df.columns},
                    "numeric_columns": {},
                    "categorical_columns": {}
                }

                for col in df.select_dtypes(include=['number']).columns:
                    summary["numeric_columns"][str(col)] = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0
                    }

                for col in df.select_dtypes(exclude=['number']).columns:
                    value_counts = df[col].value_counts().head(10).to_dict()
                    summary["categorical_columns"][str(col)] = {str(k): int(v) for k, v in value_counts.items()}

                file_data[file_name] = summary

            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
                logger.error(traceback.format_exc())
                file_data[file_name] = {"error": str(e)}

        if file_data:
            results = file_data
            results["success"] = True

        return results

    except Exception as e:
        logger.error(f"Exception in process_files_directly: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def generate_dashboard_data_from_files(processed_data, file_paths, session_id):
    """
    Generate dashboard data from processed files.
    """
    logger.info("Generating dashboard data from actual file data")
    try:
        dashboard_data = {
            "metrics": {},
            "charts": {}
        }
        metrics = generate_metrics(processed_data)
        dashboard_data["metrics"] = metrics
        charts = generate_chart_data_from_files(processed_data, file_paths, session_id)
        dashboard_data["charts"] = charts
        return dashboard_data

    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "metrics": {
                "totalRecords": {"label": "Total Records", "value": "Error"},
                "dataQuality": {"label": "Data Quality", "value": "Error"},
                "completeness": {"label": "Completeness", "value": "Error"},
                "fields": {"label": "Fields", "value": "Error"}
            },
            "charts": {},
            "error": str(e)
        }

def generate_metrics(processed_data):
    """
    Generate metrics from processed data.
    """
    logger.info("Generating metrics")
    try:
        data_dict = {k: v for k, v in processed_data.items() if k != 'success'}
        total_records = 0
        total_fields = 0
        total_cells = 0
        missing_cells = 0

        for filename, file_data in data_dict.items():
            if 'error' in file_data:
                logger.warning(f"Skipping file with error: {filename}")
                continue
            if 'shape' in file_data:
                rows = file_data['shape'].get('rows', 0)
                cols = file_data['shape'].get('columns', 0)
                total_records += rows
                total_fields += cols
            if 'missing_data' in file_data:
                file_missing = sum(file_data['missing_data'].values())
                file_cells = rows * cols
                missing_cells += file_missing
                total_cells += file_cells

        data_quality = 100.0 if total_cells == 0 else 100 - ((missing_cells / total_cells) * 100)
        completeness = 100.0 if total_cells == 0 else 100 - ((missing_cells / total_cells) * 100)

        metrics = {
            "totalRecords": {"label": "Total Records", "value": f"{total_records:,}"},
            "dataQuality": {"label": "Data Quality", "value": f"{data_quality:.1f}%"},
            "completeness": {"label": "Completeness", "value": f"{completeness:.1f}%"},
            "fields": {"label": "Fields", "value": f"{total_fields}"}
        }
        logger.debug(f"Generated metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "totalRecords": {"label": "Total Records", "value": "0"},
            "dataQuality": {"label": "Data Quality", "value": "0%"},
            "completeness": {"label": "Completeness", "value": "0%"},
            "fields": {"label": "Fields", "value": "0"}
        }

def generate_chart_data_from_files(processed_data, file_paths, session_id):
    """
    Generate chart data from actual file data.
    """
    logger.info("Generating chart data from file contents")
    logger.debug(f"File paths: {file_paths}")
    logger.debug(f"Processed data keys: {list(processed_data.keys())}")

    try:
        charts = {
            "performance": {"labels": [], "values": []},
            "category": {"labels": [], "values": []},
            "comparison": {"labels": [], "values": []},
            "correlation": {"labels": [], "values": []},
            "radar": {"labels": [], "datasets": []}
        }

        data_dict = {k: v for k, v in processed_data.items() if k != 'success'}
        valid_files = []
        for f in file_paths:
            base_name = os.path.basename(f)
            if base_name in data_dict and 'error' not in data_dict[base_name]:
                valid_files.append((f, base_name))

        if not valid_files:
            logger.warning("No valid files for chart generation")
            logger.debug(f"Available keys in processed_data: {list(data_dict.keys())}")
            logger.debug(f"File basenames: {[os.path.basename(f) for f in file_paths]}")
            return charts

        file_path, key = valid_files[0]
        logger.info(f"Using file {file_path} for chart generation")
        try:
            df = load_dataframe(file_path)
            if df is None:
                logger.warning(f"Failed to load dataframe from {file_path}")
                return charts
            logger.info(f"Successfully loaded dataframe with shape {df.shape}")
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame dtypes: {df.dtypes}")
        except Exception as e:
            logger.error(f"Error loading dataframe: {str(e)}")
            logger.error(traceback.format_exc())
            return charts

        # PERFORMANCE CHART
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            chosen_col = numeric_cols[0]
            logger.info(f"Using numeric column {chosen_col} for performance chart")
            date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]
            if date_cols:
                try:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    df_valid = df.dropna(subset=[date_cols[0], chosen_col])
                    if len(df_valid) > 0:
                        df_sorted = df_valid.sort_values(by=date_cols[0])
                        if len(df_sorted) > 30:
                            step = len(df_sorted) // 30
                            df_sampled = df_sorted.iloc[::step, :]
                        else:
                            df_sampled = df_sorted
                        date_labels = df_sampled[date_cols[0]].dt.strftime('%Y-%m-%d').tolist()
                        numeric_values = df_sampled[chosen_col].tolist()
                        charts["performance"]["labels"] = date_labels
                        charts["performance"]["values"] = numeric_values
                        logger.info(f"Generated time series chart with {len(date_labels)} points")
                except Exception as e:
                    logger.error(f"Error creating time series chart: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                try:
                    values = df[chosen_col].dropna().head(30).tolist()
                    labels = [f"Row {i+1}" for i in range(len(values))]
                    charts["performance"]["labels"] = labels
                    charts["performance"]["values"] = values
                    logger.info(f"Generated performance chart with {len(values)} points")
                except Exception as e:
                    logger.error(f"Error creating basic performance chart: {str(e)}")
                    logger.error(traceback.format_exc())

        # CATEGORY CHART
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if categorical_cols:
            try:
                chosen_cat_col = categorical_cols[0]
                logger.info(f"Using categorical column {chosen_cat_col} for category chart")
                value_counts = df[chosen_cat_col].value_counts().head(7)
                cat_labels = [str(x) for x in value_counts.index.tolist()]
                cat_values = [float(x) for x in value_counts.values.tolist()]
                if len(df[chosen_cat_col].unique()) > 7:
                    other_count = df[chosen_cat_col].value_counts().iloc[7:].sum()
                    cat_labels.append("Other")
                    cat_values.append(float(other_count))
                charts["category"]["labels"] = cat_labels
                charts["category"]["values"] = cat_values
                logger.info(f"Generated category chart with {len(cat_labels)} categories")
            except Exception as e:
                logger.error(f"Error creating category chart: {str(e)}")
                logger.error(traceback.format_exc())

        # COMPARISON CHART (only if at least 2 numeric columns exist)
        if len(numeric_cols) >= 2:
            try:
                cols_to_use = numeric_cols[:6]
                logger.info(f"Using columns {cols_to_use} for comparison chart")
                means = []
                for col in cols_to_use:
                    col_mean = df[col].mean()
                    means.append(float(col_mean) if not pd.isna(col_mean) else 0.0)
                charts["comparison"]["labels"] = [str(col) for col in cols_to_use.tolist()]
                charts["comparison"]["values"] = means
                logger.info(f"Generated comparison chart with {len(cols_to_use)} columns")
            except Exception as e:
                logger.error(f"Error creating comparison chart: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            charts["comparison"] = {"labels": [], "values": []}

        # CORRELATION CHART
        if len(numeric_cols) > 1:
            try:
                cols_for_corr = numeric_cols[:5]
                logger.info(f"Using columns {cols_for_corr} for correlation chart")
                corr_matrix = df[cols_for_corr].corr().fillna(0).values.tolist()
                corr_matrix = [[float(cell) for cell in row] for row in corr_matrix]
                charts["correlation"]["labels"] = [str(col) for col in cols_for_corr.tolist()]
                charts["correlation"]["values"] = corr_matrix
                logger.info(f"Generated correlation chart with {len(cols_for_corr)} dimensions")
            except Exception as e:
                logger.error(f"Error creating correlation chart: {str(e)}")
                logger.error(traceback.format_exc())

        # RADAR CHART (Spider Chart for Data Dimensions)
        if len(numeric_cols) > 0:
            try:
                radar_values = []
                for col in numeric_cols:
                    value = df[col].mean()
                    radar_values.append(float(value) if not pd.isna(value) else 0.0)
                charts["radar"]["labels"] = [str(col) for col in numeric_cols.tolist()]
                charts["radar"]["datasets"] = [{
                    "label": "Data Dimensions",
                    "data": radar_values
                }]
                logger.info(f"Generated radar chart with {len(numeric_cols)} dimensions")
            except Exception as e:
                logger.error(f"Error creating radar chart: {str(e)}")
                logger.error(traceback.format_exc())
                charts["radar"]["labels"] = []
                charts["radar"]["datasets"] = []
        else:
            charts["radar"]["labels"] = []
            charts["radar"]["datasets"] = []

        ensure_charts_have_data(charts)
        return charts

    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        logger.error(traceback.format_exc())
        return get_default_charts()

def ensure_charts_have_data(charts):
    """
    Ensure all charts have data, filling in defaults if needed.
    """
    if not charts["performance"]["labels"]:
        charts["performance"]["labels"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
        charts["performance"]["values"] = [65, 59, 80, 81, 56, 55, 40]

    if not charts["category"]["labels"]:
        charts["category"]["labels"] = ["Category A", "Category B", "Category C", "Category D"]
        charts["category"]["values"] = [30, 50, 20, 10]

    if not charts["comparison"]["labels"]:
        charts["comparison"]["labels"] = ["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"]
        charts["comparison"]["values"] = [12, 19, 3, 5, 2]

    if not charts["correlation"]["labels"]:
        charts["correlation"]["labels"] = ["A", "B", "C", "D", "E"]
        charts["correlation"]["values"] = [
            [1, 0.8, 0.6, 0.2, 0.1],
            [0.8, 1, 0.7, 0.3, 0.2],
            [0.6, 0.7, 1, 0.5, 0.4],
            [0.2, 0.3, 0.5, 1, 0.9],
            [0.1, 0.2, 0.4, 0.9, 1]
        ]

    if not charts["radar"].get("datasets") or not charts["radar"]["datasets"]:
        charts["radar"]["labels"] = ["Metric 1", "Metric 2", "Metric 3", "Metric 4"]
        charts["radar"]["datasets"] = [{
            "label": "Data Dimensions",
            "data": [0, 0, 0, 0]
        }]

def get_default_charts():
    """
    Get default chart data.
    """
    return {
        "performance": {
            "labels": ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
            "values": [65, 59, 80, 81, 56]
        },
        "category": {
            "labels": ["Category A", "Category B", "Category C", "Category D"],
            "values": [30, 50, 20, 10]
        },
        "comparison": {
            "labels": ["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"],
            "values": [12, 19, 3, 5, 2]
        },
        "correlation": {
            "labels": ["A", "B", "C", "D", "E"],
            "values": [
                [1, 0.8, 0.6, 0.2, 0.1],
                [0.8, 1, 0.7, 0.3, 0.2],
                [0.6, 0.7, 1, 0.5, 0.4],
                [0.2, 0.3, 0.5, 1, 0.9],
                [0.1, 0.2, 0.4, 0.9, 1]
            ]
        },
        "radar": {
            "labels": ["Metric 1", "Metric 2", "Metric 3", "Metric 4"],
            "datasets": [{
                "label": "Data Dimensions",
                "data": [0, 0, 0, 0]
            }]
        }
    }

@main.route('/ask', methods=['POST'])
def ask_question():
    """
    Answer a question about the data using AI.
    """
    logger.info("Handling question request")
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logger.warning("No question in the request")
            return jsonify({"success": False, "error": "No question provided"}), 400

        question = data['question']
        logger.info(f"Received question: {question}")

        if 'session_id' not in session:
            logger.warning("No session ID found")
            return jsonify({"success": False, "error": "No active session"}), 400

        session_id = session['session_id']
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        session_folder = os.path.join(upload_folder, session_id)

        if not os.path.exists(session_folder):
            logger.warning(f"Session folder does not exist: {session_folder}")
            return jsonify({
                "answer": "I don't have any data to analyze. Please upload some files first."
            })

        files = [os.path.join(session_folder, f) for f in os.listdir(session_folder)
                 if os.path.isfile(os.path.join(session_folder, f))]

        if not files:
            logger.warning(f"No files found in session folder: {session_folder}")
            return jsonify({
                "answer": "I don't have any data to analyze. Please upload some files first."
            })

        processed_data = process_files_directly(files)
        ai_client = get_ai_instance()
        if not ai_client:
            logger.warning("AI client is not available")
            return jsonify({
                "answer": "I'm analyzing your data. It appears to contain " +
                          f"{processed_data.get('totalRecords', 'some')} records across " +
                          f"{len(files)} files. What specific insights are you looking for?"
            })

        data_context = {
            "files": [os.path.basename(f) for f in files],
            "data_overview": processed_data
        }
        answer = ai_client.answer_question(question, data_context)
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Exception in ask_question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "answer": f"I'm sorry, I encountered an error while processing your question: {str(e)}"
        })

@main.route('/file/<path:filename>')
def get_file(filename):
    """
    Serve an uploaded file.
    """
    logger.info(f"Request to get file: {filename}")
    try:
        if 'session_id' not in session:
            logger.warning("No session ID found")
            abort(403)

        session_id = session['session_id']
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], session_id, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            abort(404)

        return send_from_directory(
            os.path.join(current_app.config['UPLOAD_FOLDER'], session_id),
            filename
        )
    except Exception as e:
        logger.error(f"Exception in get_file: {str(e)}")
        logger.error(traceback.format_exc())
        abort(500)

@main.route('/delete/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    """
    Delete an uploaded file.
    """
    logger.info(f"Request to delete file: {filename}")
    try:
        if 'session_id' not in session:
            logger.warning("No session ID found")
            return jsonify({"success": False, "error": "No active session"}), 400

        session_id = session['session_id']
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], session_id, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({"success": False, "error": "File not found"}), 404

        os.remove(file_path)
        logger.info(f"Successfully deleted file: {filename}")
        return jsonify({
            "success": True,
            "message": f"File {filename} deleted successfully"
        })

    except Exception as e:
        logger.error(f"Exception in delete_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
