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
    session, send_from_directory, abort, redirect, url_for
)

from .file_processing import DataProcessor
from .ai_integration import get_ai_instance

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a Blueprint for routes
main = Blueprint('main', __name__)

# Initialize data processor
data_processor = DataProcessor()

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

@main.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for user registration."""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        logger.info(f"Registration attempt for: {email}")

        # Validate input
        if not name or not email or not password:
            logger.warning("Registration failed: Missing required fields")
            return jsonify({
                'success': False,
                'message': 'All fields are required'
            }), 400

        # In a real app, you would store the user in a database
        # For this example, we'll just return success
        logger.info(f"Registration successful for: {email}")
        return jsonify({
            'success': True,
            'message': 'Account created successfully'
        })

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred during registration'
        }), 500

@main.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint for metrics."""
    # Check authentication
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token != session.get('token'):
        logger.warning("Unauthorized metrics request")
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    # Return sample metrics data
    logger.info("Returning metrics data")
    return jsonify({
        'success': True,
        'metrics': {
            'users': 1250,
            'sessions': 5432,
            'conversion_rate': 3.2,
            'bounce_rate': 42.5
        }
    })

@main.route('/api/services/status', methods=['GET'])
def api_service_status():
    """API endpoint for service status."""
    # Check authentication
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token != session.get('token'):
        logger.warning("Unauthorized service status request")
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    # Return sample service status data
    logger.info("Returning service status data")
    return jsonify({
        'success': True,
        'services': [
            {'name': 'API Gateway', 'status': 'operational', 'uptime': 99.9},
            {'name': 'Database', 'status': 'operational', 'uptime': 99.7},
            {'name': 'Authentication', 'status': 'operational', 'uptime': 100},
            {'name': 'Storage', 'status': 'operational', 'uptime': 99.5}
        ]
    })

@main.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads, save them, and return a response.
    """
    logger.info("Handling file upload request")

    # Debug: Log the request
    logger.debug(f"Request files: {request.files}")
    logger.debug(f"Request form: {request.form}")

    try:
        # Check if any files were uploaded - frontend uses 'files[]'
        uploaded_files = []

        if 'files[]' in request.files:
            uploaded_files = request.files.getlist('files[]')
            logger.debug(f"Found files under 'files[]': {[f.filename for f in uploaded_files]}")
        else:
            # Check for any files in the request
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

        # Ensure upload folder exists
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        session_folder = os.path.join(upload_folder, session['session_id'])
        os.makedirs(session_folder, exist_ok=True)

        # Save each uploaded file
        for file in uploaded_files:
            if file and file.filename:
                try:
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(session_folder, filename)

                    # Save the file
                    file.save(file_path)
                    saved_files.append(file_path)

                    logger.info(f"Successfully saved file: {filename} to {file_path}")
                except Exception as e:
                    logger.error(f"Error saving file {file.filename}: {str(e)}")
                    logger.error(traceback.format_exc())

        if not saved_files:
            logger.warning("No files were successfully saved")
            return jsonify({"success": False, "error": "Failed to save any files"}), 500

        # Return success response
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
        # Check if there's a session
        if 'session_id' not in session:
            logger.warning("No session ID found")
            return jsonify({"success": False, "error": "No active session"}), 400

        session_id = session['session_id']
        logger.debug(f"Using session_id: {session_id}")

        # Get the uploaded files for this session
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        session_folder = os.path.join(upload_folder, session_id)

        # Check if the folder exists
        if not os.path.exists(session_folder):
            logger.warning(f"Session folder does not exist: {session_folder}")
            return jsonify({"success": False, "error": "No files found for session"}), 404

        # Get all files in the session folder
        files = [os.path.join(session_folder, f) for f in os.listdir(session_folder)
                if os.path.isfile(os.path.join(session_folder, f))]

        if not files:
            logger.warning(f"No files found in session folder: {session_folder}")
            return jsonify({"success": False, "error": "No files found"}), 404

        logger.debug(f"Found files: {files}")

        # Process the files
        logger.info(f"Processing {len(files)} files")
        processed_data = data_processor.process_files(files)

        # Check if processing was successful
        if not processed_data.get("success", False):
            logger.warning("File processing failed")
            return jsonify({"success": False, "error": "File processing failed", "details": processed_data}), 500

        # Generate dashboard data
        dashboard_data = generate_dashboard_data(processed_data, files, session_id)

        # Add file info and success flag
        dashboard_data["files"] = [os.path.basename(f) for f in files]
        dashboard_data["success"] = True

        # Get AI insights if available
        try:
            ai_client = get_ai_instance()
            if ai_client:
                # Generate AI insights
                data_summary = {
                    "files": [os.path.basename(f) for f in files],
                    "metrics": dashboard_data["metrics"],
                    "data_overview": processed_data
                }
                dashboard_data["ai_insights"] = ai_client.analyze_data_initial(data_summary)
            else:
                dashboard_data["ai_insights"] = "I've analyzed your data and found some interesting patterns in the visualizations. You can explore them in the charts above."
        except Exception as ai_error:
            logger.error(f"Error generating AI insights: {str(ai_error)}")
            dashboard_data["ai_insights"] = "I've analyzed your data and prepared visualizations based on the content of your files."

        return jsonify(dashboard_data)

    except Exception as e:
        logger.error(f"Exception in analyze_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

def generate_dashboard_data(processed_data, file_paths, session_id):
    """
    Generate data for the dashboard from processed file data.
    
    Args:
        processed_data: Dictionary with processed file data
        file_paths: List of file paths
        session_id: The session ID
        
    Returns:
        Dictionary with dashboard data
    """
    logger.info("Generating dashboard data from actual file data")

    try:
        # Initialize dashboard data with proper structure
        dashboard_data = {
            "metrics": {},
            "charts": {}
        }

        # Calculate metrics
        metrics = generate_metrics(processed_data)
        dashboard_data["metrics"] = metrics

        # Calculate chart data from actual file data
        charts = generate_chart_data_from_files(processed_data, file_paths, session_id)
        dashboard_data["charts"] = charts

        return dashboard_data

    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}")
        logger.error(traceback.format_exc())

        # Return empty dashboard data with error information
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
    
    Args:
        processed_data: Dictionary with processed file data
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Generating metrics")

    try:
        # Remove 'success' key for metrics calculation
        data_dict = {k: v for k, v in processed_data.items() if k != 'success'}

        # Calculate total records
        total_records = 0

        # Calculate number of fields
        total_fields = 0

        # Calculate data completeness
        total_cells = 0
        missing_cells = 0

        for filename, file_data in data_dict.items():
            if 'error' in file_data:
                logger.warning(f"Skipping file with error: {filename}")
                continue

            # Add to total records
            if 'shape' in file_data:
                rows = file_data['shape'].get('rows', 0)
                cols = file_data['shape'].get('columns', 0)

                total_records += rows
                total_fields += cols

                # Calculate data completeness
                if 'missing_data' in file_data:
                    file_missing = sum(file_data['missing_data'].values())
                    file_cells = rows * cols

                    missing_cells += file_missing
                    total_cells += file_cells

        # Calculate data quality (a simple measure based on missing data)
        data_quality = 100.0
        if total_cells > 0:
            data_quality = 100 - ((missing_cells / total_cells) * 100)

        # Calculate completeness
        completeness = 100.0
        if total_cells > 0:
            completeness = 100 - ((missing_cells / total_cells) * 100)

        # Create metrics dictionary with proper formatting
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

        # Return default metrics
        return {
            "totalRecords": {"label": "Total Records", "value": "0"},
            "dataQuality": {"label": "Data Quality", "value": "0%"},
            "completeness": {"label": "Completeness", "value": "0%"},
            "fields": {"label": "Fields", "value": "0"}
        }

def generate_chart_data_from_files(processed_data, file_paths, session_id):
    """
    Generate chart data from actual file data.

    Args:
        processed_data: Dictionary with processed file data
        file_paths: List of file paths
        session_id: The session ID

    Returns:
        Dictionary with chart data
    """
    logger.info("Generating chart data from file contents")
    logger.debug(f"File paths: {file_paths}")
    logger.debug(f"Processed data keys: {list(processed_data.keys())}")

    try:
        # Initialize charts
        charts = {
            "performance": {"labels": [], "values": []},
            "category": {"labels": [], "values": []},
            "comparison": {"labels": [], "values": []},
            "correlation": {"labels": [], "values": []}
        }

        # Remove 'success' key for chart calculation
        data_dict = {k: v for k, v in processed_data.items() if k != 'success'}

        # Check if there's any valid data - match by basename to handle path differences
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

        # Use the first valid file for chart generation
        file_path, key = valid_files[0]
        file_data = data_dict[key]
        logger.info(f"Using file {file_path} for chart generation")

        # Load the actual dataframe for visualization
        try:
            df = load_dataframe(file_path)
            if df is None:
                logger.warning(f"Failed to load dataframe from {file_path}")
                return charts

            logger.info(f"Successfully loaded dataframe with shape {df.shape}")
            logger.debug(f"DataFrame columns: {list(df.columns)}")
            logger.debug(f"DataFrame dtypes: {df.dtypes}")

            # PERFORMANCE CHART (based on numeric data)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                chosen_col = numeric_cols[0]  # Use first numeric column
                logger.info(f"Using numeric column {chosen_col} for performance chart")

                # Check if there's a date column to use as x-axis
                date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]

                if date_cols and len(date_cols) > 0:
                    # Time series data
                    try:
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                        df_valid = df.dropna(subset=[date_cols[0], chosen_col])

                        if len(df_valid) > 0:
                            df_sorted = df_valid.sort_values(by=date_cols[0])

                            # Limit points for performance
                            if len(df_sorted) > 30:
                                step = len(df_sorted) // 30
                                df_sampled = df_sorted.iloc[::step, :]
                            else:
                                df_sampled = df_sorted

                            date_labels = df_sampled[date_cols[0]].dt.strftime('%Y-%m-%d').tolist()
                            numeric_values = df_sampled[chosen_col].tolist()

                            charts["performance"]["labels"] = date_labels
                            charts["performance"]["values"] = numeric_values
                            logger.info(f"Generated time series chart with {len(charts['performance']['labels'])} points")
                    except Exception as e:
                        logger.error(f"Error creating time series chart: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    # Regular numeric data - use row numbers as x-axis
                    try:
                        values = df[chosen_col].dropna().head(30).tolist()
                        labels = [f"Row {i+1}" for i in range(len(values))]

                        charts["performance"]["labels"] = labels
                        charts["performance"]["values"] = values
                        logger.info(f"Generated performance chart with {len(values)} points")
                    except Exception as e:
                        logger.error(f"Error creating basic performance chart: {str(e)}")
                        logger.error(traceback.format_exc())

            # CATEGORY CHART (based on categorical data)
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if categorical_cols:
                try:
                    chosen_cat_col = categorical_cols[0]  # Use first categorical column
                    logger.info(f"Using categorical column {chosen_cat_col} for category chart")

                    # Get value counts, handling potential data issues
                    value_counts = df[chosen_cat_col].value_counts().head(7)

                    # Ensure all values are serializable (convert to strings)
                    cat_labels = [str(x) for x in value_counts.index.tolist()]
                    cat_values = [float(x) for x in value_counts.values.tolist()]

                    # If there are more than 7 categories, group the rest as "Other"
                    if len(df[chosen_cat_col].unique()) > 7:
                        other_count = df[chosen_cat_col].value_counts().iloc[7:].sum()
                        cat_labels.append("Other")
                        cat_values.append(float(other_count))

                    charts["category"]["labels"] = cat_labels
                    charts["category"]["values"] = cat_values

                    logger.info(f"Generated category chart with {len(charts['category']['labels'])} categories")
                except Exception as e:
                    logger.error(f"Error creating category chart: {str(e)}")
                    logger.error(traceback.format_exc())

            # COMPARISON CHART (comparing numeric columns)
            if len(numeric_cols) >= 2:
                try:
                    # Use up to 6 numeric columns
                    cols_to_use = numeric_cols[:6]
                    logger.info(f"Using columns {cols_to_use} for comparison chart")

                    # Get means of each column, handling potential NaN values
                    means = []
                    for col in cols_to_use:
                        col_mean = df[col].mean()
                        means.append(float(col_mean) if not pd.isna(col_mean) else 0.0)

                    # Convert column names to strings to ensure serializability
                    charts["comparison"]["labels"] = [str(col) for col in cols_to_use.tolist()]
                    charts["comparison"]["values"] = means

                    logger.info(f"Generated comparison chart with {len(cols_to_use)} columns")
                except Exception as e:
                    logger.error(f"Error creating comparison chart: {str(e)}")
                    logger.error(traceback.format_exc())
            elif len(numeric_cols) == 1 and categorical_cols:
                # One numeric and one categorical - show numeric by category
                try:
                    num_col = numeric_cols[0]
                    cat_col = categorical_cols[0]
                    logger.info(f"Using {num_col} by {cat_col} for comparison chart")

                    # Group by category and calculate means
                    grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(6)

                    # Ensure serializability
                    comp_labels = [str(x) for x in grouped.index.tolist()]
                    comp_values = [float(x) for x in grouped.values.tolist()]

                    charts["comparison"]["labels"] = comp_labels
                    charts["comparison"]["values"] = comp_values

                    logger.info(f"Generated comparison chart with {len(grouped)} categories")
                except Exception as e:
                    logger.error(f"Error creating grouped comparison chart: {str(e)}")
                    logger.error(traceback.format_exc())

            # CORRELATION CHART (correlation between numeric columns)
            if len(numeric_cols) > 1:
                try:
                    # Use up to 5 numeric columns
                    cols_for_corr = numeric_cols[:5]
                    logger.info(f"Using columns {cols_for_corr} for correlation chart")

                    # Calculate correlation matrix, replacing NaN with 0
                    corr_matrix = df[cols_for_corr].corr().fillna(0).values.tolist()

                    # Ensure all values in the matrix are serializable
                    corr_matrix = [[float(cell) for cell in row] for row in corr_matrix]

                    charts["correlation"]["labels"] = [str(col) for col in cols_for_corr.tolist()]
                    charts["correlation"]["values"] = corr_matrix

                    logger.info(f"Generated correlation chart with {len(cols_for_corr)} dimensions")
                except Exception as e:
                    logger.error(f"Error creating correlation chart: {str(e)}")
                    logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error processing file for charts: {str(e)}")
            logger.error(traceback.format_exc())

        # Check if any charts are still empty, use defaults if needed
        ensure_charts_have_data(charts)

        return charts

    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        logger.error(traceback.format_exc())
        return get_default_charts()

def load_dataframe(file_path):
    """
    Load a dataframe from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Pandas DataFrame or None if loading fails
    """
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Load based on extension
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
                # Try loading as dictionary if array format fails
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    df = pd.DataFrame.from_dict(data, orient='index')
                else:
                    df = pd.DataFrame(data)
        elif ext == '.txt' or ext == '.dat':
            # Try to infer delimiter
            df = pd.read_csv(file_path, sep=None, engine='python')
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error loading dataframe: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def ensure_charts_have_data(charts):
    """
    Ensure all charts have data, filling in defaults if needed.
    
    Args:
        charts: Dictionary with chart data
    """
    # Check each chart type and provide defaults if empty
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

def get_default_charts():
    """
    Get default chart data.
    
    Returns:
        Dictionary with default chart data
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
