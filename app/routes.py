import os
import uuid
import logging
import traceback
from flask import Blueprint, render_template, request, jsonify, current_app, session, send_from_directory
from werkzeug.utils import secure_filename
from .file_processing import DataProcessor
from .visualization import DataVisualizer
from .ai_integration import OpenRouterAI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create blueprint
main = Blueprint('main', __name__)

# Initialize AI client
ai_client = OpenRouterAI()

@main.route('/')
def index():
    """Render the main application page."""
    # Make sure you have the template file named exactly 'interactive_dashboard.html'
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for data analysis."""
    try:
        logger.info("Upload route called")
        
        # Check if files are in the request
        if 'files[]' not in request.files and 'file' not in request.files:
            logger.error("No files in request")
            return jsonify({'success': False, 'error': 'No files in request'}), 400
        
        # Get files from request
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        
        if not files or not files[0].filename:
            logger.error("No file selected")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Create upload folder if it doesn't exist
        upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate a session ID for this analysis
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Create session folder
        session_folder = os.path.join(upload_dir, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save files
        file_paths = []
        file_names = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(session_folder, filename)
                file.save(file_path)
                file_paths.append(file_path)
                file_names.append(filename)
        
        # Store file paths in session
        session['file_paths'] = file_paths
        session['file_names'] = file_names
        
        # Process files to get summary
        data_processor = DataProcessor()
        summary = data_processor.process_files(file_paths)
        
        # Store summary in session
        session['data_summary'] = summary
        
        # Return success response
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(file_paths)} files',
            'files': file_names,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/analyze', methods=['GET'])
def analyze_data():
    """Analyze uploaded files and return visualization data."""
    try:
        # Get file paths from session
        file_paths = session.get('file_paths', [])
        
        if not file_paths:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        # Process files
        data_processor = DataProcessor()
        summary = data_processor.process_files(file_paths)
        
        # Generate visualizations
        visualizer = DataVisualizer()
        visualization_data = visualizer.generate_visualizations(file_paths)
        
        # Get AI insights
        data_summary = str(summary)
        ai_insights = ai_client.analyze_data_initial(data_summary)
        
        # Combine all data
        response_data = {
            'success': True,
            'metrics': {
                'metric1': {'label': 'Total Records', 'value': summary.get('total_records', 'N/A')},
                'metric2': {'label': 'Data Quality', 'value': summary.get('data_quality', 'N/A')},
                'metric3': {'label': 'Completeness', 'value': summary.get('completeness', 'N/A')},
                'metric4': {'label': 'Fields', 'value': summary.get('fields_count', 'N/A')}
            },
            'charts': visualization_data,
            'resource_usage': 72.2,  # Placeholder value
            'ai_insights': ai_insights
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions about the data."""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'}), 400
        
        # Get data context from session
        file_paths = session.get('file_paths', [])
        data_summary = session.get('data_summary', {})
        
        if not file_paths or not data_summary:
            return jsonify({'success': False, 'error': 'No data available for analysis'}), 400
        
        # Convert data summary to string
        data_context = str(data_summary)
        
        # Get answer from AI
        answer = ai_client.answer_question(question, data_context)
        
        return jsonify({'success': True, 'answer': answer})
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/visualizations/<viz_type>', methods=['GET'])
def get_visualization(viz_type):
    """Get a specific visualization."""
    try:
        # Get file paths from session
        file_paths = session.get('file_paths', [])
        
        if not file_paths:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        # Generate visualization
        visualizer = DataVisualizer()
        
        if viz_type == 'line':
            visualization = visualizer.generate_line_chart(file_paths[0])
        elif viz_type == 'pie':
            visualization = visualizer.generate_pie_chart(file_paths[0])
        elif viz_type == 'bar':
            visualization = visualizer.generate_bar_chart(file_paths[0])
        else:
            return jsonify({'success': False, 'error': 'Invalid visualization type'}), 400
        
        return jsonify({'success': True, 'visualization': visualization})
        
    except Exception as e:
        logger.error(f"Error in get_visualization: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': 'No session found'}), 404
    
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], session_id)
    return send_from_directory(upload_dir, filename)

@main.route('/metrics', methods=['GET'])
def get_metrics():
    """Get metrics for the uploaded data."""
    try:
        # Get file paths from session
        file_paths = session.get('file_paths', [])
        
        if not file_paths:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        # Process files to get metrics
        data_processor = DataProcessor()
        metrics = data_processor.get_metrics(file_paths)
        
        return jsonify({'success': True, 'metrics': metrics})
        
    except Exception as e:
        logger.error(f"Error in get_metrics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500