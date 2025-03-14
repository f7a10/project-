from flask import Flask
import os
import logging
from .config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_object=None):
    """
    Application factory function to create and configure the Flask app.

    Args:
    config_object: Configuration object to use, defaults to get_config()

    Returns:
    Configured Flask application
    """
    # Create the Flask app
    app = Flask(__name__,
                template_folder='templates',  # Explicitly set template folder
                static_folder='static')      # Explicitly set static folder

    # Load configuration
    if config_object is None:
        config_object = get_config()
    app.config.from_object(config_object)

    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize extensions
    initialize_extensions(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Log successful initialization
    logger.info(f"SmartDataHub application initialized in {app.config.get('ENV', 'development')} mode")

    return app

def initialize_extensions(app):
    """Initialize Flask extensions"""
    # In a production app, you might initialize extensions like database, login manager, etc.
    # For this app, we don't have extensions that need app-level initialization

    # However, we can make the OpenAI client available through the app if needed
    from .ai_integration import OpenRouterAI

    # Initialize OpenRouter AI with the API key from config
    ai_client = OpenRouterAI(api_key=app.config.get('OPENROUTER_API_KEY'))

    # Store it in app context for global access if needed
    app.ai_client = ai_client

    logger.info("Extensions initialized")

def register_blueprints(app):
    """Register Flask blueprints"""
    # Import and register the main blueprint
    from .routes import main
    app.register_blueprint(main)

    logger.info("Blueprints registered")

def register_error_handlers(app):
    """Register error handlers"""

    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 error: {error}")
        return {"error": "Resource not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return {"error": "Internal server error"}, 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return {"error": "An unexpected error occurred"}, 500

    logger.info("Error handlers registered")
