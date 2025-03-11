#!/usr/bin/env python
# run.py - Application entry point for SmartDataHub

import os
import sys
import logging
from dotenv import load_dotenv
from app import create_app

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    logger.info("Loading environment variables from .env file")
    load_dotenv()
else:
    logger.warning("No .env file found, using default environment variables")

def main():
    """Main function to run the Flask application."""
    try:
        # Set Flask environment to development if not already set
        if not os.environ.get('FLASK_ENV'):
            os.environ['FLASK_ENV'] = 'development'
            logger.info("Setting FLASK_ENV to development as it was not set")

        # Create the Flask application
        app = create_app()

        # Ensure template folder exists and is correct
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')
        if not os.path.exists(template_path):
            logger.error(f"Template directory does not exist at: {template_path}")
            logger.info(f"Creating template directory at: {template_path}")
            os.makedirs(template_path, exist_ok=True)

        # Ensure uploads folder exists
        upload_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(upload_path, exist_ok=True)

        # Log all registered routes for debugging
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"Route: {rule}, Methods: {rule.methods}")

        # Run the application
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')

        logger.info(f"Starting SmartDataHub application on {host}:{port}")
        logger.info(f"Debug mode: {app.debug}")
        logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")

        app.run(host=host, port=port, debug=True)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
