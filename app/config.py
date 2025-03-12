import os
import logging
from dotenv import load_dotenv
import secrets

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    # Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DEBUG = False
    TESTING = False

    # Application config
    TEMPLATES_FOLDER = os.path.join('app', 'templates')
    STATIC_FOLDER = os.path.join('app', 'static')

    # File upload config
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size
    ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx', 'xls', 'txt', 'dat'}

    # Database config
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API keys
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY') or "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"

    # AI model config
    OPENROUTER_MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    ENV = 'development'
    # Additional development settings can go here


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    ENV = 'testing'
    # Test-specific settings can go here
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """Production configuration."""
    # Production-specific settings can go here
    DEBUG = False
    TESTING = False
    ENV = 'production'

    # In production, check if secret key is set
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        # Log warning if using default secret key
        if not os.environ.get('SECRET_KEY'):
            logger.warning("No SECRET_KEY set for production environment. Using auto-generated key, which will change on restart!")


# Dictionary to easily select the config based on environment
config_dict = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    # Default to development
    'default': DevelopmentConfig
}


def get_config():
    """Get the configuration based on the FLASK_ENV environment variable."""
    flask_env = os.environ.get('FLASK_ENV', 'development')  # Default to development
    return config_dict.get(flask_env, config_dict['default'])