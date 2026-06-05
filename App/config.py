import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    # Flask config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'

    # API config
    NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY')
    NEWS_API_BASE_URL = os.getenv('NEWS_API_BASE_URL', 'https://newsdata.io/api/1/latest')

    # Cohere config (AI-generated recommendations)
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    COHERE_MODEL = os.getenv('COHERE_MODEL', 'command-a-03-2025')

    # Validate required config
    if not NEWSDATA_API_KEY:
        raise ValueError("NEWSDATA_API_KEY must be set in environment variables")

    # Default fetch parameters
    DEFAULT_COUNTRY = os.getenv('DEFAULT_COUNTRY', 'us')
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
    DEFAULT_MAX_ARTICLES = 20

    # Model config (for later)
    MODEL_PATH = 'models/fake_news_model.pkl'
    VECTORIZER_PATH = 'models/vectorizer.pkl'