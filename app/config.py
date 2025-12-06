"""
Application configuration management
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/astrology_db"

    # LLM API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Application
    debug: bool = True
    log_level: str = "INFO"

    # Paths
    ephemeris_path: str = "./data/ephemeris"
    ml_model_path: str = "./data/models"
    training_data_path: str = "./data/training"

    # Meta-Learning Paths
    meta_learning_data_path: str = "./data/meta_learning"
    meta_learning_datasets_path: str = "./data/datasets"
    meta_learning_experiments_path: str = "./data/meta_learning/experiments"
    meta_learning_checkpoints_path: str = "./data/meta_learning/checkpoints"
    meta_learning_logs_path: str = "./data/meta_learning/logs"
    meta_learning_visualizations_path: str = "./data/meta_learning/visualizations"

    # Swiss Ephemeris
    use_swiss_ephemeris: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
