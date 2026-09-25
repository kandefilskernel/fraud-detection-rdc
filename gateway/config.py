# gateway/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Nuru Fraud Detection API Gateway"
    ENVIRONMENT: str = "development"
    RATE_LIMIT_REQUESTS: int = 120
    RATE_LIMIT_WINDOW_SEC: int = 60
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore les variables d'environnement non déclarées ici
    )

settings = Settings()