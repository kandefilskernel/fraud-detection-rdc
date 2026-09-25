from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "backoffice-api"
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480          # une journée de travail d'analyste
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:19092"
    KAFKA_ENABLED: bool = True
    # Compte administrateur créé au premier démarrage s'il n'existe aucun utilisateur
    BOOTSTRAP_ADMIN_EMAIL: str = "admin@fraud-rdc.local"
    BOOTSTRAP_ADMIN_PASSWORD: str = "ChangeMoi-2026!"
    CORS_ORIGINS: str = "http://localhost:3000"


settings = Settings()
