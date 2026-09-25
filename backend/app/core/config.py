from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "fraud-rdc"
    ENVIRONMENT: str = "development"
    DATABASE_URL: str
    REDIS_URL: str | None = None
    SECRET_KEY: str = "change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_EMAIL: str = "admin@fraud-rdc.local"
    ADMIN_PASSWORD: str = "Admin@12345"


settings = Settings()
