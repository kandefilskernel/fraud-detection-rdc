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
    # Clé de service de l'API interne (retours des opérateurs, via l'integration-layer).
    # Vide = API interne désactivée.
    INTERNAL_API_KEY: str = ""

    # Assistant d'enquête (RAG). Sans ANTHROPIC_API_KEY dans l'environnement, la note est
    # assemblée sans modèle de langage (mode extractif) ; la recherche de cas fonctionne toujours.
    ASSISTANT_LLM_ENABLED: bool = True
    ASSISTANT_MODEL: str = "claude-opus-5"
    ASSISTANT_EFFORT: str = "medium"           # low | medium | high : latence vs profondeur
    ASSISTANT_MAX_TOKENS: int = 8000           # réflexion adaptative comprise
    ASSISTANT_TIMEOUT_S: float = 120.0
    ASSISTANT_USE_FALLBACKS: bool = True       # repli serveur en cas de refus du modèle
    ASSISTANT_CACHE_S: int = 900               # évite de refacturer un double clic
    KNOWLEDGE_DIR: str = ""                    # vide : services/backoffice-api/knowledge


settings = Settings()
