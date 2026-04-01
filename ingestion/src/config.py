from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    database_url: str

    postgres_db: str = "stocks"
    postgres_user: str = "stocks_user"
    postgres_password: str = "stocks_pass"
    postgres_host: str = "db"
    postgres_port: int = 5432

    brapi_base_url: str = "https://brapi.dev/api"
    brapi_token: str | None = None


settings = Settings()