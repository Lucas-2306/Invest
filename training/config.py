from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str

    training_output_dir: str = "datasets"
    dataset_start_date: str = "2015-01-01"


settings = Settings()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(settings.training_output_dir)