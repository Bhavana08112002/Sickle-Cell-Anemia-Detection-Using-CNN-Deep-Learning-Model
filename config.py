from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Sickle Cell Detection API"

    # Path to your trained model (default inside repo)
    MODEL_PATH: str = str(Path(__file__).parent / "models" / "best_model.h5")

    # Input size expected by the model (height, width)
    MODEL_INPUT_SIZE: tuple[int, int] = (256, 256)

    # Allowed extensions for upload
    ALLOWED_FILE_TYPES: list[str] = ["jpg", "jpeg", "png"]

    MAX_FILE_SIZE_MB: int = 5

    DEBUG: bool = True
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "API for detecting sickle cell anemia from blood smear images using a CNN model."
    )

    # Runtime environment and server bind settings
    APP_ENV: str = "development"
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000



@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
