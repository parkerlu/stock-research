from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db"
    database_url_sync: str = "postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db"
    tushare_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
