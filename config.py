from datetime import date,timedelta
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    DEFAULT_TO_DATE: date = date.today()
    DEFAULT_FROM_DATE: date = date.today() - timedelta(days=365)
    DEFAULT_DAYS_TERM: int = 7  ### 365
    
settings = Settings()