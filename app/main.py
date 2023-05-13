import uvicorn
from fastapi import FastAPI
from pydantic import BaseSettings
from routers import summary
import openai


class Settings(BaseSettings):
    OPENAI_API_KEY: str = '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'


settings = Settings()
app = FastAPI()
openai.api_key = settings.OPENAI_API_KEY
app.include_router(summary.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")

