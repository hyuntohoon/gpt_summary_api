import uvicorn
from fastapi import FastAPI
from pydantic import BaseSettings
from routers import summary
import openai


def unicode_escape_to_text(escaped_str: str) -> str:
    return escaped_str.encode('utf-8').decode('unicode_escape')


def main():
    class Settings(BaseSettings):
        OPENAI_API_KEY: str = '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'
        class Config:
            env_file = '.env'

    settings = Settings()
    openai.api_key = settings.OPENAI_API_KEY

    app = FastAPI()
    app.include_router(summary.router)
    return app


if __name__ == "__main__":
    app = main()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
else:
    app = main()


