from fastapi import FastAPI
from pydantic import BaseSettings, BaseModel
import openai


def unicode_escape_to_text(escaped_str: str) -> str:
    return escaped_str.encode('utf-8').decode('unicode_escape')


class Settings(BaseSettings):
    OPENAI_API_KEY: str = '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'

    class Config:
        env_file = '.env'

settings = Settings()
openai.api_key = settings.OPENAI_API_KEY

app = FastAPI()

class Text(BaseModel):
    text: str


@app.post("/summarize")
async def summarize(input: Text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input.text,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    summary = response.choices[0].text.strip()
    return {"summary": summary}

