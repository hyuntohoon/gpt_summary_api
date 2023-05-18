import textwrap
from typing import List
from fastapi import APIRouter

import openai
from pydantic import BaseModel


class SummarizeInput(BaseModel):
    text: List[str]
    max_summarize_chars: int = 9000
    max_chars_per_request: int = 4000
    summary_length: int = 1000


router = APIRouter()


async def generate_summary(text: str, max_length: int = 1000):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= f"Could you please summarize the following sentence in Korean?\n\n{text}\n",
        max_tokens=2000,
        temperature=0.7,
    )
    summary = response.choices[0].text.strip()
    return summary


@router.post("/summarize_large_text")
async def summarize_large_text(input_data: SummarizeInput):
    text = input_data.text # 텍스트 가져오기 리스트
    max_summarize_chars = input_data.max_summarize_chars
    max_chars_per_request = input_data.max_chars_per_request
    summary_length = input_data.summary_length
    final_summary_texts = []

    for text_chunk in text:
        wrapped_text = textwrap.wrap(text_chunk, max_chars_per_request)
        length = max_summarize_chars // max_chars_per_request
        wrapped_text = wrapped_text[:length]
        summary_chunks = []
        for sub_chunk in enumerate(wrapped_text):
            summarized_text = await generate_summary(sub_chunk, summary_length)
            summary_chunks.append(summarized_text)
        final_summary_texts.append(summary_chunks)

    return {"summary": final_summary_texts}

