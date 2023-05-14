import textwrap
from fastapi import APIRouter
import openai
from pydantic import BaseModel


class Text(BaseModel):
    text: str
router = APIRouter()


@router.post("/summarize")
async def summarize(input: Text):

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= f"Could you please summarize the following sentence in Korean?\n\n{input.text}\n",
        max_tokens=2000,
        temperature=0.3,
    )
    summary = response.choices[0].text.strip()
    return {"summary": summary}



"""
def summarize_large_text(conversations: Conversations,
                         text: str,
                         max_summarize_chars: int = 9000,
                         max_chars_per_request: int = 4000,
                         summary_length: int = 1000) -> Conversations:


    wrapped_text = textwrap.wrap(text, max_chars_per_request)
    length =  max_summarize_chars // max_chars_per_request
    wrapped_text = wrapped_text[:length]

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for idx, chunk in enumerate(wrapped_text):
        my_bar.progress(idx, text=progress_text)
        summary_chunk = generate_summary(chunk, summary_length)
        conversations.add_message("user", f"summarize: {chunk}")
        conversations.add_message("assistant", summary_chunk)

    return conversations
"""