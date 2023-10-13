import textwrap
from typing import List
from fastapi import APIRouter

import openai
from pydantic import BaseModel, validator


import kss

from asyncio import run


class Input_Text(BaseModel):
    text: List[str]
    max_summarize_chars: int = 9000
    max_chars_per_request: int = 3000
    summary_length: int = 1000

    @validator('text')
    def validate_text(cls, value):
        if len(value) == 0:
            raise ValueError("Text list should not be empty.")
        return value


class Chat_Text(BaseModel):
    text: List[str]
    chat: str
    role: str = "고등학교"
    max_token: int = 500

    @validator('text', 'chat')
    def validate_text(cls, value):
        if len(value) == 0:
            raise ValueError("Text list should not be empty.")
        return value


router = APIRouter()


async def generate_summary_turbo(text: str, max_token: int = 500):
    prompt = f"summarize this for a student in Korean : {text}"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for text summarization."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    summary = completion.choices[0].message["content"]
    return summary


async def extract_word(text: str, max_token: int = 500):
    prompt = f"당신의 기능을 통해 전문용어 사전을 만들 계획입니다. 아래의 글에서 전문용어를 추출하고 해당 전문용어의 설명을 :이 기호 이후에 설명해주세요. {text}"

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    response = completion.choices[0].message["content"]
    return response

async def generate_summary_davinci(text: str, max_length: int = 1000):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"summarize the following sentence in Korean : {text}",
        max_tokens=2000,
        temperature=1,
    )
    summary = response.choices[0].text.strip()
    return summary


async def generate_refine_gpt3(text: str, max_length: int = 1000):
    prompt = f"한국어로 아래 글을 읽기 쉽게 수정해줘. : {text}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    refine = completion.choices[0].message["content"]
    return refine


async def extract_table(text: str, max_length: int = 1000):
    prompt = f"아래의 글을 통해 목차로 정할 가장 추천하는 10글자 이내의 구문을 말해줘 : {text}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {"role": "system", "content": "너는 글의 목차 생성 돕는 유능한 어시스던트야."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    output = completion.choices[0].message["content"]
    extract = output.replace("\\", "").replace("\"", "").replace(".", "")
    return extract


@router.post("/chat_gpt")
async def chat_gpt(chat_text: Chat_Text):
    prompt = f"아래의 글에 대해 답변해줘 {chat_text.chat}"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {"role": "system", "content": f"너는 유능한 {chat_text.role} 분야의 전문가야"},
            {"role": "user", "content": "한국어로 아래 글을 읽기 쉽게 수정해줘."},
            {"role": "assistant", "content": f"{chat_text.text}"},
            {"role": "user", "content": f"{chat_text.chat}"}
        ],
    )

    response = completion.choices[0].message["content"]
    return response


async def handle_large_text(input_data: Input_Text, process_function: callable, purpose: str = "output"):
    text = input_data.text  # 텍스트 가져오기 리스트
    max_summarize_chars = input_data.max_summarize_chars
    max_chars_per_request = input_data.max_chars_per_request
    output_length = input_data.summary_length
    final_output_texts = []

    for text_chunk in text:
        wrapped_text = textwrap.wrap(text_chunk, max_chars_per_request)
        length = max_summarize_chars // max_chars_per_request
        wrapped_text = wrapped_text[:length]
        output_chunks = []
        for sub_chunk in enumerate(wrapped_text):
            processed_text = await process_function(sub_chunk, output_length)
            output_chunks.append(processed_text)
            split = await split_sentences(output_chunks)
        final_output_texts.append(split)
    return {f"{purpose}": final_output_texts}


async def split_sentences(input_data):
    split_sentence = kss.split_sentences(input_data)
    return split_sentence

def word_separate(input_sentences): ## 단어 분리
    word = []       # 단어를 저장할 리스트
    sentence = []   # 설명을 저장할 리스트

    for sentence_str in input_sentences:
        # ":" 기준으로 문자열을 나눕니다.
        parts = sentence_str.split(":")

        if len(parts) == 2:  # ":"가 발견되면
            word.append(parts[0].strip())
            sentence.append(parts[1].strip())
        elif sentence:  # 이전에 저장한 sentence가 있다면
            sentence[-1] += " " + sentence_str  # 이전 sentence에 추가

    return word, sentence

@router.post("/summarize_large_text_davinci")
async def summary_large_text_davinci(input_data: Input_Text):
    result = await handle_large_text(input_data, generate_summary_davinci)  # summarize_large_text 함수를 호출하여 결과를 받아옴
    return result


@router.post("/summarize_large_text_GPT3.5_Turbo")
async def summary_large_text_turbo(input_data: Input_Text):
    result = await handle_large_text(input_data, generate_summary_turbo)  # summarize_large_text 함수를 호출하여 결과를 받아옴
    return result


@router.post("/refine_large_text_GPT3.5_Turbo")
async def refine_large_text(input_data: Input_Text):
    result = await handle_large_text(input_data, generate_refine_gpt3)  # summarize_large_text 함수를 호출하여 결과를 받아옴
    return result


@router.post("/extract_table_GPT3.5")
async def extract_table_large_text(input_data: Input_Text):
    result = await handle_large_text(input_data, extract_table)  # summarize_large_text 함수를 호출하여 결과를 받아옴
    return result


@router.post("/one_task_refine_extract")
def extract_table_large_text(input_data: Input_Text):
    refine = run(handle_large_text(input_data, generate_refine_gpt3))
    extract = run(handle_large_text(input_data, extract_table, "extract"))
    return refine, extract


@router.post("/one_task_summary_extract")
async def one_task_summary_extract_table(input_data: Input_Text):
    summary = await handle_large_text(input_data, generate_summary_turbo)
    extract = await handle_large_text(input_data, extract_table, "extract")
    return summary, extract

@router.post("/test")
async def test(input_data: Input_Text):
    extracted_word = await handle_large_text(input_data, extract_word())
    word, sentence = word_separate(extracted_word)
    return word, sentence


