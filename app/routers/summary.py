import os
import re
import textwrap
from typing import List
from fastapi import APIRouter

import openai
from pydantic import BaseModel, validator

import kss

from asyncio import run

import firebase_admin
from firebase_admin import credentials, db, firestore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
firebase_auth_path = os.path.join(BASE_DIR, "firebase_auth_key.json")
cred = credentials.Certificate(firebase_auth_path)
firebase_admin.initialize_app(cred)

# Firestore 클라이언트 생성
db = firestore.client()

# "pdfs" 컬렉션에 대한 참조 설정
pdfs_ref = db.collection("pdfs")


# "pdfs" 컬렉션의 모든 문서 가져오기
docs = pdfs_ref.stream()

parent_doc_ref = db.collection("pdfs").document("AVwKksft99pxb2mgShed")
subcollection_ref = parent_doc_ref.collection("dictionary")

# 하위 컬렉션의 모든 문서 가져오기
subcollection_docs = subcollection_ref.stream()

for doc in subcollection_docs:
    # 문서의 데이터 출력
    # print(f"Subcollection Document ID: {doc.id}")
    head_value = doc.get("head")
    print(f"Head Value: {head_value}")


class Problem_Text(BaseModel):
    type: str
    text: List[str]
    problem_count: str = 3
    max_summarize_chars: int = 9000
    max_chars_per_request: int = 3000
    summary_length: int = 2000

    @validator('text')
    def validate_text(cls, value):
        if len(value) == 0:
            raise ValueError("Text list should not be empty.")
        return value


class Input_Text(BaseModel):
    text: List[str]
    max_summarize_chars: int = 9000
    max_chars_per_request: int = 4000
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


class Input_Text_Check(BaseModel):
    problem: str
    userAnswer: str
    impomation : str


router = APIRouter()


async def generate_summary_turbo(text: str, max_token: int = 500):
    prompt = f"summarize this for a student in Korean : {text}"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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
    prompt = f"당신의 기능을 통해 전문용어 사전을 만들 계획입니다. 아래의 글에서 전문용어를 추출하여 번호 없이 해당 전문용어의 설명을 :이 기호 이후에 설명해주세요. {text}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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

async def add_recommand_word(text: str, max_token: int = 500):
    prompt = f"당신의 기능을 통해 전문용어 사전을 만들 계획입니다. 해당 분야나 주제와 관련된 번호 없이 전문 용어를 추천해 주세요.또한 전문용어의 설명을 :이 기호 이후에 설명해주세요. 아래는 몇 가지 이미 알려진 용어의 예시입니다: {text} 이러한 용어를 고려하여 추가 전문 용어를 추천해 주세요."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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


async def create_problem(text: str, type: str, problem_count: int):
    prompt = f"아래 정보를 기반으로 {type}형식의 {problem_count}개의 문제를 만들어주세요. 그리고 그에 대한 답안을 함께 제시해주세요. 문제에 번호를 붙히지 말아주세요. 객관식 형식의 경우 4개의 보기를 주고 알맞은 보기를 고르는 형식입니다. 문제는 '문제 : ', 답안은 '답안 : ' 형태로 제시해주세요. {text}"
    print(prompt)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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


async def check_problem(problem: str, userAnswer: str, impomation: str):
    prompt = f" 문제와 사용자가 제시한 답안을 제공하겠습니다. 추가로 제시한 정보를 통해 정답을 확인해주세요. 맞다면 '정답', 틀렸다면 '오답'을 첫번째 줄에 말해주세요. 해당문제에 대한 '해설:'를 통해 해설해주세요. 문제와 사용자가 제시한 답안입니다. 문제 :{problem}, 답안 :{userAnswer}, 이는 정답확인을 위해 제공하는 정보입니다. 정보 :{impomation} "
    print(prompt)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {"role": "system", "content": "당신은 주어진 문제와, 사용자가 제시한 답안을 통해 답이 맞았는지, 틀렸는지 확인하는 유능한 어시스던트입니다."},
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    response = completion.choices[0].message["content"]
    print(response)
    return response



async def generate_refine_gpt3(text: str, max_length: int = 1000):
    prompt = f"한국어로 아래 글을 읽기 쉽게 수정해주세요. : {text}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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
    print(text)
    prompt = f"아래의 글을 통해 목차로 정할 가장 추천하는 10글자 이내의 구문 단 하나만 말해주세요 : {text}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {"role": "system", "content": "너는 글의 목차 생성 돕는 유능한 어시스던트입니다.."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    print(completion.choices[0].message["content"])
    output = completion.choices[0].message["content"]
    extract = output.replace("\\", "").replace("\"", "").replace(".", "")
    return extract


@router.post("/chat_gpt")
async def chat_gpt(chat_text: Chat_Text):
    prompt = f"아래의 글에 대해 답변해주세요 {chat_text.chat}"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        messages=[
            {"role": "system", "content": f"너는 유능한 {chat_text.role} 분야의 전문가입니다."},
            {"role": "user", "content": "한국어로 아래 글을 읽기 쉽게 수정해주세요."},
            {"role": "assistant", "content": f"{chat_text.text}"},
            {"role": "user", "content": f"{chat_text.chat}"}
        ],
    )

    response = completion.choices[0].message["content"]
    return response


async def handle_large_text(input_data: Input_Text, process_function: callable, purpose: str = "output"):
    text = input_data.text  # 텍스트 가져오기 리스트
    print("sasa")
    print(input_data)
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


async def handle_large_text_word(input_data: Input_Text, process_function: callable, purpose: str = "output"):
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

    word, sentence = word_separate(final_output_texts)
    return {f"word": word, f"sentence": sentence}

async def handle_large_text_problem(input_data: Input_Text, process_function: callable, purpose: str = "output"):
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
            processed_text = await process_function(sub_chunk, input_data.type, input_data.problem_count)
            output_chunks.append(processed_text)
        final_output_texts.append(output_chunks)

    problem, answer = parse_response(final_output_texts)
    return {f"problem": problem, f"answer": answer}


async def handle_large_text_problem_check(input_data: Input_Text_Check, process_function: callable, purpose: str = "output"):
    problem = input_data.problem
    userAnswer = input_data.userAnswer
    impomation = input_data.impomation
    processed_text = await process_function(problem, userAnswer, impomation)
    print(processed_text)
    correctness_status, explanation = parse_response_check(processed_text)
    return {f"correctness_status": correctness_status, f"explanation": explanation}

def parse_response(response):
    # Extract the first element of the list which is the actual response string
    response_str = response[0][0]
    print(response_str)
    # Split the response into parts where "문제" indicates a new question
    parts = response_str.split('문제 : ')
    print(parts)
    # Initialize empty lists to hold problems and answers
    problems = []
    answers = []
    for part in parts[1:]:  # The first split is empty so we skip it
        # Now, we further split each part into problem and answer using "답안:"
        problem, answer = part.split('\n답안 : ')
        # Append the problem part to problems list trimming whitespace
        problem = problem.replace('\n보기:', '')
        problem = problem.replace('\n보기 :', '')
        problem = problem.replace('1.', '')
        problem = problem.replace('2.', '')
        problem = problem.replace('3.', '')
        problem = problem.replace('4.', '')
        problems.append(problem.strip())
        # Append the answer part to answers list trimming whitespace
        answers.append(answer.strip())

    return problems, answers

def parse_response_check(response):
    # Extract the first element of the list which is the actual response string
    # Split the response into parts based on the line breaks
    # Initialize variables to hold the correctness status and explanation
    correctness_status = None
    explanation = []
    parts =response.split('\n')
    # Iterate over the parts to find and extract the required information
    for part in parts:
        part = part.strip()  # Preprocess to remove any leading/trailing whitespaces
        print(f"Checking part: '{part}'")  # Debug print

        # Check if the part indicates correctness status
        if part.startswith("오답") or part.startswith("정답"):
            correctness_status = part
            print(f"Status found '{correctness_status}'")

        # Extract the explanation part after the "해설:" keyword
        if part.startswith("해설:"):
            explanation = part.split("해설:")[1].strip()
            print(f"Explanation found'{explanation}'")

    # Return the correctness status and the explanation as a tuple
    return correctness_status, explanation



def word_separate(input_list):
    word = []
    sentence = []
    for A in input_list:
        for item in A:
            # ":"를 기준으로 문자열을 분할
            item_str = str(item)
            parts = item_str.split(":")

            if len(parts) == 1:
                # ":"가 없으면 이전 문장 문자열에 추가
                previous_sentence += " " + parts[0].strip()
            elif len(parts) == 2:
                word.append(parts[0].strip())
                sentence.append(parts[1].strip())
                previous_sentence = ""  # 새로운 단어와 설명을 시작하면 이전 문장 초기화

    return word, sentence


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


@router.post("/word")
async def word(input_data: Input_Text):
    extracted_word = await handle_large_text_word(input_data, extract_word)
    return extracted_word


@router.post("/problem")
async def problem(input_data: Problem_Text):
    extract_problem = await handle_large_text_problem(input_data, create_problem)
    return extract_problem

@router.post("/checkProblem")
async def problem(Input_Text_Check: Input_Text_Check):
    extract_problem = await handle_large_text_problem_check(Input_Text_Check, check_problem)
    return extract_problem
