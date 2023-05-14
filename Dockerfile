FROM python:3.9

WORKDIR /app

COPY ./app /app

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
