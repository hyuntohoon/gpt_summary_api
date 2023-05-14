FROM python:3.9

WORKDIR /app

COPY ./app /app
COPY ./docker-compose.yml /app/docker-compose.yml

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
