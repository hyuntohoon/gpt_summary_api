FROM python:3.9

WORKDIR /app

# Copy the application and docker-compose.yml
COPY ./app /app
COPY ./docker-compose.yml /app

# Install docker-compose in the Docker image
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get update && \
    apt-get install -y docker-compose

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
