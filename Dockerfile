# 베이스 이미지로 python:3.9-alpine 사용
FROM python:3.9-alpine

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 필요한 파일들을 복사
COPY ./app /app

# Dockerfile에서 사용되는 빌드-타임 변수(argument) 정의
ARG OPENAI_API_KEY

# 빌드-타임 변수를 사용하여 런타임 환경 변수 설정
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# 필요한 패키지 설치를 별도의 레이어로 분리하여 최적화
RUN apk --no-cache add --virtual .build-deps gcc musl-dev \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apk del .build-deps

# 서버 시작 구문
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
