# 파이썬 3.9 버전으로 실행
FROM python:3.9
# work 디렉토리를 지정하여 해당 프로젝트가 어디서 실행되는지 알려줌
WORKDIR /app
# work 디렉토리에서 ./app 경로에 있는 파일 및 디렉토리를 컨테이너 내부의 /app 경로로 복사
COPY ./app /app
# Dockerfile에서 사용되는 빌드-타임 변수(argument)를 정의
ARG OPENAI_API_KEY
# 정의된 빌드-타임 변수를 사용하여 런타임 환경 변수(environment variable)를 설정
ENV OPENAI_API_KEY=$OPENAI_API_KEY
# 해당 이미지를 통해 컨테이너 실행 했을 때의 명령어, requirements 내의 파일을 설치
RUN pip install --no-cache-dir -r /app/requirements.txt
# 이후 명령어를 통한 서버 시작 구문
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]
