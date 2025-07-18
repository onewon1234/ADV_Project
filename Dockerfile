# syntax=docker/dockerfile:1
FROM python:3.10-slim

# 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 파일 복사
COPY . .

# 필수 Python 패키지 설치
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pandas \
    torch \
    torchvision \
    pillow \
    transformers

# Fly.io는 8080 포트로 요청받음
ENV PORT=8080

# transformers의 advisory warning 비활성화
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Gunicorn 실행 (worker 1개, timeout 120초로 안정성 보완)
CMD ["gunicorn", "-w", "1", "-t", "120", "-b", "0.0.0.0:8080", "app:app"]
