# Python 기반 이미지 선택 (PyTorch + transformers 지원)
FROM python:3.10-slim

# 시스템 패키지 설치 (pandas, Pillow, torch 등 필수 의존성)
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 모든 파일 복사 (app.py, utils2.py, templates/, static/, csv 등)
COPY . .

# Python 패키지 설치
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    pandas \
    torch \
    torchvision \
    pillow \
    transformers

# Gunicorn으로 실행 (Fly.io는 포트 8080으로 들어오므로 매핑 필요)
ENV PORT 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
