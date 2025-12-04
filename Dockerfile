# Python 3.10 slim 이미지 사용 (호환성 위해 3.10 권장)
FROM python:3.12.1

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 (필요한 경우)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 파일 복사
COPY . .

# 포트 노출
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]