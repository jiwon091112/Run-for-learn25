## 0. Prerequisite
- .env의 OPENAI_API_KEY
- `pip install -r requirements.txt`
- `npm install`
## 1. News Crawling 
`node src/crawling.js`
## 2. FastAPI Server
`python -m uvicorn src.api.app:app --reload`
-  포트 가시성 : Private -> Public
## 3. Vector DB
`python src/api/main.py
## 4. Chrome Extension Installation
- 1. chorme-extension download
- 2. chrome://extensions/ 
- 3. 개발자 모드 ON
- 4. '압축해제된 확장프로그램 로드'
- 5. 속성 -> 확장 프로그램 옵션 -> `https://expert-space-lamp-v5w4r9w765pcw65p-8000.app.github.dev/` -> 저장
