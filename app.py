import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- .env 파일 로드 라이브러리 ---
from dotenv import load_dotenv

# --- LangChain 및 DB 관련 라이브러리 ---
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

# --- .env 파일 로드 (스크립트 시작 시) ---
load_dotenv()

# --- 1. 설정 ---
DB_FAISS_PATH = "faiss_index"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- 2. 모델 및 DB 로딩 (서버 시작 시 1회 실행) ---
print("FastAPI 서버 시작 중...")

embeddings: Embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    raise ValueError("OPENAI_API_KEY가 필요합니다.")

try:
    print(f"OpenAI 임베딩 모델 로드 중: {OPENAI_EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model=OPENAI_EMBEDDING_MODEL
    )
except Exception as e:
    print(f"OpenAI 임베딩 모델 로드 실패: {e}")
    raise

if not os.path.exists(DB_FAISS_PATH):
    print(f"오류: Vector DB 경로를 찾을 수 없습니다. '{DB_FAISS_PATH}'")
    raise FileNotFoundError(f"FAISS 인덱스 '{DB_FAISS_PATH}'를 찾을 수 없습니다.")

try:
    print(f"Vector DB 로드 중: {DB_FAISS_PATH}")
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings,
        allow_dangerous_deserialization=True 
    )
    print("서버가 성공적으로 준비되었습니다.")
except Exception as e:
    print(f"Vector DB 로드 실패: {e}")
    raise

# --- 3. FastAPI 앱 생성 ---
app = FastAPI(
    title="Latent Space API (OpenAI)",
    description="FAISS Vector DB와 상호작용하는 API"
)

# --- 4. 응답 모델 정의 (JSON 출력을 위함) ---

# ▼▼▼▼▼▼▼▼▼▼▼▼▼ 'score' 필드 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼
class SearchResult(BaseModel):
    """
    개별 검색 결과를 위한 Pydantic 모델
    """
    content: str
    metadata: Dict[str, Any]
    score: float  # <-- '거리' 점수를 저장할 필드
# ▲▲▲▲▲▲▲▲▲▲▲▲▲ 'score' 필드 추가 ▲▲▲▲▲▲▲▲▲▲▲▲▲

class SearchResponse(BaseModel):
    """
    /search 엔드포인트의 최종 응답 모델
    """
    query: str
    results: List[SearchResult]

class EmbedResponse(BaseModel):
    """
    /embed 엔드포인트의 최종 응답 모델
    """
    text: str
    vector: List[float]


# --- 5. API 엔드포인트 정의 ---

@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "OK", "message": "Latent Space API가 실행 중입니다."}


# ▼▼▼▼▼▼▼▼▼▼▼▼▼ 'search' 엔드포인트 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼
@app.get("/search", 
         response_model=SearchResponse,
         summary="잠재 공간에서 유사 문서 검색 (점수 포함)")
def search_latent_space(q: str, k: int = 3):
    """
    쿼리 텍스트(q)를 받아 잠재 공간에서 가장 유사한 k개의 문서를
    '거리 점수(score)'와 함께 검색합니다.
    """
    if not q:
        raise HTTPException(status_code=400, detail="쿼리 파라미터 'q'가 필요합니다.")
        
    try:
        # 'similarity_search' 대신 'similarity_search_with_score' 사용
        # 이 함수는 (Document, score) 튜플의 리스트를 반환합니다.
        docs_with_scores = db.similarity_search_with_score(q, k=k)
        
        # 결과를 Pydantic 모델에 맞게 가공
        results_list = []
        for doc, score in docs_with_scores:
            results_list.append(
                SearchResult(
                    content=doc.page_content, 
                    metadata=doc.metadata, 
                    score=score  # <-- 점수를 응답에 포함
                )
            )
        
        return SearchResponse(query=q, results=results_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {e}")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲ 'search' 엔드포인트 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲


@app.get("/embed", 
         response_model=EmbedResponse,
         summary="텍스트의 잠재 공간 벡터 확인")
def get_embedding(q: str):
    """
    입력된 텍스트(q)를 잠재 공간의 실제 벡터(좌표)로 변환하여 반환합니다.
    """
    if not q:
        raise HTTPException(status_code=400, detail="쿼리 파라미터 'q'가 필요합니다.")
        
    try:
        vector = embeddings.embed_query(q)
        return EmbedResponse(text=q, vector=vector)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 중 오류 발생: {e}")