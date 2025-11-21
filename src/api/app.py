import os
import asyncio
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- LangChain 및 DB 관련 라이브러리 ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- .env 파일 로드 ---
load_dotenv()

# --- 1. 설정 ---
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "nlpai-lab/KURE-v1" # 고려대학교 NLP & AI 연구실의 KURE 모델

# --- 2. 모델 및 DB 로딩 (서버 시작 시 1회 실행) ---
print("FastAPI 서버 시작 중...")

# 임베딩 모델 로드 (HuggingFace)
try:
    print(f"임베딩 모델 로드 중: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
except Exception as e:
    print(f"임베딩 모델 로드 실패: {e}")
    raise

# Vector DB 로드
if not os.path.exists(DB_FAISS_PATH):
    print(f"오류: Vector DB 경로를 찾을 수 없습니다. '{DB_FAISS_PATH}'")
    # raise FileNotFoundError(f"FAISS 인덱스 '{DB_FAISS_PATH}'를 찾을 수 없습니다.")
    # DB가 없어도 서버는 띄우되 검색 시 에러 처리
    db = None
else:
    try:
        print(f"Vector DB 로드 중: {DB_FAISS_PATH}")
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings,
            allow_dangerous_deserialization=True 
        )
        print("Vector DB 로드 완료.")
    except Exception as e:
        print(f"Vector DB 로드 실패: {e}")
        db = None

# LLM 초기화 (비동기 호출용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 3. FastAPI 앱 생성 ---
app = FastAPI(
    title="FactCheck RAG API",
    description="뉴스 기사를 분석하여 팩트체크 DB와 비교하는 API"
)

# --- 4. 응답 모델 정의 ---
class FactCheckResponse(BaseModel):
    original_claims: List[Dict[str, str]]
    related_factchecks: List[Dict[str, Any]]

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

# --- 5. 비동기 헬퍼 함수 ---

async def crawl_naver_news_async(url: str) -> str:
    """
    네이버 뉴스 크롤링 (비동기 래퍼)
    """
    def _crawl():
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 네이버 뉴스 본문 추출 (dic_area)
            content = soup.select_one('#dic_area')
            if not content:
                content = soup.select_one('#articeBody')
                
            if content:
                return content.get_text(strip=True)
            return ""
        except Exception as e:
            print(f"크롤링 오류: {e}")
            return ""

    # requests는 동기 라이브러리이므로 run_in_executor로 실행
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _crawl)

async def extract_claims_async(text: str):
    """
    기사 텍스트에서 주장 추출 (비동기)
    """
    if not text or len(text) < 50:
        return []
        
    system_prompt = """당신은 뉴스 기사 분석 전문가입니다. 입력된 텍스트에서 핵심 주장들을 추출하여 구조화된 JSON 형식으로 출력해야 합니다.

### 지시사항
1. 텍스트를 분석하여 핵심 주장(Claim)을 3~5개 추출하시오.
2. 복합문은 반드시 단문으로 분리하시오.
3. 각 주장에 대해 다음 속성을 분류하시오:
   - type: "Fact" (사실) 또는 "Opinion" (의견/해석)
   - query: 이 주장의 진위 여부나 다른 관점을 찾기 위해 검색할 검색어

### 출력 형식 (JSON)
[
  {{
    "claim": "추출된 주장 문장",
    "type": "Fact",
    "query": "검색 엔진에 입력할 중립적인 쿼리"
  }},
  ...
]"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "### 입력 텍스트\n{text}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        return await chain.ainvoke({"text": text})
    except Exception as e:
        print(f"클레임 추출 오류: {e}")
        return []

# --- 6. API 엔드포인트 ---

@app.get("/check-facts", response_model=FactCheckResponse, summary="네이버 뉴스 링크로 팩트체크 검색")
async def check_facts_by_url(url: str):
    """
    네이버 뉴스 URL을 받아 기사 내용을 크롤링하고, 
    핵심 주장을 추출한 뒤, 저장된 팩트체크 DB에서 관련 정보를 검색합니다.
    """
    if not db:
        raise HTTPException(status_code=503, detail="Vector DB가 로드되지 않았습니다.")

    # 1. 크롤링 (비동기)
    article_text = await crawl_naver_news_async(url)
    if not article_text:
        raise HTTPException(status_code=400, detail="기사 내용을 가져올 수 없습니다.")
        
    # 2. 클레임 추출 (비동기)
    claims = await extract_claims_async(article_text)
    if not claims:
        raise HTTPException(status_code=400, detail="기사에서 주장을 추출할 수 없습니다.")
        
    # 3. DB 검색 (동기 - FAISS는 CPU 연산 위주라 비동기 효과 적음, 필요시 executor 사용)
    related_results = []
    
    # 각 주장에 대해 검색 수행
    for claim in claims:
        query = claim.get('query')
        if not query:
            continue
            
        # 유사도 검색
        docs_with_scores = db.similarity_search_with_score(query, k=2)
        
        search_hits = []
        for doc, score in docs_with_scores:
            # 점수 필터링 (KURE 모델 기준, 필요시 조정)
            # L2 거리 기준: 0에 가까울수록 유사함
            # 너무 거리가 먼(관련 없는) 결과는 제외
            if score > 0.9: 
                continue

            search_hits.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) # float 변환
            })
            
        related_results.append({
            "claim": claim.get('claim'),
            "query": query,
            "related_facts": search_hits
        })
        
    return FactCheckResponse(
        original_claims=claims,
        related_factchecks=related_results
    )

@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "OK", "message": "FactCheck RAG API is running."}


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