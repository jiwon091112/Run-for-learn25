import os
import httpx # requests 대신 사용하는 비동기 라이브러리 (pip install httpx)
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # CORS 미들웨어
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- LangChain 관련 ---
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings # 로컬 모델 사용 시 주석 해제
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI 사용 시
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- .env 로드 ---
load_dotenv()

# --- 1. 설정 ---
# ★ 중요: DB 만들 때 쓴 모델과 똑같은 걸 써야 합니다!
# DB_FAISS_PATH = "faiss_index"         # 로컬 모델(KURE)로 만든 DB 경로
DB_FAISS_PATH = "faiss_index_openai"  # OpenAI로 만든 DB 경로
USE_OPENAI_EMBEDDING = True           # True면 OpenAI, False면 KURE(로컬)

# 전역 변수 (DB, Embeddings, LLM)
resources = {}

# --- 2. Lifespan (서버 시작/종료 시 실행) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [시작 시 실행]
    print("🚀 FastAPI 서버 시작 중...")
    
    # 1. 임베딩 모델 로드
    try:
        if USE_OPENAI_EMBEDDING:
            print("Settings: OpenAI Embeddings (text-embedding-3-small)")
            resources['embeddings'] = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            print("Settings: HuggingFace Embeddings (nlpai-lab/KURE-v1)")
            # resources['embeddings'] = HuggingFaceEmbeddings(
            #     model_name="nlpai-lab/KURE-v1",
            #     model_kwargs={'device': 'cpu'}
            # )
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        raise

    # 2. Vector DB 로드
    if os.path.exists(DB_FAISS_PATH):
        try:
            print(f"📂 Vector DB 로드 중: {DB_FAISS_PATH}")
            resources['db'] = FAISS.load_local(
                DB_FAISS_PATH, 
                resources['embeddings'],
                allow_dangerous_deserialization=True
            )
            print("✅ Vector DB 로드 완료.")
        except Exception as e:
            print(f"❌ Vector DB 로드 에러: {e}")
            resources['db'] = None
    else:
        print(f"⚠️ 경고: '{DB_FAISS_PATH}' 경로에 DB가 없습니다.")
        resources['db'] = None

    # 3. LLM 초기화
    resources['llm'] = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    yield # 여기서부터 서버 가동

    # [종료 시 실행]
    print("👋 서버 종료. 리소스를 정리합니다.")
    resources.clear()

# --- 3. FastAPI 앱 생성 ---
app = FastAPI(
    title="FactCheck RAG API",
    description="뉴스 기사 팩트체크 및 유사도 검색 API",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 연동 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 보안상 실제 운영 시에는 ["http://localhost:3000"] 등으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. 데이터 모델 ---
# [삭제/주석] GET 방식에서는 Request Body용 모델이 필요 없습니다.
# class FactCheckRequest(BaseModel):
#     url: str 

class FactCheckResponse(BaseModel):
    original_claims: List[Dict[str, str]]
    related_factchecks: List[Dict[str, Any]]

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class EmbedResponse(BaseModel):
    text: str
    vector: List[float]

# --- 5. 비동기 헬퍼 함수 ---

async def crawl_naver_news_async(url: str) -> str:
    """
    httpx를 사용한 진정한 비동기 크롤링
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # follow_redirects=True: 단축 URL 등 리다이렉트 자동 처리
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            response.raise_for_status()
            html = response.text
            
            soup = BeautifulSoup(html, 'html.parser')
            # 본문 추출 로직
            content = soup.select_one('#dic_area')
            if not content:
                content = soup.select_one('#articeBody')
            
            if content:
                # 불필요한 태그 제거
                for tag in content(['script', 'style', 'iframe', 'button']):
                    tag.decompose()
                return content.get_text(strip=True)
            return ""
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return ""

async def extract_claims_async(text: str):
    """
    뉴스 기사에서 주장 추출 (LLM)
    """
    if not text or len(text) < 50:
        return []

    system_prompt = """당신은 팩트체크를 위한 뉴스 분석가입니다. 
주어진 텍스트에서 '검증 가능한 핵심 주장(Claim)'을 3개 추출하세요.

[출력 형식 - JSON Only]
[
  {
    "claim": "주장 내용 (한 문장)",
    "type": "Fact" 또는 "Opinion",
    "query": "검색용 쿼리 (핵심 키워드 위주)"
  }
]
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    
    # 리소스에서 LLM 가져오기
    llm = resources.get('llm')
    if not llm: return []

    chain = prompt | llm | JsonOutputParser()
    
    try:
        # 본문이 너무 길면 잘라서 보냄 (토큰 절약)
        return await chain.ainvoke({"text": text[:3500]})
    except Exception as e:
        print(f"클레임 추출 오류: {e}")
        return []

# --- 6. 엔드포인트 ---

@app.get("/check-facts", response_model=FactCheckResponse, summary="URL 기반 팩트체크 (GET)")
async def check_facts_by_url(url: str):
    """
    [GET] /check-facts?url=https://n.news.naver.com/... 형태로 요청
    """
    db = resources.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Vector DB가 로드되지 않았습니다.")

    # 1. 크롤링
    article_text = await crawl_naver_news_async(url)
    if not article_text:
        raise HTTPException(status_code=400, detail="기사 내용을 가져올 수 없습니다.")

    # 2. 클레임 추출
    claims = await extract_claims_async(article_text)
    if not claims:
        # 주장이 안 뽑혔을 경우 빈 결과 반환 대신 에러 처리 선택 가능
        return FactCheckResponse(original_claims=[], related_factchecks=[])

    # 3. DB 검색
    related_results = []
    
    for claim in claims:
        query = claim.get('query')
        if not query: continue

        # k=2, 유사도 검색
        docs_with_scores = db.similarity_search_with_score(query, k=2)
        
        search_hits = []
        for doc, score in docs_with_scores:
            # 거리(Distance) 기반 필터링
            # OpenAI Embeddings + FAISS(L2)의 경우:
            # 0.0 = 완전 일치, 1.0 이상 = 관련 없음
            # 보통 0.5 ~ 0.7 사이를 임계값으로 잡음 (데이터에 따라 다름)
            
            # 너무 먼 결과 제외 (임계값 조정 필요)
            if score > 1.2: 
                continue

            search_hits.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
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

@app.get("/search", response_model=SearchResponse, summary="단순 검색")
def search_latent_space(q: str, k: int = 3):
    db = resources.get('db')
    if not db:
        raise HTTPException(status_code=503, detail="Vector DB Not Ready")
    
    docs_with_scores = db.similarity_search_with_score(q, k=k)
    
    results_list = []
    for doc, score in docs_with_scores:
        results_list.append(SearchResult(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score
        ))
    
    return SearchResponse(query=q, results=results_list)

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "OK", "model": "OpenAI" if USE_OPENAI_EMBEDDING else "Local"}