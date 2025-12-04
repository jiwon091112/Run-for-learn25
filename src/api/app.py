import os
import httpx # requests 대신 사용하는 비동기 라이브러리 (pip install httpx)
import asyncio
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
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- .env 로드 ---
load_dotenv()

# --- 1. 설정 ---
# ★ 중요: DB 만들 때 쓴 모델과 똑같은 걸 써야 합니다!
# DB_FAISS_PATH = "faiss_index"         # 로컬 모델(KURE)로 만든 DB 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "faiss_index_openai")  # 절대 경로로 변경
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
    allow_origins=[
        "*", # 개발 편의를 위해 유지하되, 아래에 명시적 출처 추가
        "https://n.news.naver.com",
        "https://news.naver.com",
        "https://m.news.naver.com",
    ], 
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
  {{
    "claim": "주장 내용 (한 문장)",
    "type": "Fact" 또는 "Opinion",
    "query": "검색용 쿼리 (핵심 키워드 위주)"
  }}
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

async def verify_claim_with_llm(claim: str, related_docs: list):
    """
    주장과 검색된 팩트체크 기사를 비교하여 진위 여부를 판단
    """
    if not related_docs:
        return {"judgment": "판단 불가", "reason": "관련된 팩트체크 기사가 없습니다.", "reference_index": []}

    # 검색된 기사 내용 합치기
    context = ""
    for i, doc in enumerate(related_docs):
        context += f"[기사 {i+1}] (출처: {doc['metadata'].get('press')})\n{doc['content']}\n\n"

    system_prompt = """당신은 팩트체크 검증 AI입니다. 
사용자의 '주장(Claim)'과 이를 검증할 수 있는 '팩트체크 기사들(Context)'이 주어집니다.
기사 내용을 바탕으로 주장이 '사실', '거짓', '판단 불가' 중 무엇인지 판별하고, 
만약 '거짓'이라면 기사의 내용을 인용하여 왜 틀렸는지 구체적으로 반박하세요.

[출력 형식 - JSON]
{{
    "judgment": "사실" | "거짓" | "판단 불가",
    "reason": "판단 이유 및 반박 내용 (3문장 이내)",
    "reference_index": [참고한 기사 번호 (예: 1, 2)]
}}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "주장: {claim}\n\n팩트체크 기사들:\n{context}")
    ])
    
    llm = resources.get('llm')
    chain = prompt | llm | JsonOutputParser()
    
    try:
        return await chain.ainvoke({"claim": claim, "context": context})
    except Exception as e:
        print(f"검증 오류: {e}")
        return {"judgment": "오류", "reason": "검증 중 오류가 발생했습니다.", "reference_index": []}

async def process_single_claim(claim: Dict[str, str], db: Any) -> Optional[Dict[str, Any]]:
    """
    개별 주장에 대한 검색 및 검증을 수행하는 비동기 함수
    """
    query = claim.get('query')
    if not query: 
        return None

    # 1. DB 검색 (동기 함수이므로 별도 스레드에서 실행 고려 가능하나, FAISS가 빠르면 그냥 실행)
    # LangChain FAISS wrapper는 동기 함수임.
    docs_with_scores = db.similarity_search_with_score(query, k=2)
    
    search_hits = []
    for doc, score in docs_with_scores:
        # 거리(Distance) 기반 필터링
        if score > 1.2: 
            continue

        search_hits.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    
    # 2. LLM 검증 (비동기)
    verification = await verify_claim_with_llm(claim.get('claim'), search_hits)
    
    return {
        "claim": claim.get('claim'),
        "query": query,
        "related_facts": search_hits,
        "verification": verification
    }

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

    # 3. DB 검색 및 검증 (병렬 처리)
    # 각 주장에 대해 process_single_claim을 동시에 실행
    tasks = [process_single_claim(claim, db) for claim in claims]
    results = await asyncio.gather(*tasks)
    
    # None 결과 필터링
    related_results = [res for res in results if res is not None]

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