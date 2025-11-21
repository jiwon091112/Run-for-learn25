import json
import re
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- Configuration ---
FILE_PATH = "asset/data.json"
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "nlpai-lab/KURE-v1" # 고려대학교 NLP & AI 연구실의 KURE 모델

load_dotenv()

def load_articles(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', re.sub(r"\\", '', text)).strip()

async def extract_claims_async(article_text, llm):
    """
    기사 텍스트에서 핵심 주장을 비동기로 추출합니다.
    """
    if not article_text or len(article_text) < 50:
        return []

    system_prompt = """당신은 팩트체크 전문가입니다. 기사에서 검증 가능한 핵심 주장(Claim)을 3개 이내로 추출하세요.
    
    출력 형식 (JSON):
    [
      {{
        "claim": "주장 내용 (한 문장)",
        "type": "Fact" 또는 "Opinion",
        "query": "검색용 쿼리"
      }}
    ]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{article_text}")
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        return await chain.ainvoke({"article_text": article_text})
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []

async def process_articles_to_docs(articles):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    docs = []
    
    print(f"Processing {len(articles)} articles...")
    
    # 비동기로 여러 기사를 동시에 처리 (속도 향상)
    # 주의: Rate Limit에 걸릴 수 있으므로 배치 처리가 필요할 수 있음
    tasks = []
    for i, article in enumerate(articles):
        # 테스트를 위해 5개만 처리 (전체 처리 시 제거)
        #if i >= 5: break 
        
        text = clean_text(article.get('dic_area'))
        if not text: continue
        
        tasks.append((article, extract_claims_async(text, llm)))

    # 결과 수집
    for article, task in tasks:
        claims = await task
        for claim in claims:
            content = f"Claim: {claim['claim']}\nType: {claim['type']}"
            metadata = {
                "source_url": article.get("url"),
                "source_title": article.get("title"),
                "press": article.get("press"),
                "original_claim": claim['claim'],
                "search_query": claim['query']
            }
            docs.append(Document(page_content=content, metadata=metadata))
            
    return docs

def main():
    # 1. 데이터 로드
    articles = load_articles(FILE_PATH)
    if not articles: return

    # 2. 주장 추출 (비동기 실행)
    docs = asyncio.run(process_articles_to_docs(articles))
    print(f"Extracted {len(docs)} claims from articles.")

    if not docs:
        print("No documents to save.")
        return

    # 3. 임베딩 및 DB 저장
    print("Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Creating Vector Store...")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(DB_FAISS_PATH)
    print(f"Vector DB saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()