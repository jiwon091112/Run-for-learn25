import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# 현재 파일(main.py)의 위치를 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "../../asset/factcheck_ai_summary.txt")
DB_FAISS_PATH = os.path.join(BASE_DIR, "faiss_index_openai")

# OpenAI의 최신 임베딩 모델 (가성비와 성능이 가장 좋음)
# text-embedding-3-small: 빠르고 저렴함 (추천)
# text-embedding-3-large: 성능이 더 좋음 (정밀도가 중요할 때)
EMBEDDING_MODEL = "text-embedding-3-small" 

load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("Error: .env 파일에 OPENAI_API_KEY가 없습니다.")
    exit(1)

def load_documents_from_txt(file_path):
    """
    [언론사][ShortURL] 내용 형식의 텍스트 파일을 파싱
    """
    documents = []
    pattern = re.compile(r'^\[(.*?)\]\[(.*?)\]\s*(.*)$')

    if not os.path.exists(file_path):
        print(f"Error: 파일이 존재하지 않습니다 - {file_path}")
        return []

    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            match = pattern.match(line)
            if match:
                press, short_url, content = match.groups()
                full_url = f"https://n.news.naver.com/article/{short_url}"

                metadata = {
                    "source": full_url,
                    "press": press,
                    "type": "factcheck_summary"
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
    return documents

def validate_claim(vector_store, claim):
    """
    벡터 DB에서 관련 내용을 검색하고 LLM을 통해 팩트체크를 수행합니다.
    """
    # 1. 검색 (Retrieval)
    # 유사도 점수 기반으로 검색 (score가 낮을수록 유사함)
    docs_with_scores = vector_store.similarity_search_with_score(claim, k=3)
    
    # 관련성 필터링 (예: score < 0.5 인 것만 사용 등, 여기서는 일단 다 사용)
    relevant_docs = [doc for doc, score in docs_with_scores]
    
    if not relevant_docs:
        return "관련된 팩트체크 기사를 찾을 수 없습니다."

    # 2. 문맥 구성
    context_text = ""
    for i, doc in enumerate(relevant_docs):
        context_text += f"[기사 {i+1}] (출처: {doc.metadata.get('press')})\n{doc.page_content}\n\n"

    # 3. LLM 검증 (Generation)
    template = """당신은 팩트체크 전문 AI 어시스턴트입니다.
사용자의 주장이 사실인지 거짓인지, 아래 제공된 [관련 팩트체크 기사]만을 근거로 판단해주세요.

[관련 팩트체크 기사]
{context}

[사용자 주장]
{claim}

[지시사항]
1. 주장이 기사 내용과 일치하면 '사실', 반대되면 '거짓', 알 수 없으면 '판단 불가'로 결론 내리세요.
2. '거짓'인 경우, 기사의 어떤 부분과 배치되는지 구체적으로 설명하고 올바른 정보를 제시하세요.
3. 답변은 한국어로 정중하고 명확하게 작성하세요.
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context_text, "claim": claim})

def main():
    # 1. OpenAI 임베딩 모델 초기화
    print(f"Initializing OpenAI Embeddings ({EMBEDDING_MODEL})...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = None

    # 2. 벡터 스토어 로드 또는 생성
    if os.path.exists(DB_FAISS_PATH):
        print(f"Loading existing Vector DB from '{DB_FAISS_PATH}'...")
        try:
            vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading DB: {e}")
            print("Re-creating DB...")
    
    if vector_store is None:
        print("Creating new Vector DB...")
        docs = load_documents_from_txt(FILE_PATH)
        if not docs:
            print("저장할 데이터가 없습니다. 프로그램을 종료합니다.")
            return
        print(f"Successfully loaded {len(docs)} documents.")
        print("Creating Vector Store (FAISS)... Sending data to OpenAI...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(DB_FAISS_PATH)
        print(f"Done! Vector DB saved to '{DB_FAISS_PATH}'")

    print("✅ Vector DB setup complete. Now you can run the API server.")


if __name__ == "__main__":
    main()