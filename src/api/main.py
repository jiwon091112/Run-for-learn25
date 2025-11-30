import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings # 변경된 부분

# --- Configuration ---
FILE_PATH = "../../asset/factcheck_ai_summary.txt" 
DB_FAISS_PATH = "faiss_index_openai" # 저장 폴더명 변경 (구분 위해)

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

def main():
    # 1. 데이터 로드
    docs = load_documents_from_txt(FILE_PATH)
    
    if not docs:
        print("저장할 데이터가 없습니다.")
        return

    print(f"Successfully loaded {len(docs)} documents.")

    # 2. OpenAI 임베딩 모델 초기화
    print(f"Initializing OpenAI Embeddings ({EMBEDDING_MODEL})...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        # openai_api_key는 os.environ에서 자동으로 가져옵니다.
    )
    
    # 3. 벡터 스토어 생성 및 저장
    # OpenAI 서버로 텍스트를 보내 벡터값을 받아오므로 인터넷 연결 필수
    print("Creating Vector Store (FAISS)... Sending data to OpenAI...")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    
    vector_store.save_local(DB_FAISS_PATH)
    print(f"Done! Vector DB saved to '{DB_FAISS_PATH}'")

if __name__ == "__main__":
    main()