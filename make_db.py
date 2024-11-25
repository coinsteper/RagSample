from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_db(directory_path, db_name):
    # 문서 로더 설정
    loaders = {
        '.txt': DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader),
        '.pdf': DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        '.xlsx': DirectoryLoader(directory_path, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader),
        '.xls': DirectoryLoader(directory_path, glob="**/*.xls", loader_cls=UnstructuredExcelLoader)
    }
    
    # 모든 문서 로드
    documents = []
    for file_type, loader in loaders.items():
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_type} files: {e}")
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # 임베딩 모델 설정 (한국어) - CPU 버전
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'},  # CPU 사용
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # FAISS 벡터 DB 생성
    vectordb = FAISS.from_documents(texts, embeddings)
    
    # 벡터 DB 저장
    vectordb.save_local(f"vectordb/{db_name}")
    
    return vectordb

# 사용 예시
if __name__ == "__main__":
    # 벡터 DB를 저장할 디렉토리 생성
    os.makedirs("vectordb", exist_ok=True)
    
    # 문서가 있는 디렉토리 경로와 저장할 DB 이름 지정
    directory_path = r"D:\Repo\finalmobile5\finalUtilities\AppReversing\document\포렌식 관련 논문모음"
    db_name = "my_gen_db"
    
    vectordb = create_vector_db(directory_path, db_name)
    print("Vector DB created successfully!")
