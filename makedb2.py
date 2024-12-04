import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import markdown

# 특정 폴더 내의 파일 가져오기
def get_all_files(folder_path, extensions):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(tuple(extensions)):
                files.append(os.path.join(root, filename))
    return files

# Markdown 파일 읽기
def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    html = markdown.markdown(md_content)
    return html

# PDF 파일 읽기
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# 파일을 Document 리스트로 변환
def convert_to_documents(pdf_files, md_files):
    documents = []

    # PDF 처리
    for pdf_file in pdf_files:
        pdf_documents = read_pdf(pdf_file)
        for doc in pdf_documents:
            doc.metadata["source"] = pdf_file
        documents.extend(pdf_documents)

    # Markdown 처리
    for md_file in md_files:
        text = read_markdown(md_file)
        documents.append(Document(page_content=text, metadata={"source": md_file}))

    return documents

# 텍스트를 벡터로 변환하여 FAISS 데이터베이스 생성
def build_faiss_index(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(documents)

    # Hugging Face 임베딩 모델 로드
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 텍스트 및 메타데이터 추출
    texts = [doc.page_content for doc in splitted_docs]
    metadatas = [doc.metadata for doc in splitted_docs]

    # FAISS 벡터 스토어 생성
    faiss_index = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    return faiss_index

# PDF와 Markdown 폴더 경로 설정
pdf_folder = r"D:\Repo\finalmobile5\finalUtilities\AppReversing\document\포렌식 관련 논문모음"  # PDF 파일이 있는 폴더
md_folder = r"T:\Repo\FinalProductNet.wiki"    # Markdown 파일이 있는 폴더

# 파일 가져오기
pdf_files = get_all_files(pdf_folder, [".pdf"])
md_files = get_all_files(md_folder, [".md"])

# 모든 파일 처리 및 FAISS 생성
documents = convert_to_documents(pdf_files, md_files)
faiss_index = build_faiss_index(documents)

# 벡터 데이터베이스 저장
faiss_index.save_local("faiss_vector_db")

print("FAISS 벡터 데이터베이스 생성 완료!")
