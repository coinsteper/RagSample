import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS

# 텍스트 분할
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_faiss_db(documents, db_path="faiss_db"):
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'},  # CPU 사용
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(db_path)  # DB 파일 저장
    print(f"FAISS DB가 {db_path}에 저장되었습니다.")
    return vectordb


# PDF 파일 로드 함수
def load_pdfs_from_directory(folder_path):
    documents = []
    
    # 폴더 내 모든 PDF 파일을 읽어 Document 형식으로 변환
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            
            # LangChain Document 형식으로 추가
            documents.append(Document(page_content=text, metadata={"source": file_path}))
    
    return documents

# Streamlit 앱 시작
def main():
    st.title("PDF 기반 RAG Chatbot")

    # PDF 파일이 저장된 폴더 경로
    pdf_folder_path = r"D:\Repo\finalmobile5\finalUtilities\AppReversing\document\포렌식 관련 논문모음"  # PDF 파일이 있는 폴더 경로를 지정하세요
    db_path = r"T:\Repo\python\web2\vectordb\my_gen_db"  # 벡터 DB를 저장할 경로를 지정하세요
    #db_path = r"T:\Repo\python\web2\vec2"

    # 첫 번째 실행 시 벡터 데이터베이스 생성
    if not os.path.exists(db_path):
        st.write("PDF 파일을 벡터화하여 데이터베이스를 생성하는 중입니다...")

        # 임베딩 모델 설정
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

        ### 1.크로마 방식
        #documents = load_pdfs_from_directory(pdf_folder_path)        
        # 텍스트 분할
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #texts = text_splitter.split_documents(documents)

        #vectordb = Chroma.from_documents(texts, embeddings, persist_directory=db_path)

        #vectordb.persist()

        ### 1.파이스 방식
        # documents = load_pdfs_from_directory(pdf_folder_path)
        # split_docs = split_documents(documents)
        # faiss_db = create_faiss_db(split_docs)
        
        st.write("벡터 DB 생성 완료")
    else:
        ### Chroma DB 로드
        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'},  # CPU 사용
        encode_kwargs={'normalize_embeddings': True}
    )
        #vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

        ## FAISS DB 로드
        vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # Llama 모델 로드
    llm = OllamaLLM(model="llama3.2")

    # 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # RAG 체인 생성
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(),
        memory=memory
    )

    # 질문 입력
    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 질문에 대한 답변 생성
        result = qa_chain({"question": user_input})
        answer = result['answer']

        # 답변 출력
        st.write(f"**답변:** {answer}")
        
        # 출처 정보 출력 (출처가 있을 경우에만)
        sources = result.get('source_documents', [])
        if sources:
            st.write("**출처:**")
            for i, doc in enumerate(sources, 1):
                source_path = doc.metadata.get("source", "Unknown source")
                st.write(f"- {source_path}")

# Streamlit 애플리케이션 실행
if __name__ == "__main__":
    main()
