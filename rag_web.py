import os
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import markdown

def get_all_files(folder_path, extensions):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(tuple(extensions)):
                files.append(os.path.join(root, filename))
    return files

def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    html = markdown.markdown(md_content)
    return html

def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def convert_to_documents(pdf_files, md_files):
    documents = []

    for pdf_file in pdf_files:
        pdf_documents = read_pdf(pdf_file)
        for doc in pdf_documents:
            doc.metadata["source"] = pdf_file
        documents.extend(pdf_documents)

    for md_file in md_files:
        text = read_markdown(md_file)
        documents.append(Document(page_content=text, metadata={"source": md_file}))

    return documents

def build_faiss_index(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(documents)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [doc.page_content for doc in splitted_docs]
    metadatas = [doc.metadata for doc in splitted_docs]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    faiss_index = FAISS.from_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)
    return faiss_index

pdf_folder = r"D:\Repo\finalmobile5\finalUtilities\AppReversing\document\포렌식 관련 논문모음"
md_folder = r"T:\Repo\FinalProductNet.wiki"

pdf_files = get_all_files(pdf_folder, [".pdf"])
md_files = get_all_files(md_folder, [".md"])

documents = convert_to_documents(pdf_files, md_files)
faiss_index = build_faiss_index(documents)

faiss_index.save_local("faiss_vector_db")
print("FAISS 벡터 데이터베이스 생성 완료!")
