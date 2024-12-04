import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Groq API 설정
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Groq LLM 호출 함수
def query_groq(prompt, history):
    messages = [
        {
            "role": "system", 
            "content": "너는 한국어로 대화하는 전문 AI 어시스턴트야. 사용자의 질문에 상세하고 명확하게 한국어로 답변해줘. RAG참조 데이터 기반으로 대답하고 거기에 없는 말은 모른다고 해줘"
        }
    ]
    for chat in history:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        stream=False
    )
    return chat_completion.choices[0].message.content

# FAISS 데이터베이스 로드 함수
def load_faiss_index(db_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# 질문 처리 함수
def handle_question(question, faiss_index, chat_history):
    relevant_docs = faiss_index.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"문맥:\n{context}\n\n질문: {question}\n\n답변:"
    answer = query_groq(prompt, chat_history)
    chat_history.append({"question": question, "answer": answer})
    return answer, relevant_docs

# Streamlit UI 구성
st.set_page_config(page_title="RAG 시스템", layout="wide")
st.title("FINAL ChatBot")

# 벡터 데이터베이스 로드
db_path = "faiss_vector_db"
faiss_index = load_faiss_index(db_path)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 질문 입력
question = st.text_input("질문을 입력하세요:", placeholder="예: 포렌식 기법에 대해 알려주세요")

# 질문 제출
if st.button("질문 제출"):
    if question:
        with st.spinner("Groq API로 처리 중..."):
            answer, sources = handle_question(question, faiss_index, st.session_state.chat_history)
            st.markdown(f"### 답변: {answer}")
            st.markdown("### 관련 문서:")
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**{i}.** {doc.metadata.get('source', 'Unknown Source')}")
    else:
        st.warning("질문을 입력하세요!")

# 대화 기록 표시
st.markdown("### 대화 기록:")
for chat in st.session_state.chat_history:
    st.markdown(f"- **질문:** {chat['question']}")
    st.markdown(f"  **답변:** {chat['answer']}")