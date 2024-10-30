import streamlit as st
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai

# Streamlit 설정
st.title("ChatBot")
st.write("아래에 질문을 입력하고 '전송' 버튼을 클릭하세요.")

# API 키와 LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key="OPENAPIKEY")
client = openai.OpenAI(api_key="OPENAPIKEY")

# 웹 페이지에서 데이터 로딩
loader = WebBaseLoader(
    web_paths=("https://spartacodingclub.kr/blog/all-in-challenge_winner",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent", "css-1hnxdb7")
        )
    ),
)
docs = loader.load()

# 텍스트 분할 및 임베딩 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(openai_api_key="OPENAPIKEY")  
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# 사용자 입력을 받는 UI
user_msg = st.text_input("질문을 입력하세요:", "")
if st.button("전송"):
    if user_msg:
        # Retrieve documents based on the question
        retrieved_docs = retriever.invoke(user_msg)

        # 포맷팅 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt 설정 및 응답 생성
        prompt = hub.pull("rlm/rag-prompt")
        user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_msg})

        # 결과 출력
        response = llm.invoke(user_prompt)
        st.write("답변:")
        st.write(response.content)
    else:
        st.write("질문을 입력해주세요!")


