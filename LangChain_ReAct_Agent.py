import os
from dotenv import load_dotenv
import langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool


# 0. Load openai api key
load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

# 1. Load Data (txt 파일 문서)
loader = TextLoader(file_path='졸업이수규정.txt', encoding='utf-8')
text = loader.load()
text_content = text[0].page_content

# 2. Text Split (Txt doc -> Small chunks(texts); RecursiveCharacterTextSplitter 활용)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 50,
    length_function = len
)

texts = text_splitter.split_text(text_content)

# 3. Indexing (Texts -> Embedding -> Vector Store; Chroma, OpenAIEmbeddings, 유사도 기반 검색(similarity search) 활용)
vectorstore = Chroma.from_texts(
    texts=texts, 
    embedding=OpenAIEmbeddings(api_key=API_KEY)
    )

retriever = vectorstore.as_retriever()

# 4. RAG Tool Generation
RAG_tool = create_retriever_tool(
    retrieve=retriever
    name="graduation_rules_retriever",
    description="서울대학교 산업공학과의 졸업을 위한 이수 규정, 필요 학점, 전공 필수, 교양 학점 등 졸업 요건에 대한 질문에 답할 때 사용"
)