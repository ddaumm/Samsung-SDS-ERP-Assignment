import os
from dotenv import load_dotenv
import langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 0. Load openai api key
load_dotenv()
openai_api_key = os.getenv("OPEN_API_KEY")

# 1. Load Data (txt 파일 문서)
loader = TextLoader(file_path='졸업이수규정.txt', encoding='utf-8')
data = loader.load()
data_content = data[0].page_content

# # logger
# print(f"원본 데이터 : \n{data_content}")
# print(f"원본 데이터 크기 : {len(data_content)}")

# 2. Text Split (Txt doc -> Small chunks(texts); RecursiveCharacterTextSplitter 활용)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 50,
    length_function = len
)

texts = text_splitter.split_text(data_content)

# # logger
# print(f"분할된 데이터 : \n{texts}")
# print(f"청크 수 : {len(texts)}")

# 3. Indexing (Texts -> Embedding -> Vector Store; Chroma, OpenAIEmbeddings, 유사도 기반 검색(similarity search) 활용)
embeddings_model = OpenAIEmbeddings()

vector_store = Chroma.from_texts(
    texts=texts, 
    embedding=embeddings_model
    )

# logger
docs = vector_store.similarity_search("학문의 세계 관련 내규를 설명해주세요.")
print(docs[0].page_content)