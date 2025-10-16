# tools
'''agent가 사용할 수 있는 여러 개의 tool을 정의'''

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
import webbrowser

from config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID

# document RAG (txt, pdf 파일; 추후 다른 형식의 파일들에 대한 기능 추가 구현 예정)
def create_rag_tool(file_path, encoding, chunk_size, chunk_overlap, retriever_name, retriever_description, do_extract_images=False):
    file_name, file_ext = os.path.splitext(file_path)

    if os.path.exists("{file_name}_chroma_db"):
        vectorestore = Chroma(
            persist_directory=f"{file_name}_chroma_db",
            embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
            collection_name=file_name
        )

    else:
        try:
            if file_ext == ".txt":
                loader = TextLoader(file_path=file_path, encoding=encoding)

            elif file_ext == ".pdf":
                loader = PyPDFLoader(file_path=file_path, extract_images=do_extract_images)

            else:
                raise ValueError("지원하지 않는 파일 형식입니다. 'txt' 또는 'pdf' 파일을 사용해주세요.")
        
        except Exception as e:
            print("오류가 발생했습니다.", e)
            return None
        
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다. 파일 경로를 확인해주세요. \n파일 경로 : {file_path}")
            return None
        
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        documents = text_splitter.split_documents(document)

        # 추후 persist_directory 지정해서, 로컬에 저장하도록 하기 -> 이미 db가 있는 경우, 해당 db를 load 후에 retriever 생성하기
        vectorestore = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(
                model='text-embedding-3-small', 
                api_key=OPENAI_API_KEY
            ),
            persist_directory=f"{file_name}_chroma_db",
            collection_name=file_name
        )

    retriever = vectorestore.as_retriever()

    rag_tool = create_retriever_tool(
        retriever=retriever,
        name=retriever_name,
        description=retriever_description
    )

    return rag_tool


# websearch tool
def create_google_search_tool():
    search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

    google_search_tool = Tool(
        func=search.run,
        name='google_web_search',
        description='최신 정보, 실시간 정보, 일반적인 정보 등 웹 검색이 필요할 때 사용'
    )

    return google_search_tool

# open website tool
DEFINED_WEBSITES = {
    "서울대학교 홈페이지" : "https://www.snu.ac.kr",
    "서울대학교 수강 신청 사이트" : "https://www.sugang.snu.ac.kr",
    "서울대학교 강의 시간표" : "https://snutt.wafflestudio.com/",
    "서울대학교 학업 이수 현황" : "https://snugenie.snu.ac.kr/mypage/stdCmplCond.do",
    "서울대학교 졸업 시물레이션" : "https://snugenie.snu.ac.kr/mypage/stdCmplCond.do"
}

def open_url(website_name):
    url = DEFINED_WEBSITES.get(website_name)
    
    if url:
        webbrowser.open(url)
        return f"성공적으로 '{website_name}' [{url}]을 브라우저에서 열었음."
    else:
        available_sites = ", ".join(DEFINED_WEBSITES.keys())
        return f"오류 발생. 지정된 웹사이트 '{website_name}'을/를 찾을 수 없음. 가능한 옵션은 다음과 같음: {available_sites}" 

def create_website_opener_tool():
    website_opener_tool = Tool(
        func=open_url,
        name='website_opener',
        description=f'미리 지정된 웹사이트 중 하나를 사용자의 브라우저에 열 때 사용. 다음 중 하나의 웹사이트 이름을 입력해야 함: {", ".join(DEFINED_WEBSITES.keys())}'
    )

    return website_opener_tool