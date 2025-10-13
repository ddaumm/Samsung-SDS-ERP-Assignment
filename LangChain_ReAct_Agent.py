import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool

from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

'''API KEY LOADING'''
# Load api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


'''RAG TOOL'''
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
    embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    )

retriever = vectorstore.as_retriever()

# 4. RAG Tool Generation
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="graduation_rules_retriever",
    description="서울대학교 산업공학과의 졸업을 위한 이수 규정, 필요 학점, 전공 필수, 교양 학점 등 졸업 요건에 대한 질문에 답할 때 사용"
)


'''WEB SEARCH TOOL'''
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

search_tool = Tool(
    func=search.run,
    name='google_web_search',
    description='최신, 실시간 정보, 일반적인 정보 등 외부 웹 검색이 필요할 때 사용'
)


'''REACT AGENT'''
tools = [retriever_tool, search_tool]

llm = ChatOpenAI(
    model='gpt-4-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0

)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)