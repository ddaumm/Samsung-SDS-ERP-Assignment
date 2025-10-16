from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

from config import OPENAI_API_KEY
from tools import create_rag_tool, create_google_search_tool, create_website_opener_tool

tools = [
    create_rag_tool('simple_graduation_rules.txt', 'utf-8', 150, 50, 'simple_graduation_rules_retriever', '서울대학교 산업공학과 졸업을 위한 졸업 요건에 대한 간단한 질문에 답할 때 사용'),
    create_rag_tool('detailed_graduation_rules.pdf', 'utf-8', 300, 100, 'detailed_graduation_rules_retriever', '서울대학교 산업공학과 졸업을 위한 졸업 요건에 대한 복잡한 질문에 답할 때 사용'),
    create_google_search_tool(),
    create_website_opener_tool()
]

llm = ChatOpenAI(
    model='gpt-4-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)