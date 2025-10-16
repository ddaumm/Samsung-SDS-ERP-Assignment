from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from config import OPENAI_API_KEY
from tools import create_rag_tool, create_google_search_tool, create_website_opener_tool

tools = [
    create_rag_tool('detailed_graduation_rules.pdf', 'utf-8', 300, 100, 'detailed_graduation_rules_retriever', '서울대학교 산업공학과 졸업을 위한 졸업 요건에 대한 세부 사항을 물어보는 질문에 답할 때 사용'),
    create_google_search_tool(),
    create_website_opener_tool()
]

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0
)

prompt = hub.pull('hwchase17/react-chat')

agent = create_react_agent(llm, tools, prompt)

# ConversationBufferMemory vs ConversationSummaryMemory (추후, 결정하기)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    memory=memory,
    # ReAct 과정 출력 여부
    verbose=True,
    # 출력 형식 오류 자동 처리 및 복구 여부
    handling_parsing_erros=True
)