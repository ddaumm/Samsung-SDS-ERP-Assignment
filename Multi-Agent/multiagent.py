import os
import hashlib
import operator
from typing import Annotated, Sequence, TypedDict, List

# LangChain & LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 로컬 모듈
from config import OPENAI_API_KEY
from tools import (
    create_rag_tool, 
    create_google_search_tool,
    create_snulife_tool,
    create_gmail_tools,
    create_calendar_tools
)

def get_cot_system_prompt(agent_name, role_description):
    """
    기존 단일 에이전트의 CoT(ReAct) 스타일을 계승한 시스템 프롬프트
    """
    return f"""
    당신은 {agent_name}입니다. {role_description}
    
    질문에 답하기 위해 다음 형식을 반드시 준수하세요(CoT):
    
    1. **Thought**: 해야 할 작업에 대해 생각합니다. (사용자가 무엇을 원하는가? 어떤 도구가 필요한가?)
    2. **Action**: 필요한 도구를 선택합니다.
    3. **Action Input**: 도구에 입력할 내용입니다.
    4. **Observation**: 도구의 결과를 확인합니다.
    ... (필요한 만큼 반복)
    5. **Final Answer**: 최종 답변을 한국어로 작성합니다.

    [매우 중요 - 출력 규칙]
    1. 데이터 보존 필수: 도구 실행 결과(Observation)에 **'표(Table)', '링크(URL)', '리스트'**가 포함되어 있다면, **절대 요약하거나 생략하지 마세요.**
    2. 형식 유지: Observation에 있는 마크다운 표나 링크 형식을 Final Answer에 **그대로 복사해서 붙여넣으세요.**
    3. 친절한 설명: 표나 링크를 출력한 뒤에 부가적인 설명을 덧붙이세요.
    
    [주의사항]
    - 당신의 전문 분야가 아니거나 도구로 해결할 수 없으면 솔직히 모른다고 하세요.
    - 자신이 가진 도구 외의 다른 기능은 수행할 수 없다고 가정하세요.
    """

# [State] 그래프 상태 정의
class AgentState(TypedDict):
    # 대화 기록을 누적
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # 다음 순서(노드 이름)
    next: str

# 에이전트 노드 생성 함수
def create_agent_node(llm, tools, system_prompt):
    """특정 도구와 프롬프트를 가진 ReAct 에이전트 노드를 반환"""
    return create_react_agent(llm, tools, prompt=system_prompt)

# 그래프 빌더 함수
def build_graph(upload_dir="./uploaded_docs"):
    """
    설정된 모델과 업로드 폴더 경로를 받아 실행 가능한 LangGraph 객체를 반환합니다.
    """
    # 1. LLM 초기화
    llm_high = ChatOpenAI(model='gpt-5', api_key=OPENAI_API_KEY, temperature=0) # 고지능 LLM 모델 : Supervisor, Task, RAG 용
    llm_fast = ChatOpenAI(model='gpt-5-mini', api_key=OPENAI_API_KEY, temperature=0) # 고효율/속도 LLM 모델 : General 용
    
    # 2. 도구(Tools) 준비
    # 2-1. Task Tools (SNULife, Gmail, Calendar)
    task_tools = [create_snulife_tool()] + create_gmail_tools() + create_calendar_tools()
    
    # 2-2. RAG Tools (파일 스캔)
    rag_tools = []
    if os.path.exists(upload_dir):
        files = [f for f in os.listdir(upload_dir) if not f.startswith('.')]
        for file_name in files:
            file_path = os.path.join(upload_dir, file_name)
            # PDF/TXT만 처리
            if file_path.lower().endswith(('.pdf', '.txt')):
                try:
                    safe_name = "doc_" + hashlib.md5(file_name.encode()).hexdigest()[:8]
                    rag_tool = create_rag_tool(
                        file_path, 
                        retriever_name=safe_name, 
                        retriever_description=f"문서 '{file_name}' 내용 검색"
                    )
                    if rag_tool:
                        rag_tools.append(rag_tool)
                except Exception as e:
                    print(f"[Warn] RAG Tool 생성 실패 ({file_name}): {e}")

    # 2-3. General Tools (Google Search)
    general_tools = [create_google_search_tool()]

    # 3. Worker 노드 생성
    task_agent = create_agent_node(llm_high, task_tools, get_cot_system_prompt("Task_Worker", "업무 처리 담당입니다. Gmail, Calendar, Snulife 도구를 사용하여 요청을 처리하세요."))
    rag_agent = create_agent_node(llm_high, rag_tools, get_cot_system_prompt("RAG_Worker", "문서 분석 담당입니다. 제공된 문서 검색 도구를 사용하여 질문에 답하세요."))
    general_agent = create_agent_node(llm_fast, general_tools, get_cot_system_prompt("General_Worker", "일반 대화 담당입니다. 인사, 상식, 웹 검색(Google)을 처리하세요."))

    # 4. Supervisor (관리자) 노드 생성
    members = ["Task_Worker", "RAG_Worker", "General_Worker"]
    options = ["FINISH"] + members
    
    system_prompt = (
        "당신은 팀의 관리자(Supervisor)입니다. 아래 대화 내용을 보고 다음 작업자를 선택하세요.\n"
        f"옵션: {members}\n\n"
        "[판단 기준]\n"
        "- 메일 검색, 메일 전송, 일정 검색, 일정 생성, 일정 변경, 일정 삭제, 강의 족보 검색 -> Task_Worker\n"
        "- 첨부 문서 내용 질문 -> RAG_Worker\n"
        "- 그 외 일반 대화 -> General_Worker\n"
        "- 답변이 충분하거나 종료하고 싶음 -> FINISH\n"

        "먼저 사용자의 의도를 한 문장으로 생각(Thought)한 뒤, 가장 적절한 next를 선택하세요.\n" # CoT: 아주 짧게 의도만 파악하게 함

        "[매우 중요] 종료 조건\n"
        "- 최근 메시지에서 작업자가 'Final Answer'를 제시했거나, 사용자의 요청에 대한 답변이 완료되었다면 반드시 'FINISH'를 선택하세요.\n"
        "- 작업자가 이미 적절한 대답을 했는데도 다시 호출하지 마세요."
    )

    class RouteResponse(TypedDict):
        next: str

    def supervisor_node(state: AgentState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", f"위 대화 내용을 바탕으로 다음 순서를 정하세요. 옵션: {options}")
        ])
        chain = prompt | llm_high.with_structured_output(RouteResponse)

        return chain.invoke(state)

    # 5. 그래프 연결
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Task_Worker", task_agent)
    workflow.add_node("RAG_Worker", rag_agent)
    workflow.add_node("General_Worker", general_agent)

    # 노드 연결
    for member in members:
        workflow.add_edge(member, "Supervisor")

    # 진입 노드 설정
    workflow.set_entry_point("Supervisor")
    
    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "Task_Worker": "Task_Worker",
            "RAG_Worker": "RAG_Worker",
            "General_Worker": "General_Worker",
            "FINISH": END
        }
    )

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)