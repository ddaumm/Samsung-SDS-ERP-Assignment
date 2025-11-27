import streamlit as st
import datetime
import os
import shutil
import hashlib

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler

# 로컬 모듈 임포트 (tools.py, config.py)
from config import OPENAI_API_KEY
from tools import (
    create_rag_tool, 
    create_google_search_tool,
    create_snulife_tool,
    create_gmail_tools,
    create_calendar_tools
)

# 1. 페이지 설정
st.set_page_config(
    page_title="SNU AI Agent",
    page_icon="🎓",
    layout="wide"
)

# 2. 스타일링 및 세션 상태 초기화
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 서울대학교 학습 도우미 AI Agent입니다. 무엇을 도와드릴까요?"}]

# [폴더 설정] 업로드 파일 저장소
UPLOAD_DIR = "./uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# 3. 에이전트 생성 함수 (핵심 로직)
def get_agent_executor(uploaded_file_path=None, model_name="gpt-3.5-turbo"):
    """
    모든 도구(기본 + 업로드된 파일 RAG)를 포함한 에이전트를 생성하여 반환합니다.
    """
    # (1) 기본 도구 목록 구성
    tools = []
    
    # 1-1. 기본 도구들 추가
    tools.append(create_google_search_tool())
    tools.append(create_snulife_tool())
    tools.extend(create_gmail_tools())
    tools.extend(create_calendar_tools())
    
    # 1-2. [동적 RAG] 업로드된 파일이 있다면 도구로 변환하여 추가
    if os.path.exists(UPLOAD_DIR):
        # 폴더 내 모든 파일 리스트업
        files = os.listdir(UPLOAD_DIR)
        
        for file_name in files:
            # 숨김 파일이나 시스템 파일 제외
            if file_name.startswith('.'): 
                continue
                
            file_path = os.path.join(UPLOAD_DIR, file_name)
            
            # PDF나 TXT인 경우에만 처리
            if file_path.lower().endswith(('.pdf', '.txt')):
                try:
                    tool_desc = f"사용자가 업로드한 문서('{file_name}')의 내용을 검색합니다. 이 문서에 대한 질문이 들어오면 이 도구를 사용하세요."
                    
                    safe_name = "doc_" + hashlib.md5(file_name.encode()).hexdigest()[:8]
                    
                    rag_tool = create_rag_tool(
                        file_path, 
                        retriever_name=safe_name, 
                        retriever_description=tool_desc
                    )
                    
                    if rag_tool:
                        tools.append(rag_tool)
                        
                except Exception as e:
                    print(f"파일 '{file_name}' 도구 생성 실패: {e}")
    
    # (2) LLM 설정
    llm = ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    # (3) 프롬프트 설정
    korean_react_template = """
        당신은 서울대학교 학생들을 돕는 유능한 AI 학습 도우미입니다. 

        현재 날짜(Today)는 {today} 입니다. 날짜와 관련된 질문(내일, 이번 주 등)에는 이 정보를 기준으로 계산하세요.

        사용 가능한 도구들은 다음과 같습니다:
        {tools}

        질문에 답하기 위해 다음 형식을 반드시 준수하세요:

        Question: 답변해야 할 입력 질문
        Thought: 무엇을 해야 할지 생각합니다. (반드시 한국어로 생각하세요)
        Action: 수행할 도구의 이름. [{tool_names}] 중 하나여야 합니다.
        Action Input: 도구에 입력할 값. (주의: 입력값 앞뒤에 따옴표(' 또는 ")를 절대 붙이지 말고 순수 텍스트만 입력하세요.)
        Observation: 도구의 실행 결과
        ... (이 Thought/Action/Action Input/Observation 과정은 필요한 만큼 반복될 수 있습니다)
        Thought: 이제 최종 정답을 알게 되었습니다.
        Final Answer: 원래 질문에 대한 최종 답변을 한국어로 작성합니다.

        [매우 중요 - 도구 사용 규칙]
        0. 도구는 꼭 사용할 필요가 없습니다. 당신이 스스로 답변할 수 있다면 도구를 사용하지 마세요.
        1. 인사말(안녕, 반가워 등), 일반적인 대화, 단순 지식 질문, 번역, 요약 등은 당신이 스스로 할 수 있는 작업입니다. 당신이 스스로 할 수 있는 작업의 경우에는 **절대로 도구를 사용하지 마세요.**
        2. 특히 'google_web_search' 도구는 최신 뉴스나 실시간 정보가 명확히 필요한 경우가 아니라면 사용하지 마세요.
        3. 도구를 사용할 필요가 없다면, Action 단계를 건너뛰고 바로 'Final Answer'를 작성하세요. 
        4. 절대로 'Action: None'이나 'Action: 없음'이라고 적지 마세요.

        시작합니다!

        이전 대화 기록:
        {chat_history}

        Question: {input}
        {agent_scratchpad}
        """
    
    # [핵심 수정] .partial()을 사용하여 {today} 변수에 오늘 날짜를 미리 주입합니다.
    # 이렇게 하면 나중에 invoke할 때 'today'를 따로 넘겨줄 필요가 없습니다.
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    prompt = PromptTemplate.from_template(korean_react_template).partial(today=today_str)

    # (4) 에이전트 생성
    agent = create_react_agent(llm, tools, prompt)
    
    # (5) 메모리 (세션 상태 활용)
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferWindowMemory(k=5, memory_key='chat_history', return_messages=True) # 최근 k=5개의 대화만 기억함
        
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=st.session_state.agent_memory,
        verbose=True,
        handle_parsing_errors=True
    )


# 4. 사이드바 UI (파일 업로드)
with st.sidebar:
    st.header("⚙️ 설정 및 제어")

    st.subheader("AI 모델 선택")
    selected_model = st.selectbox(
        "사용할 LLM 모델을 선택하세요:",
        ("gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"),
        index=0 # 기본값: gpt-3.5-turbo
    )

    st.markdown("---")

    st.subheader("📂 문서 업로드")
    uploaded_file = st.file_uploader("PDF나 TXT 파일을 업로드하세요", type=["pdf", "txt"])
    
    current_file_path = None

    if uploaded_file:
        # 파일을 서버에 저장
        current_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(current_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"'{uploaded_file.name}' 학습 완료!")
        st.info("이제 문서 내용에 대해 질문할 수 있습니다.")
    
    if st.button("🗑️ 대화 및 파일 초기화", type="primary"):
        # 메시지 기록을 초기 인사말로 초기화
        st.session_state.messages =  [{"role": "assistant", "content": "안녕하세요! 서울대학교 학습 도우미 AI Agent입니다. 무엇을 도와드릴까요?"}]
        
        # 에이전트 메모리 삭제
        if "agent_memory" in st.session_state:
            st.session_state.agent_memory.clear()

        # 화면 새로고침
        st.rerun()

# 5. 에이전트 로드/갱신 로직
# (파일이 업로드될 때마다 uploaded_docs 폴더 내용이 바뀌므로, 
#  단순히 파일명 비교보다는 '마지막 업데이트 시간'이나 '파일 목록'을 기준으로 갱신하는 것이 좋습니다.
#  여기서는 간단히 '업로드 이벤트가 발생했을 때' 강제로 갱신하도록 UI 로직을 맞춥니다.)

last_model = st.session_state.get("last_selected_model")
# 현재 폴더 내 파일 목록 해시 등을 사용하여 변경 감지 (여기선 단순화하여 업로드 시 즉시 반영)

# [수정] get_agent_executor 호출 시 더 이상 file_path를 넘기지 않습니다. (함수 내부에서 폴더 스캔)
if "agent_executor" not in st.session_state or last_model != selected_model or uploaded_file: 
    # uploaded_file이 있으면 -> 방금 새 파일이 들어왔으므로 갱신 필요
    with st.spinner("AI 에이전트의 지식을 업데이트하는 중입니다..."):
        st.session_state.agent_executor = get_agent_executor(model_name=selected_model)
        st.session_state.last_selected_model = selected_model


# 6. 메인 채팅 UI
st.title("🎓 SNU AI Agent")

# 이전 대화 출력
MAX_MESSAGES = 10  # 최대 메시지 수 제한
display_messages = st.session_state.messages[-MAX_MESSAGES:]

for msg in display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. 사용자 입력 처리
if prompt_input := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # 에이전트 실행
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        with st.spinner("응답 생성 중입니다..."):
            try:               
                # 에이전트 실행 (단일 input 키만 전달)
                response = st.session_state.agent_executor.invoke({
                    "input": prompt_input
                })
                
                answer = response["output"]
                st.markdown(answer)
                
                # 답변 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})