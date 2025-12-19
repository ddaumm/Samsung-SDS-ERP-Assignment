import streamlit as st
import os
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

# 분리한 백엔드 모듈 임포트
from multiagent import build_graph

# 1. 페이지 설정
st.set_page_config(page_title="SNU Multi-Agent Chatbot", layout="wide")
st.markdown("""<style>.stChatMessage { border-radius: 10px; padding: 10px; }</style>""", unsafe_allow_html=True)

# 2. 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 서울대학교 멀티 에이전트 시스템입니다! 무엇을 도와드릴까요?"}]

# 메모리 생성
if "memory_saver" not in st.session_state:
    st.session_state.memory_saver = MemorySaver()

# Multi-turns 대화를 위한 고유 thread_id 생성
if 'thread_id' not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# 문서 저장소 설정
UPLOAD_DIR = "./uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 3. 사이드바 (설정)
with st.sidebar:
    st.subheader("문서 업로드")
    uploaded_file = st.file_uploader("PDF/TXT 파일", type=["pdf", "txt"])
    
    # 파일 업로드 시 처리
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"업로드 완료: {uploaded_file.name}")

        # 파일이 추가되었으므로 그래프를 재빌드하도록 플래그 설정
        st.session_state.graph_needs_update = True

    # if st.button("초기화", type="primary"):
    #     st.session_state.messages = [{"role": "assistant", "content": "대화가 초기화되었습니다."}]
    #     # 그래프 초기화(=삭제)
    #     if "agent_graph" in st.session_state:
    #         del st.session_state.agent_graph
        
        # thread_id 새롭게 생성
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 4. 그래프 로드 (캐싱 또는 상태 관리)
# 그래프가 없거나, 새로운 파일이 업로드 된 경우 새로 빌드
if "agent_graph" not in st.session_state or st.session_state.get("graph_needs_update"):
    with st.spinner("멀티 에이전트 시스템을 구동 중입니다..."):
        # (multi_agent.py 내부에서 모델을 지정했으므로 경로만 넘겨줌)
        st.session_state.agent_graph = build_graph(
            upload_dir=UPLOAD_DIR,
            checkpointer=st.session_state.memory_saver
            )
        
        st.session_state.graph_needs_update = False
        # print("Graph Rebuilt (Fixed Models)")

# 5. 채팅 UI 렌더링
st.title("SNU AI Agent")
# st.caption("Supervisor Architecture: Task / RAG / General")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. 사용자 입력 처리
if prompt_input := st.chat_input("무엇을 도와드릴까요?"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").markdown(prompt_input)

    with st.chat_message("assistant"):
        # LangGraph 입력 형식
        inputs = {"messages": [HumanMessage(content=prompt_input)]}

        # config 설정 (이 thread_id를 가진 이전 대화 기록을 불러옴)
        config = {'configurable' : {'thread_id': st.session_state.thread_id}}
        
        try:
            with st.spinner("에이전트들이 협업 중입니다..."):
                final_response = ""
                
                # 그래프 스트리밍 실행
                for output in st.session_state.agent_graph.stream(inputs, config=config):
                    for key, value in output.items():
                        # Supervisor 단계는 건너뛰고, Worker들의 작업만 로그로 표시
                        if key != "Supervisor":
                            if "messages" in value and len(value["messages"]) > 0:
                                # Worker의 마지막 메시지 확인
                                last_msg = value['messages'][-1]
                                content = last_msg.content
                                full_log = content # 최신 로그 업데이트

                                # Expander 내부에 전체 사고 과정 출력
                                with st.expander(f"{key} 작업 수행"):
                                    st.write(last_msg.content)
                        
                        # 최종 결과값 업데이트
                        if "messages" in value and len(value["messages"]) > 0:
                            final_response = value["messages"][-1].content

                if final_response:
                    import re

                    # "Final Answer" 또는 "Final Answer:" 뒤에 오는 텍스트를 잡는 정규식
                    match = re.search(r"Final Answer\s*:?\s*(.*)", final_response, re.DOTALL | re.IGNORECASE)

                    # 파싱 성공 -> 뒷부분만 출력
                    if match:
                        clean_answer = match.group(1).strip()
                        st.markdown(clean_answer)
                        st.session_state.messages.append({"role": "assistant", "content": clean_answer})
                    
                    # 파싱 실패 -> 그냥 전체 출력
                    else:
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})

                else:
                    st.warning("에이전트로부터 응답을 받지 못했습니다.")

        except Exception as e:
            st.error(f"시스템 오류가 발생했습니다: {e}")