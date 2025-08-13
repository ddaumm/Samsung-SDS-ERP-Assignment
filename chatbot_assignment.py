import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일의 값을 환경변수로 자동 등록
load_dotenv()

# openai api key 받기 -> client 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 모델 옵션
MODEL_OPTIONS = [
    'chatgpt-4o-latest',
    'gpt-4',
    'gpt-3.5-turbo'
]

selected_model = st.sidebar.radio("Select Model", MODEL_OPTIONS, index=2)

# 초기 상태(=대화 X)에서 초기 messages 생성
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role':'assistant', 'content':"자유롭게 질문을 입력해주세요!"}]

# 초기 messages 화면에 띄우기
for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

# 프롬프트 입력 받기
if prompt := st.chat_input("질문을 입력하세요."):
    # 프롬프트 기록 & 화면에 생성
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message('user').write(prompt)

    # 답변을 실시간으로 생성
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in client.chat.completions.create(
            model=selected_model,
            messages=st.session_state.messages,
            stream=True
        ):
            content = getattr(chunk.choices[0].delta, "content", "") or ""
            full_response += content
            message_placeholder.markdown(full_response +" ▌")

        message_placeholder.markdown(full_response)

    # 답변 기록
    st.session_state.messages.append({'role':'assistant', 'content':full_response})
