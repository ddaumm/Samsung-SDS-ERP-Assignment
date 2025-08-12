import streamlit as st
from openai import OpenAI
import os

# openai api key 받기 -> client 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 초기 상태(=대화 X)에서 초기 messages 생성
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role':'assistant', 'content':"자유롭게 질문을 입력해주세요!"}]

# 초기 messages 화면에 띄우기
for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

# 프롬프트 입력 받기
if prompt := st.chat_input("질문을 입력하세요."):
    # 프롬프트 화면에 띄우기
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message('user').write(prompt)

    # openai로부터 답변 받기
    completion = client.chat.completions.create(
        # 사용할 모델 설정
        model='chatgpt-4o-latest',

        # 받은 프롬프트를 messages로 설정
        messages = st.session_state.messages,

        # 답변의 최대 토큰 길이 설정
        # max_tokens = 100
    )

    # 답변 저장
    response = completion.choices[0].message.content

    # 답변 화면에 띄우기
    st.session_state.messages.append({'role':'assistant', 'content':response})
    st.chat_message('assistant').write(response)