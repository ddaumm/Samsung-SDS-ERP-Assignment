from dotenv import load_dotenv
import os

import langchain
from langchain_openai import ChatOpenAI

print(f"lanchain 버전 : {langchain.__version__}")

# simple test
# .env 파일 로드
load_dotenv()

# 환경 변수 사용 -> api_key 받아오기
openai_api_key = os.getenv("OPEN_API_KEY")

try:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    response = llm.invoke("안녕하세요!!")
    print("설치 완료 - 정상 작동")
    print(f"응답 : {response.content}")

except Exception as e:
    print(f"설정 오류 : {e}")