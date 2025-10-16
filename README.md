# Samsung-SDS-ERP-Assignment
삼성SDS 연구장학 과제 / 사업지원그룹(ERP전략) / 지식 기반 멀티 에이전트 AI 시스템 Pilot

# 추후 계획 정리
## streamlit 프론트엔드 구현
1. chat stream, chat history, 새로운 session 열기 기능 등 chatbot 과제 진행하면서 구현했던 기본적인 기능들 모두 구현
2. 파일 추가 -> RAG Tool 추가 생성할 수 있는 기능 구현
일단, 위 사항들만 우선적으로 구현 (나머지는 agent의 기능을 보완 or 추가 구현 후, 프론트엔드 구현)

## agent 보완
1. 메모리 -> bufferwindow or summarymemory 등으로 효율적 관리
2. 수강신청 홈페이지 or snutt 페이지 동적 크롤링을 통해, 조건에 맞는 강의 목록을 찾아서 반환하는 tool 추가 구현 (어떤 형식으로 반환할까? 표 or 그림 or 파일 or 말(=대화))
3. 도메인 관련, 에이전트의 역할 및 목적 부여 프롬프트 추가 (https://davi06000.tistory.com/173)
4. test 진행하면서, 모델 적절히 설정 (3.5가 잘 수행 못하면 -> 4로 변경, embedding도 마찬가지) 
5. 추가적으로 정의할 수 있는 Tool 구상/기존 Tool 보완 -> 구현 (실제, ai agent 서비를 확인해보자. 은행 ai 등)