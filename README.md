# Samsung-SDS-ERP-Assignment
삼성SDS 연구장학 과제 / 사업지원그룹(ERP전략) / 지식 기반 멀티 에이전트 AI 시스템 Pilot

# 주제 : 학습 도움 AI Agent
- RAG : 강의 계획서 기반으로 주요 항목(주차별 강의 계획, 평가 비중, 교재, 연락처 등)의 내용 요약 및 확인
- 크롤링 : 알림(공지) 확인 및 내용 요약 => Myetl 사이트 크롤링 불가능
- Gmail : (메일 수신) 수업 관련 내용 확인 및 요약 / (메일 송신) 수업 관련 메일 전송 기능 & 템플릿 추천 기능
- Google Calendar : (일정 확인) 수업 관련 일정 확인 및 요약 / (일정 추가 및 편집) 수업 관련 일정 추가 및 편집
- Google Web Search : 교수/조교 연락처 웹 검색 (RAG 문서를 통해 정보를 구하기 어려운 경우)
- SNU Life : 강의 후기 검색 -> 후기 내용 요약 / 족보 다운로드
- 관련 페이지 연결 : sugang.snu.ac.kr, snutt, everytime, snulife 등?
- 지도(굳이 필요한가...? 주제와의 관련성 떨어지긴 함) : 지도 검색 및 길찾기 기능 (카카오맵 api 활용 가능할 듯!)
- 보안 : getpass로 pw/id 받기 -> chat history와 분리

# Pipeline
1. 크롤링 공부 -> Tool 구현
2. Gmail, Google Calendar API 연동 및 활용법 공부 -> Tool 구현
3. Multi-Agent Orchestration 구축
4. 대화형 인터페이스(UI) 구축
5. Agent 개선 및 최적화(버퍼, 프롬프트 등)

# 참고 사항
- 메모리 -> bufferwindow or summarymemory 등으로 효율적 관리
- 도메인 관련, 에이전트의 역할 및 목적 부여 프롬프트 추가 (https://davi06000.tistory.com/173)
- test 진행하면서, 모델 적절히 설정 (3.5가 잘 수행 못하면 -> 4로 변경, embedding도 마찬가지)

**Deliverables**
- 프로젝트  결과  보고서 (PPT)
    - ① 문제정의 & 사용자 시나리오
    - ② 시스템 동작 구조
        - Ex. Task 1: 수집→정제→요약→표시
    - ③ **가설 카드 (3개)**
        - 아래 설명 참조
    - ⑤ 실제 작동 예시
    - ⑥ 결론
- 사용한 소스코드 프로젝트
- 프로젝트 최종 발표 영상 (15 min)

1. 과제 개요
    - 지식 기반 멀티 에이전트 AI 시스템 Pilot
    - 도메인 선정
    - 시스템 Architecture
2. 시스템 동작 구조
    - Langchain ReAct
    - RAG
    - 개별적으로 정의한 Tool: SNULIFE 웹 크롤링, Gmail/Calendar, Google Search
    - 챗봇 UI: Streamlit
3. 주요 이슈 및 해결 과정 (Challenges & Solutions)
    - 잘못된 도구 호출(사용) -> Negative Constraints & Description 등
    - 환각(Hallucination) -> Context Injection & Wrapper Function 등
3. 시스템 데모 영상
    - 시나리오 별 데모 영상
    - (복합 시나리오)