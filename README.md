# Samsung-SDS-ERP-Assignment
삼성SDS 연구장학 과제 / 사업지원그룹(ERP전략) / 지식 기반 멀티 에이전트 AI 시스템 Pilot

# 주제 : 학습 도움 AI Agent
- RAG : 강의 계획서 기반으로 주요 항목(주차별 강의 계획, 평가 비중, 교재, 연락처 등)의 내용 요약 및 확인
- 크롤링 : 알림(공지) 확인 및 내용 요약 
- Gmail : (메일 수신) 수업 관련 내용 확인 및 요약 / (메일 송신) 수업 관련 메일 전송 기능 & 템플릿 추천 기능
- Google Calendar : (일정 확인) 수업 관련 일정 확인 및 요약 / (일정 추가 및 편집) 수업 관련 일정 추가 및 편집
- Google Web Search : 교수/조교 연락처 웹 검색 (RAG 문서를 통해 정보를 구하기 어려운 경우)
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