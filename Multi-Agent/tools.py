# 1. 표준 라이브러리
import base64
import datetime
import hashlib
import mimetypes
import os
import time
import webbrowser
import shutil # 파일/폴더 삭제용 추가
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

# 2. 서드 파티 라이브러리
# 2-1. Google API 관련
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 2-2. LangChain 관련
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2-3. Selenium 관련
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# 2-4. 기타 유틸리티
from bs4 import BeautifulSoup
from tabulate import tabulate

# 3. 로컬 모듈 (설정 파일)
from config import OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, SNULIFE_ID, SNULIFE_PW

# --- [전역 설정] ---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar"
]
KST = datetime.timezone(datetime.timedelta(hours=9))


# [Tool 1] RAG (문서 검색) 도구
def build_rag_db(file_path):
    """
    업로드된 파일로 벡터 DB를 구축하고, 해당 DB를 사용할 수 있는 Retriever를 반환.
    """
    file_name_original, file_ext = os.path.splitext(file_path)
    # 한글 파일명 등 오류 방지를 위한 해시 사용
    file_hash = hashlib.sha256(file_name_original.encode()).hexdigest()[:16]
    persist_dir = f"./chroma_db_{file_hash}"
    collection_name = f"col_{file_hash}"
    
    embedding = OpenAIEmbeddings(model='text-embedding-3-small', api_key=OPENAI_API_KEY)

    # 이미 DB가 있으면 로드, 없으면 새로 생성
    if os.path.exists(persist_dir):
        print(f"기존 벡터 DB 로드: {persist_dir}")
        try:
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding, collection_name=collection_name)
        except Exception as e:
            print(f"DB 로드 실패 (재생성 시도): {e}")
            shutil.rmtree(persist_dir, ignore_errors=True) # 손상된 DB 삭제
            return build_rag_db(file_path) # 재귀 호출로 다시 생성 시도
    else:
        print(f"새 벡터 DB 생성: {persist_dir}")
        try:
            if file_ext.lower() == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext.lower() == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                print(f"지원하지 않는 파일 형식: {file_ext}")
                return None
            
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
            documents = splitter.split_documents(docs)
            
            if not documents:
                print("문서 내용이 비어있습니다.")
                return None

            vectorstore = Chroma.from_documents(
                documents=documents, embedding=embedding, persist_directory=persist_dir, collection_name=collection_name
            )
        except Exception as e:
            print(f"RAG DB 생성 실패: {e}")
            return None

    return vectorstore.as_retriever()

def create_rag_tool(file_path, encoding='utf-8', chunk_size=300, chunk_overlap=60, retriever_name="doc_search", retriever_description="문서 검색", do_extract_images=False):
    """
    파일 경로를 받아 즉석에서 RAG Tool을 생성하여 반환합니다.
    (기존 create_rag_tool과 build_rag_db를 통합하여 활용)
    """
    # build_rag_db 함수를 사용하여 Retriever 생성
    retriever = build_rag_db(file_path)
    
    if not retriever:
        return None
        
    return create_retriever_tool(
        retriever,
        name=retriever_name,
        description=retriever_description
    )


# Google API Service Authenticator
def get_google_service(service_name, version):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except:
                flow = InstalledAppFlow.from_client_secrets_file('google_oauth_credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('google_oauth_credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        return build(service_name, version, credentials=creds)
    except HttpError as e:
        print(f"{service_name} 서비스 빌드 실패 : {e}")
        return None


# Gmail 관련 함수
def list_emails_by_keyword_and_date(service, start_date, end_date, keyword, message_num=10):
    email_list = []
    try:
        date_start_kst = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=KST)
        date_end_kst = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=KST) + datetime.timedelta(days=1)
        st_timestamp = int(date_start_kst.timestamp())
        end_timestamp = int(date_end_kst.timestamp())

        query = f"(subject:({keyword}) OR body:({keyword})) -in:sent after:{st_timestamp} before:{end_timestamp}"
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])

        if not messages: return "해당 키워드를 포함하는 메일이 존재하지 않습니다."
        
        for msg in messages[:message_num]:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = msg_data['payload']
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "제목 없음")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "발신인 불명")

            body_data = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        body_data = part.get('body', {}).get('data', '')
                        break
                    elif part['mimeType'] == 'text/html':
                        body_data = part.get('body', {}).get('data', '')
            elif 'body' in payload:
                body_data = payload['body'].get('data', '')

            decoded_body = "본문 없음"
            if body_data:
                try:
                    decoded_html = base64.urlsafe_b64decode(body_data.encode('ASCII')).decode('utf-8')
                    soup = BeautifulSoup(decoded_html, 'html.parser')
                    decoded_body = soup.get_text(separator=' ', strip=True)[:200] + "..."
                except: decoded_body = "(디코딩 실패)"

            mail_link = f"https://mail.google.com/mail/u/0/#inbox/{msg['id']}"
            email_list.append([subject, sender, decoded_body, mail_link])

        return tabulate(email_list, headers=["제목", "발신인", "본문요약", "링크"], tablefmt="grid")
    except Exception as e: return f"메일 검색 중 오류: {e}"

def send_email(service, to, subject, body_text, file_path=None):
    try:
        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = subject
        message.attach(MIMEText(body_text, "plain"))

        if file_path and os.path.exists(file_path):
            content_type, encoding = mimetypes.guess_type(file_path)
            if content_type is None: content_type = 'application/octet-stream'
            main_type, sub_type = content_type.split("/", 1)
            with open(file_path, 'rb') as fp:
                file_data = fp.read()
            attachment = MIMEBase(main_type, sub_type)
            attachment.set_payload(file_data)
            encoders.encode_base64(attachment)
            attachment.add_header('content-disposition', 'attachment', filename=os.path.basename(file_path))
            message.attach(attachment)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return f"✅ 메일 전송 성공! (수신: {to})"
    except Exception as e: return f"메일 전송 실패: {e}"


# Calendar 관련 함수
def list_events(service, start_date, end_date, keyword=None):
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=KST)
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=KST) + datetime.timedelta(days=1)
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()

        events_result = service.events().list(
            calendarId='primary', timeMin=start_iso, timeMax=end_iso, q=keyword,
            maxResults=10, singleEvents=True, orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        if not events: return "검색된 일정이 없습니다."

        res = []
        id_guide_list = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            try:
                if 'T' in start:
                    dt_obj = datetime.datetime.fromisoformat(start)
                    start_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                else: start_str = start + " (종일)"
            except: start_str = start
            
            summary = event.get('summary', '(제목 없음)')
            eid = event['id']
            res.append([start_str, summary, eid])
            id_guide_list.append(f"- '{summary}': {eid}")
        
        output = tabulate(res, headers=["시간", "일정명", "ID"], tablefmt="grid")
        output += "\n\n[수정/삭제용 ID 목록]\n" + "\n".join(id_guide_list)
        return output
    except Exception as e: return f"일정 검색 실패: {e}"

def create_event(service, title, start_str, end_str, description, overrides=None, location=None, attendees=None):
    event_body = {
        'summary': title, 'description': description, 'location': location,
        'attendees': [{"email": email} for email in (attendees or [])]
    }
    if overrides: event_body['reminders'] = {'useDefault': False, 'overrides': overrides}
    else: event_body['reminders'] = {'useDefault': False, 'overrides': [{'method': 'popup', 'minutes': 10}]}

    try:
        if len(start_str) > 10: 
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
            end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M").replace(tzinfo=KST)
            event_body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Seoul"}
            event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": "Asia/Seoul"}
        else:
            start_date_obj = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
            end_date_obj = datetime.datetime.strptime(end_str, "%Y-%m-%d").date() + datetime.timedelta(days=1)
            event_body["start"] = {"date": start_date_obj.isoformat()}
            event_body["end"] = {"date": end_date_obj.isoformat()}

        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return f"✅ 일정 생성 성공! 링크: {created.get('htmlLink')}"
    except Exception as e: return f"일정 생성 실패: {e}"

def modify_event(service, event_id, new_title=None, new_desc=None, new_start=None, new_end=None):
    try:
        try:
            event = service.events().get(calendarId='primary', eventId=event_id).execute()
        except HttpError as e:
            if e.resp.status == 404: return f"오류: Event ID '{event_id}'를 찾을 수 없습니다."
            raise e

        if new_title and new_title != "None": event["summary"] = new_title
        if new_desc and new_desc != "None": event["description"] = new_desc
        
        def parse_dt(s):
            s = s.strip()
            if 'T' in s: return datetime.datetime.fromisoformat(s).replace(tzinfo=KST), False
            elif len(s) > 10: return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=KST), False
            else: return datetime.datetime.strptime(s, "%Y-%m-%d").date(), True

        if new_start and new_start != "None":
            dt, is_all = parse_dt(new_start)
            event["start"] = {"date": dt.isoformat()} if is_all else {"dateTime": dt.isoformat(), "timeZone": "Asia/Seoul"}
        
        if new_end and new_end != "None":
            dt, is_all = parse_dt(new_end)
            event["end"] = {"date": dt.isoformat()} if is_all else {"dateTime": dt.isoformat(), "timeZone": "Asia/Seoul"}

        updated = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        return f"✅ 일정 수정 성공! 링크: {updated.get('htmlLink')}"
    except Exception as e: return f"일정 수정 실패: {e}"

def delete_event(service, event_id):
    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return f"✅ 일정 삭제 성공 (ID: {event_id})"
    except Exception as e: return f"일정 삭제 실패: {e}"

# SNULIFE 족보 검색 (Selenium)
def search_snulife_reference(query_str):
    if not SNULIFE_ID or not SNULIFE_PW:
        return "오류: .env 파일에 SNULIFE 정보가 없습니다."
    
    clean_query = query_str.strip().strip("'").strip('"')
    
    target_title, target_prof = clean_query.split('|') if '|' in clean_query else (clean_query, "")

    if not target_title or not target_prof:
        return "오류: 강의명과 교수명을 모두 입력해주세요."
    
    # 입력값 파싱
    target_title = target_title.strip().strip("'").strip('"')
    target_prof = target_prof.strip().strip("'").strip('"')

    # 드라이버 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--window-size=2560x1440')
    
    driver = webdriver.Chrome(service=Service('./chromedriver.exe'), options=chrome_options)
    
    try:
        driver.get("https://www.snulife.com/lecture")
        wait = WebDriverWait(driver, 5)

        # 로그인
        login_xpath = '//*[@id="__next"]/div/div[2]/div[2]/div[1]/div[2]/div[1]/div/form/div[1]/div'
        wait.until(EC.element_to_be_clickable((By.XPATH, login_xpath+'/input[1]'))).send_keys(SNULIFE_ID)
        driver.find_element(By.XPATH, login_xpath+'/input[2]').send_keys(SNULIFE_PW + Keys.ENTER)
        
        time.sleep(2)

        # 검색 (인덱스만 있는 경우는 이전 검색을 가정할 수 없으므로 에러 처리하거나 로직 보완 필요)
        if not target_title:
             return "강의명을 알 수 없어 바로 선택할 수 없습니다. '강의명 | 번호'로 입력해주세요."

        search_input_xpath = '//*[@id="__next"]/div/div[2]/div[2]/div[2]/div[1]/div[1]/div/form/input'

        s_box = driver.find_element(By.XPATH, search_input_xpath)
        s_box.send_keys(target_title)
        s_box.send_keys(Keys.ENTER)

        time.sleep(2)
        
        # 강의 목록 확인
        a_xpath = '//*[@id="__next"]/div/div[2]/div/div[1]/div[3]'
        lectures = driver.find_element(By.XPATH, a_xpath).find_elements(By.XPATH, './a')

        if not lectures:
            return "검색된 강의가 없습니다."

        final_index = -1

        for i, lect in enumerate(lectures, 1):
            try:
                prof_txt = lect.find_element(By.XPATH, './div[2]/div[1]/span[1]').text
                ref_num_txt = int(lect.find_element(By.XPATH, './div[2]/div[2]/span[3]').text[3:])
                
                if target_prof == prof_txt and ref_num_txt > 0:
                    final_index = i
                    break

            except: 
                continue

        # 족보 수집
        if final_index != -1:
            # lectures[final_index-1].click()
            driver.execute_script("arguments[0].click();", lectures[final_index-1])
            
            ref_btn = '//*[@id="__next"]/div/div[2]/div[7]/div/button[3]'

            button_elem = wait.until(EC.presence_of_element_located((By.XPATH, ref_btn)))
            driver.execute_script("arguments[0].click();", button_elem)

            time.sleep(2) # 탭 전환 안정화

            # 리스트 대기
            if int(driver.find_element(By.XPATH, '//*[@id="__next"]/div/div[2]/div[7]/div/button[3]/div').text) == 0:
                return "족보 파일이 없습니다."

            items_xpath = '//*[@id="__next"]/div/div[2]/div[8]/div/div'
            items = driver.find_elements(By.XPATH, items_xpath)
            data = []

            # 상위 5개만 수집
            for i in range(1, min(len(items)+1, 6)):
                try:
                    base = f'//*[@id="__next"]/div/div[2]/div[8]/div/div[{i}]'
                    sem = driver.find_element(By.XPATH, base+'/span').text # 연도/학기
                    name = driver.find_element(By.XPATH, base+'/div[1]/span[1]').text # 파일 제목

                    # 현재 페이지 URL 저장 -> 클릭 -> URL 변경 감지 -> 수집 -> 뒤로가기
                    list_page_url = driver.current_url # 족보 목록 페이지 URL
                    
                    # 다운로드 버튼(또는 링크) 클릭
                    download_btn = driver.find_element(By.XPATH, base+'/div[3]/button/div')
                    driver.execute_script("arguments[0].click();", download_btn)

                    try:
                        wait.until(EC.url_changes(list_page_url))
                        link = driver.current_url

                        driver.back()

                        # 목록 페이지가 다시 로드될 때까지 대기
                        wait.until(EC.presence_of_element_located((By.XPATH, items_xpath)))
                        
                        # 뒤로가기 안정화
                        time.sleep(1) 
                    
                    except TimeoutException:
                        link = list_page_url  # 링크 추출 실패 시 원래 페이지 URL로 대체
                    
                    markdown_link = f"[다운로드]({link})"

                    data.append([sem, name, markdown_link])
                
                except: 
                    continue
            
            return data
        
        else:
            return "조건에 맞는 강의의 족보를 찾지 못했습니다."

    except Exception as e:
        return f"스크래핑 오류: {e}"
    
    finally:
        driver.quit()

# [Tool Wrappers] Agent용 파싱 함수들
def gmail_search_wrapper(query_str):
    try:
        clean = query_str.strip().strip("'").strip('"')
        parts = [x.strip().strip("'").strip('"') for x in clean.split('|')]
        if len(parts) != 3: \
            return "오류: '시작일 | 종료일 | 키워드' 형식 필요"
        return list_emails_by_keyword_and_date(get_google_service('gmail', 'v1'), parts[0], parts[1], parts[2])
    except Exception as e: return f"오류: {e}"

def gmail_send_wrapper(query_str):
    try:
        clean = query_str.strip().strip("'").strip('"')
        parts = [x.strip().strip("'").strip('"') for x in clean.split('|', 2)]
        if len(parts) != 3: return "오류: '수신자 | 제목 | 본문' 형식 필요"
        return send_email(get_google_service('gmail', 'v1'), parts[0], parts[1], parts[2])
    except Exception as e: return f"오류: {e}"

def calendar_search_wrapper(query_str):
    try:
        clean = query_str.strip().strip("'").strip('"')
        parts = [x.strip().strip("'").strip('"') for x in clean.split('|')]
        keyword = parts[2] if len(parts) > 2 else None
        return list_events(get_google_service('calendar', 'v3'), parts[0], parts[1], keyword)
    except Exception as e: return f"오류: {e}"

def calendar_create_wrapper(query_str):
    try:
        clean = query_str.strip().strip("'").strip('"')
        parts = [x.strip().strip("'").strip('"') for x in clean.split('|')]
        if len(parts) < 3: return "오류: '제목 | 시작 | 종료 | 설명' 형식 필요"
        desc = parts[3] if len(parts) > 3 else "AI 생성 일정"
        return create_event(get_google_service('calendar', 'v3'), parts[0], parts[1], parts[2], desc)
    except Exception as e: return f"오류: {e}"

def calendar_modify_wrapper(query_str):
    try:
        clean = query_str.strip().strip("'").strip('"')
        parts = [x.strip().strip("'").strip('"') for x in clean.split('|')]
        if len(parts) < 2: return "오류: 최소한 'ID'는 필수"
        eid = parts[0]
        if " " in eid or len(eid) < 5: return "오류: 올바른 Event ID가 아닙니다."
        params = parts[1:] + [None] * (4 - len(parts[1:]))
        return modify_event(get_google_service('calendar', 'v3'), eid, params[0], params[3], params[1], params[2])
    except Exception as e: return f"오류: {e}"

def calendar_delete_wrapper(query_str):
    return delete_event(get_google_service('calendar', 'v3'), query_str.strip().strip("'").strip('"'))

# [Tool Creators] 최종 Tool 생성 함수들
def create_gmail_tools():
    return [
        Tool(name="gmail_search", func=gmail_search_wrapper, description=(
            "Gmail 검색. 입력: 'YYYY-MM-DD | YYYY-MM-DD | 키워드'"
            "주의: 키워드에는 'in:sent', 'subject:' 같은 연산자를 절대 포함하지 말고, 오직 찾고 싶은 단어만 입력하세요."
        )),
        Tool(name="gmail_send", func=gmail_send_wrapper, description="메일 전송. 입력: '수신자 | 제목 | 본문'")
    ]

def create_calendar_tools():
    return [
        Tool(name="calendar_search", func=calendar_search_wrapper, description="일정 검색. 입력: 'YYYY-MM-DD | YYYY-MM-DD | 키워드'"),
        Tool(name="calendar_add", func=calendar_create_wrapper, description="일정 추가. 입력: '제목 | 시작 | 종료 | 설명'"),
        Tool(name="calendar_modify", func=calendar_modify_wrapper, description="일정 수정. 입력: 'ID | 새제목 | 새시작 | 새종료 | 새설명'. ID 필수."),
        Tool(name="calendar_delete", func=calendar_delete_wrapper, description="일정 삭제. 입력: 'ID'. ID 필수로 입력받아야 함.")
    ]

def create_google_search_tool():
    search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
    return Tool(func=search.run, 
                name='google_web_search', 
                description=
                    "**[주의: 사용 금지 조건]** 다음의 경우에는 이 도구를 절대 사용하지 마세요:\n"
                "1. 단순한 지식 질문 (예: '수도가 어디야?', '이 단어 뜻이 뭐야?')\n"
                "2. 번역, 요약, 문장 다듬기\n"
                "3. 일상적인 대화나 인사\n\n"
                "**[사용 조건]** 오직 당신이 알 수 없는 **'최신 정보'**(오늘 날씨, 실시간 뉴스, 현재 주가 등)가 필요할 때만 사용하세요.")

def create_snulife_tool():
    return Tool(
        name="snulife_reference_search", 
        func=search_snulife_reference, 
        description=
            "SNULIFE에서 족보를 검색합니다. 입력 형식은 '강의명 | 교수 이름'입니다. 예: '경영과학1 | 홍성필'"
        )