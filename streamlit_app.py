import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import sessionmaker
from db_manager import Base, Case, engine
import re
import logging
import json
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API 설정
API_KEY = "D/spYGY15giVS64SLvtShZlNHxAbr9eDi1uU1Ca1wrqCiU+0YMwcnFy53naflVlg5wemikAYwiugNoIepbpexQ=="
API_URL = "https://api.odcloud.kr/api/15069932/v1/uddi:3799441a-4012-4caa-9955-b4d20697b555"

legal_terms_dict = {}
CACHE_FILE = "legal_terms_cache.json"

# 법률 용어 가져오기 함수
def get_legal_terms():
    global legal_terms_dict
    
    if legal_terms_dict:
        return legal_terms_dict
    
    if os.path.exists(CACHE_FILE):
        logging.info("저장된 용어 사전 불러오기")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            legal_terms_dict = json.load(f)
        logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 캐시에서 불러왔습니다.")
    else:
        logging.info("API에서 법률 용어 데이터 가져오기 시작")
        params = {
            "serviceKey": API_KEY,
            "page": 1,
            "perPage": 1000
        }
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                legal_terms_dict = {item['용어명']: item['설명'] for item in data['data']}
                logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 가져왔습니다.")
                
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(legal_terms_dict, f, ensure_ascii=False, indent=2)
                logging.info("법률 용어 데이터를 캐시 파일에 저장했습니다.")
            else:
                logging.error("API 응답에 'data' 키가 없습니다.")
        else:
            logging.error(f"API 요청 실패: 상태 코드 {response.status_code}")
    
    return legal_terms_dict

# 데이터베이스에서 판례 불러오기
@st.cache_data
def load_cases():
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    logging.info("데이터베이스에서 판례 데이터 로딩 시작")

    try:
        total_cases = session.query(Case).count()
        logging.info(f"총 {total_cases}개의 판례가 데이터베이스에 있습니다.")

        cases = list(session.query(Case))
        logging.info(f"총 {len(cases)}개의 판례를 로드했습니다.")
        return cases

    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return []

    finally:
        session.close()


# 벡터화 준비

# 법률 용어 하이라이트 처리
def highlight_legal_terms(text):
    terms = get_legal_terms()
    for term, explanation in terms.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        replacement = f'<span style="color: blue; font-weight: bold;" title="{explanation}">{term}</span>'
        text = re.sub(pattern, replacement, text)
    return text
@st.cache_data
def prepare_tfidf():
    cases = load_cases()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([case.summary for case in cases if case.summary])
    return vectorizer, tfidf_matrix, cases

# Streamlit 앱
def main():
    st.title("법률 판례 검색 시스템")

    vectorizer, tfidf_matrix, cases = prepare_tfidf()

    legal_fields = list(set(case.class_name for case in cases if case.class_name))
    legal_fields.append("잘모르겠습니다")

    selected_fields = st.multiselect("법률 분야 선택", legal_fields)
    user_input = st.text_area("상황을 입력하세요", height=150)

    if st.button("검색"):
        if user_input == '' or len(user_input) <= 3:
            st.error('검색어가 없거나 너무 짧습니다')
        else:
            logging.info(f"사용자 입력 받음, 선택된 법률 분야: {selected_fields}")
            
            if not selected_fields or "잘모르겠습니다" in selected_fields:
                logging.info("선택된 법률 분야가 없습니다. 모든 분야를 대상으로 검색합니다.")
                filtered_cases = cases
                filtered_tfidf_matrix = tfidf_matrix
            else:
                filtered_cases = [case for case in cases if case.class_name in selected_fields]
                filtered_tfidf_matrix = vectorizer.transform([case.summary for case in filtered_cases if case.summary])
            
            logging.info("유사 판례 검색 시작")
            user_vector = vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vector, filtered_tfidf_matrix)
            most_similar_idx = similarities.argmax()
            case = filtered_cases[most_similar_idx]
             
            logging.info("법률 용어 하이라이트 처리 중")
            st.subheader("검색 결과")
            st.markdown(f"**판례 번호:** {case.id}")
            st.markdown(f"**분야:** {case.class_name}")
            st.markdown("**요약:**")
            st.markdown(highlight_legal_terms(case.summary), unsafe_allow_html=True)
            if case.jdgmnQuestion:
                st.markdown("**판결 질문:**")
                st.markdown(highlight_legal_terms(case.jdgmnQuestion), unsafe_allow_html=True)
            if case.jdgmnAnswer:
                st.markdown("**판결 답변:**")
                st.markdown(highlight_legal_terms(case.jdgmnAnswer), unsafe_allow_html=True)
            # ... (나머지 결과 표시 코드는 그대로 유지)

if __name__ == '__main__':
    main()