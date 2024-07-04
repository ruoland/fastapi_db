import streamlit as st
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

# 메인 StreamLit 앱
def main():
    st.title("법률 사례 검색 시스템")

    # 사용자 입력
    user_input = st.text_area("상황을 설명해주세요")
    
    # 법률 분야 선택 (체크박스 대신 멀티셀렉트 사용)


    if st.button("검색"):
        if user_input == '' or len(user_input) <= 3:
            st.warning('검색어가 없거나 너무 짧습니다')
        else:
            # 검색 로직

            # 결과 표시
            st.subheader("가장 유사한 판례")


if __name__ == '__main__':
    main()