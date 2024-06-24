import streamlit as st
import os
import tiktoken
import warnings
import requests
import textract
import re
import tempfile
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

FASTAPI_URL1 = os.getenv('FASTAPI_URL1')
FASTAPI_URL2 = os.getenv('FASTAPI_URL2')
FASTAPI_URL3 = os.getenv('FASTAPI_URL3')

def summarize_PDF_file(pdf_file, input_title):

    if (pdf_file is not None):

        st.write("PDF 문서를 요약 중입니다. 잠시만 기다려 주세요.")

        # PDF 파일을 바이트 스트림으로 읽기
        file_bytes = pdf_file.read()

        url = f"{FASTAPI_URL1}/upload"
        # 파일을 다시 file-like object로 만들어 requests에 전달
        files = {"file": (pdf_file.name, file_bytes, "application/pdf")}
        data = {"titles": input_title}
        
        response = requests.post(url, files=files, data=data)
        
        if response.json().get("resultCode") == 200:
            
            st.write("파일이 성공적으로 전송되었습니다.")



    else:
        st.write("PDF 파일를 업로드하세요.")
            

# ------------- 사이드바 화면 구성 -----------------------
st.sidebar.title('Menu')

# ------------- 메인 화면 구성 --------------------------  
st.title('Paragraph Extraction and Summarization from PDF')

st.header("요약 모델 설정 ")

st.write("요약 모델을 선택하세요.")

input_title = st.text_input("Paper title")
st.write("논문의 제목을 입력하세요.")

radio_selected_model = st.radio("PDF 문서 요약 모델", ["OpenAI","HuggingFace"], index=1, horizontal=True)

upload_file = st.file_uploader("PDF 파일를 업로드하세요.", type="pdf")

if radio_selected_model == "HuggingFace":
    model = "hf"
else:
    st.write("OpenAI 구상 중")

clicked_sum_model = st.button('PDF 문서 요약')
clicked_key_model = st.button('키워드 추출')


if clicked_sum_model:
    summarize_PDF_file(upload_file, input_title)
