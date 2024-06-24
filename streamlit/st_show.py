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

def summarize_PDF_file(pdf_file, model, input_title):

    if (pdf_file is not None):

        st.write("PDF 문서를 요약 중입니다. 잠시만 기다려 주세요.")

        # PDF 파일을 바이트 스트림으로 읽기
        file_bytes = pdf_file.read()
        url = f"{FASTAPI_URL3}/upload"
        headers = {"Content-Type": "application/pdf", "titles": input_title}
        response = requests.post(url, data=file_bytes, headers=headers)
        print("pdf sended")

        if response.status_code == 200:
            texts = response.json().get("texts", "")
            st.write("파일이 성공적으로 전송되었습니다.")

            # 스플리터 지정
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\\n\\n",  # 분할 기준
                chunk_size=2000,   # 청크 사이즈
                chunk_overlap=100, # 중첩 사이즈
            )

            split_texts = text_splitter.split_text(texts)
            st.write(len(split_texts), "단락이 있음.")
            
            # # 결과 출력
            # st.write("단락별 텍스트:")
            # for i, paragraph in enumerate(split_texts):
            #     st.write(f"단락 {i+1}:")
            #     st.write(paragraph)
            #     st.write("---")
        if model == "bart":
            url2 = f"{FASTAPI_URL2}/summarize"
            headers2 = {"Content-Type": "application/json", "titles": input_title}
            summaries = requests.post(url2, json={"texts": split_texts}, headers=headers2)
            sum_list = summaries.json().get("data", "")
            st.write("요약된 단락:")
            for i, summary in enumerate(sum_list):
                st.write(f"단락 {i+1}:")
                st.write(summary)
                st.write("---")
        elif model == "flan":
            print("구현중")
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

radio_selected_model = st.radio("PDF 문서 요약 모델", ["flan-t5-3b-summarizer","bart-large-cnn"], index=1, horizontal=True)

upload_file = st.file_uploader("PDF 파일를 업로드하세요.", type="pdf")

if radio_selected_model == "flan-t5-3b-summarizer":
    model = "flan"
elif radio_selected_model == "bart-large-cnn":
    model = "bart"

clicked_sum_model = st.button('PDF 문서 요약')

if clicked_sum_model:
    summarize_PDF_file(upload_file, model, input_title)
