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
import time
from bar import single_bar_all, single_bar_time, single_bar_accuracy
import httpx

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

FASTAPI_URL1 = os.getenv('FASTAPI_URL1')
FASTAPI_URL2 = os.getenv('FASTAPI_URL2')
FASTAPI_URL3 = os.getenv('FASTAPI_URL3')
FASTAPI_URL4 = os.getenv('FASTAPI_URL4')
FASTAPI_URL_LOCAL = os.getenv('FASTAPI_URL_LOCAL')

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
            
        if model == "bart":
            # url2 = f"{FASTAPI_URL2}/summarize"
            # headers2 = {"Content-Type": "application/json", "titles": input_title}
            # summaries = requests.post(url2, json={"title": input_title,"texts": texts}, headers=headers2)
            # sum_list = summaries.json().get("data", "")
            # bart_time = summaries.json().get("bart_time", "")
            # st.write(f"요약 시간: {int(bart_time // 60)}분 {(bart_time % 60):.2f}초")
            # st.write("요약된 단락:")
            # for i, summary in enumerate(sum_list):
            #     st.write(f"{summary}")
            #     st.write("---")
            
            headers2 = {"Content-Type": "application/json", "titles": input_title}
            print(FASTAPI_URL_LOCAL)

            response1 = requests.post(f"{FASTAPI_URL_LOCAL}/execute_summary1", json={"title": input_title,"texts": texts}, headers=headers2)

            sum_list = response1.json().get("data", "")

            st.write("요약된 단락:")
            st.write(f"{sum_list}")
        elif model == "flan":
            
            response2 = requests.get(f"{FASTAPI_URL_LOCAL}/execute_summary2", params={"title": input_title})
            
            sum_list = response2.json().get("data", "")
            st.write("요약된 단락:")
            st.write(f"{sum_list}")

        elif model == "mean":
            response3 = requests.get(f"{FASTAPI_URL_LOCAL}/execute_summary3", params={"title": input_title})
            sum_list = response3.json().get("data", "")
            st.write("요약된 단락:")
            st.write(f"{sum_list}")
    else:
        st.write("PDF 파일를 업로드하세요.")
        

    
def extract_time(input_title):
    if input_title:
        time1_response = requests.get(f"{FASTAPI_URL1}/getTime1?title={input_title}")
        time2_response = requests.get(f"{FASTAPI_URL1}/getTime2?title={input_title}")
        time3_response = requests.get(f"{FASTAPI_URL1}/getTime3?title={input_title}")

        time1 = time1_response.json().get("data", "0")
        time2 = time2_response.json().get("data", "0")
        time3 = time3_response.json().get("data", "0")

        # 빈 문자열 또는 None 값을 0으로 변환
        time1 = int(time1) if str(time1).isdigit() else 0
        time2 = int(time2) if str(time2).isdigit() else 0
        time3 = int(time3) if str(time3).isdigit() else 0
        # 가장 큰 값으로 나누어 백분율로 변환
        max_time = time1 + time2 + time3
        if max_time > 0:
            time1_percent = (time1 / max_time) * 100
            time2_percent = (time2 / max_time) * 100
            time3_percent = (time3 / max_time) * 100
        else:
            time1_percent = time2_percent = time3_percent = 0
        print(time1_percent)
        return time1_percent, time2_percent, time3_percent
    
def match_count(input_title):
    get_bert = requests.get(f"{FASTAPI_URL1}/getbertKeyword?title={input_title}")
    res_bert = get_bert.json().get("resultCode", "")

    get_rank = requests.get(f"{FASTAPI_URL1}/getrankKeyword?title={input_title}")
    res_rank = get_rank.json().get("resultCode", "")

    if res_bert == 200 and res_rank == 200:
        split_keywords1 = get_bert.json().get("data", [])
        split_keywords2 = get_rank.json().get("data", [])

    elif res_bert == 200 and res_rank == 404:
        split_keywords1 = get_bert.json().get("data", [])
        split_keywords2 = requests.get(f"{FASTAPI_URL4}/TextRank_Keyword?title={input_title}").json().get("data", [])

    elif res_bert == 404 and res_rank == 200:
        split_keywords1 = requests.get(f"{FASTAPI_URL4}/Bert_Keyword?title={input_title}").json().get("data", [])
        split_keywords2 = get_rank.json().get("data", [])

    else:
        split_keywords1 = requests.get(f"{FASTAPI_URL4}/Bert_Keyword?title={input_title}").json().get("data", [])
        split_keywords2 = requests.get(f"{FASTAPI_URL4}/TextRank_Keyword?title={input_title}").json().get("data", [])

    sum1 = requests.get(f"{FASTAPI_URL1}/getSummary1?title={input_title}").json().get("data", "")
    sum2 = requests.get(f"{FASTAPI_URL1}/getSummary2?title={input_title}").json().get("data", "")
    sum3 = requests.get(f"{FASTAPI_URL1}/getSummary3?title={input_title}").json().get("data", "")
    print("sum1 출력해봅시다.")
    bert_count1, bert_count2, bert_count3 = 0, 0, 0
    rank_count1, rank_count2, rank_count3 = 0, 0, 0
    print(f"키워드 1: {split_keywords1}, 키워드 2: {split_keywords2}")
    for key in split_keywords1:
        if key in sum1:
            print("sum1", key)
            bert_count1 += 1
        if key in sum2:
            print("sum2", key)
            bert_count2 += 1
        if key in sum3:
            print("sum3", key)
            bert_count3 += 1

    for key in split_keywords2:
        if key in sum1:
            rank_count1 += 1
        if key in sum2:
            rank_count2 += 1
        if key in sum3:
            rank_count3 += 1

    print(bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3)
    print(len(split_keywords1), len(split_keywords2))
    bert_count1 = int((bert_count1 / len(split_keywords1)) * 100)
    bert_count2 = int((bert_count2 / len(split_keywords1)) * 100)
    bert_count3 = int((bert_count3 / len(split_keywords1)) * 100)

    rank_count1 = int((rank_count1 / len(split_keywords2)) * 100)
    rank_count2 = int((rank_count2 / len(split_keywords2)) * 100)
    rank_count3 = int((rank_count3 / len(split_keywords2)) * 100)

    print(bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3)
    return bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3
    
    
def count_percentage(bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3):
    count_percent1 = int((bert_count1 + rank_count1) / 2)
    count_percent2 = int((bert_count2 + rank_count2) / 2)
    count_percent3 = int((bert_count3 + rank_count3) / 2)
    return count_percent1, count_percent2, count_percent3

# ------------- 사이드바 화면 구성 -----------------------
with st.sidebar:
    st.title('Summarization')
    sum_model = ["bart-large-cnn","flan-t5-3b-summarizer", "flan-semantic"]
    selected_sum_model = st.selectbox("Please select the PDF summary model.", sum_model)

    if selected_sum_model == "bart-large-cnn":
        st.caption('https://huggingface.co/facebook/bart-large-cnn \
                   bart-large-cnn 허깅페이스 모델을 사용하여 텍스트 요약을 수행합니다. \
                   이 모델의 차이점은 모델의 크기가 500MB로 적은 disk로도 요약이 가능합니다. \
                   그러나 모델의 크기가 작은만큼 AI 학습이 효과가 다른 모델에 비해 떨어질 수 있습니다.')
    elif selected_sum_model == "flan-t5-3b-summarizer":
        st.caption('https://huggingface.co/jordiclive/flan-t5-3b-summarizer')
        st.caption('flan-t5-3b-summarizer 허깅페이스 모델을 사용하여 텍스트 요약을 수행합니다. \
                   ordiclive의 flan-t5-3b-summarizer 모델은 Google의 flan-t5-xl을 요약 작업에 맞게 튜닝한 모델입니다. \
                   이 모델은 xsum, wikihow, cnn_dailymail, samsum, scitldr, billsum 등의 다양한 데이터셋으로 학습되었으며, \
                   학술적 및 일반적인 요약 작업에 적합합니다. 사용자는 소스 텍스트에 특정 지시를 추가하여 요약 유형을 제어할 수 있습니다. \
                   이 모델은 최대 512개의 토큰을 처리하고, 최대 150개의 토큰으로 요약을 생성합니다.')

    elif selected_sum_model == "flan-semantic":
        st.caption('https://huggingface.co/jordiclive/flan-t5-3b-summarizer')
        st.caption('bart-large-cnn 허깅페이스 모델과 의미론적 요약 방법을 결합해 텍스트 요약을 수행합니다. \
                   `flan-t5-3b-summarizer` 모델과 의미론적 요약 방법을 결합한 방식은, \
                   먼저 텍스트에서 주요 문장을 추출한 후, 이를 `flan-t5-3b` 모델로 요약하는 것입니다. \.')
    chart = ["Choose Type","All", "Time", "Accuracy"]
    selected_chart = st.selectbox("Please select the chart.", chart)

    st.markdown("---")

    st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" \
            alt="Streamlit logo" height="16">&nbsp by',unsafe_allow_html=True)

    st.markdown(
            '<h6> \
            <a href="https://github.com/raminicano">@raminicano</a> \
            <a href="https://github.com/sinzng">@sinzng</a> \
            <a href="https://github.com/wjdguswn1203">@wjdguswn1203</a> \
            <a href="https://github.com/chang558">@chang558</a> \
            </h6>',
            unsafe_allow_html=True,
        )

    st.markdown('<h5></h5>', unsafe_allow_html=True)



# ------------- 메인 화면 구성 --------------------------  

if "input_title" not in st.session_state:
    st.session_state.input_title = ""
    
if selected_chart == "Choose Type":
    if selected_sum_model == "bart-large-cnn":
        st.header('Summary with bart-large-cnn')
        model = "bart"
    elif selected_sum_model == "flan-t5-3b-summarizer":
        st.header('Summary with flan-t5-3b-summarizer')
        model = "flan"
    elif selected_sum_model == "flan-semantic":
        st.header('Summary with flan-semantic')
        model = "mean"

    input_title = st.text_input("논문의 제목을 입력하세요.")

    upload_file = st.file_uploader("PDF 파일를 업로드하세요.", type="pdf")

    # 컬럼을 사용하여 버튼을 오른쪽에 배치
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col3:
        clicked_sum_model = st.button('PDF 문서 요약')

    if clicked_sum_model:
        summarize_PDF_file(upload_file, model, input_title)
else:    
    input_title = st.text_input("논문의 제목을 입력하세요.") 

# 선택한 유형에 따른 그래프 표시 
if selected_chart == "All":
    if input_title : 
        time1_percent, time2_percent, time3_percent = extract_time(input_title)
        bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3 = match_count(input_title)
        count_percent1, count_percent2, count_percent3 = count_percentage(bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3)
        single_bar_all(time1_percent, time2_percent, time3_percent, count_percent1, count_percent2, count_percent3)
elif selected_chart == "Time":
    if input_title : 
        time1_percent, time2_percent, time3_percent = extract_time(input_title)
        single_bar_time(time1_percent, time2_percent, time3_percent)
elif selected_chart == "Accuracy":
    if input_title : 
        bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3 = match_count(input_title)
        single_bar_accuracy(bert_count1, bert_count2, bert_count3, rank_count1, rank_count2, rank_count3)