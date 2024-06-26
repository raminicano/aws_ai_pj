from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Form, Request
import requests
from typing import List, Dict, Any
from pydantic import BaseModel
import os
import tempfile
from langchain.text_splitter import CharacterTextSplitter
from nltk.corpus import stopwords
import string
import nltk
import torch
from transformers import pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime
import httpx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings




FASTAPI_URL1 = os.getenv('FASTAPI_URL1')

executor = ThreadPoolExecutor(max_workers=3)
# cuda == gpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

app = FastAPI()


nltk.download('stopwords')
stop_words = stopwords.words('english')

@app.get("/Hello")
async def hello():
    return {"resultCode": 200, "data": "Hello World"}



## summarizer1

model_name1 = "facebook/bart-large-cnn"
model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name1)
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)

model1.to(device)
print(model1)

## summarizer2

# model_name2 = "jordiclive/flan-t5-3b-summarizer"
# model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name2)
# tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

# model2.to(device)
# print(model2)

summarizer = pipeline("summarization", "jordiclive/flan-t5-3b-summarizer", torch_dtype=torch.bfloat16, device="mps")



def extract_key_sentences(text, num_sentences=7):
    sentences = sent_tokenize(text)
    if len(sentences) < num_sentences:
        num_sentences = len(sentences)
    tfidf = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf, tfidf)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    key_sentences = " ".join([ranked_sentences[i][1] for i in range(num_sentences)])
    return key_sentences

def summarize_text(text, prompt=f"Here are some of the documents. Please summarize the main contents based on this document list. Answer:", max_length=512, min_length=30):
    full_text = f"{prompt} {text}"
    results = summarizer(
        full_text,
        num_beams=5,
        min_length=min_length,
        no_repeat_ngram_size=3,
        truncation=True,
        max_length=max_length,
    )
    return results[0]['summary_text']

def preprocess(text):
    # -\n을 빈 문자열로 대체
    text = text.replace("-\n", "")
    # \n을 공백으로 대체
    text = text.replace("\n", " ")
    # 소문자로 변환
    text = text.lower()

    # 불용어 제거 및 토큰화
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)



def summarize_paragraph(paragraph, model, tokenizer):
    try:
        inputs = tokenizer(paragraph, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = model.generate(inputs["input_ids"], max_length=512, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Summary is okey\n")
        return summary
    except Exception as e:
        print(f"Error summarizing paragraph: {e}")
        return paragraph


@app.post('/execute_summary1')
async def summarize(request: Request):
    # 경고 메시지 무시
    warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
    # 요청 보내기 전의 시간 기록
    bart_start_time = datetime.now()

    body = await request.json()
    title = body.get("title", "")
    texts = body.get("texts", [])

    # 스플리터 지정
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\\n\\n",  # 분할 기준
        chunk_size=2000,   # 청크 사이즈
        chunk_overlap=100, # 중첩 사이즈
    )

    split_texts = text_splitter.split_text(texts)

    filtered_docs = []
    for text in split_texts:
        if "References" in text:
            text = text.split("References")[0]
            filtered_docs.append(text)
            break
        filtered_docs.append(text)

    # 각 문서에 대해 전처리 수행
    processed_docs = [preprocess(doc) for doc in filtered_docs]

    summaries = []
   

    # summary 존재 여부 확인
    get_summary = requests.get(f"{FASTAPI_URL1}/getSummary1?title={title}")
    res = get_summary.json().get("resultCode", "")
    if res == 200:
        res = get_summary.json().get("data", "")
        # summaries = res.split("\n")
        return {"resultCode" : 200, "data" : res}
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(summarize_paragraph, paragraph, model1, tokenizer1) for paragraph in processed_docs]
            for future in futures:
                summaries.append(future.result())
            # 요청 받아온 시간 기록
        bart_end_time = datetime.now()
        bart_time = int((bart_end_time - bart_start_time).seconds)

        if summaries:
            url = f"{FASTAPI_URL1}/saveSummary1"
            summaries_string = "\n".join(summaries)
            save_sum = requests.post(url, json={"title": title,"text": summaries_string})
            get_time = requests.get(f"{FASTAPI_URL1}/getTime1?title={title}").json().get("resultCode", 0)
            if get_time == 200:
                bart_time = requests.get(f"{FASTAPI_URL1}/getTime1?title={title}").json().get("data", 0)
            else:
                save_time = requests.get(f"{FASTAPI_URL1}/saveTime1?title={title}&time={bart_time}")
            return {"resultCode" : 200, "data" : summaries_string, "bart_time" : bart_time}
        else:
            return {"resultCode" : 404, "data" : summaries_string, "bart_time" : bart_time}

@app.get("/execute_summary2")
async def execute_summary2(title: str = Query(..., description="Title of the paper to summarize")):
    try:
        async with httpx.AsyncClient() as client:
            
            check = await client.get(f"{FASTAPI_URL1}/getSummary2?title={title}")
            if check.json()["resultCode"] == 200:
                return {"resultCode": 200, "data": check.json()["data"]}
            else:
                #full_text 가져오기
                response = await client.get(f"{FASTAPI_URL1}/getFullText?title={title}")
                start_time = datetime.now()
                full_text = response.json()["data"]
                # 스플리터 지정
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    separator="\\n\\n",  # 분할 기준
                    chunk_size=2000,   # 청크 사이즈
                    chunk_overlap=100, # 중첩 사이즈
                )
                split_texts = text_splitter.split_text(full_text)
                
                filtered_docs = []
                for text in split_texts:
                    if "References" in text:
                        text = text.split("References")[0]
                        filtered_docs.append(text)
                        break
                    filtered_docs.append(text)

                # 각 문서에 대해 전처리 수행
                processed_docs = [preprocess(doc) for doc in filtered_docs]
                print("전처리 완료")

                # 병렬 요약 작업 수행
                loop = asyncio.get_running_loop()
                summaries = await asyncio.gather(
                    *[loop.run_in_executor(executor, summarize_text, doc) for doc in processed_docs]
                )

                # 요약 리스트를 문자열로 변환
                summary_str = " ".join(summaries)
                end_time = datetime.now()

                print(f"elapsed time: {end_time - start_time}")

                # saveTime2 API 호출
                time_response = await client.get(
                    f"{FASTAPI_URL1}/saveTime2",
                    params={"title": title, "time": int((end_time - start_time).seconds)}
                )

                # saveSummary2 API 호출
                save_response = await client.post(
                    f"{FASTAPI_URL1}/saveSummary2",
                    json={"title": title, "text": summary_str}
                )

                summary_code = save_response.json().get("resultCode", "")

                if summary_code == 200:
                    return {"resultCode": 200, "data": summary_str}
                else:
                    return {"resultCode": 500, "data": "Failed to save summary"}

    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


@app.get("/execute_summary3")
async def execute_summary3(title: str = Query(..., description="Title of the paper to summarize")):
    try:
        async with httpx.AsyncClient() as client:
            check = await client.get(f"{FASTAPI_URL1}/getSummary3?title={title}")
            if check.json()["resultCode"] == 200:
                return {"resultCode": 200, "data": check.json()["data"]}
            
            else:
                #full_text 가져오기
                response = await client.get(f"{FASTAPI_URL1}/getFullText?title={title}")
                start_time = datetime.now()
                full_text = response.json()["data"]
                # 스플리터 지정
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    separator="\\n\\n",  # 분할 기준
                    chunk_size=4000,   # 청크 사이즈
                    chunk_overlap=1000, # 중첩 사이즈
                )
                split_texts = text_splitter.split_text(full_text)
                
                filtered_docs = []
                for text in split_texts:
                    if "References" in text:
                        text = text.split("References")[0]
                        filtered_docs.append(text)
                        break
                    filtered_docs.append(text)

                # 각 문서에 대해 전처리 수행
                processed_docs = [preprocess(doc) for doc in filtered_docs]

                # 주요 문장 추출
                key_sentences = [extract_key_sentences(doc) for doc in processed_docs]



                # 병렬 요약 작업 수행
                loop = asyncio.get_running_loop()
                summaries = await asyncio.gather(
                    *[loop.run_in_executor(executor, summarize_text, doc) for doc in key_sentences]
                )

                # 요약 리스트를 문자열로 변환
                summary_str = " ".join(summaries)
                end_time = datetime.now()
                print("시간 계산")
                print(f"elapsed time: {end_time - start_time}")
                
                result_time = int((end_time - start_time).seconds)
                # saveTime3 API 호출
                time_response = await client.get(
                    f"{FASTAPI_URL1}/saveTime3",
                    params={"title": title, "time": result_time}
                )

                # saveSummary3 API 호출
                save_response = await client.post(
                    f"{FASTAPI_URL1}/saveSummary3",
                    json={"title": title, "text": summary_str}
                )

                summary_code = save_response.json().get("resultCode", "")

                if summary_code == 200:
                    return {"resultCode": 200, "data": summary_str}
                else:
                    return {"resultCode": 500, "data": "Failed to save summary"}
        
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3500)