from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File, Form
import requests
import feedparser
from typing import List, Dict, Any
from pydantic import BaseModel
import weaviate
import os
import weaviate.classes.config as wc
from weaviate.classes.query import Filter, MetadataQuery
import deepl
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from nltk.corpus import stopwords
import string
import nltk
import torch
from transformers import pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor



FASTAPI_URL1 = os.getenv('FASTAPI_URL1')

# executor = ThreadPoolExecutor(max_workers=3)


# nltk.download('stopwords')
# stop_words = stopwords.words('english')

# summarizer = pipeline("summarization", "jordiclive/flan-t5-3b-summarizer", torch_dtype=torch.bfloat16)


# def summarize_text(text, prompt=f"Here are some of the documents. Please summarize the main contents based on this document list. Answer:", max_length=512, min_length=5):
#     full_text = f"{prompt} {text}"
#     results = summarizer(
#         full_text,
#         num_beams=5,
#         min_length=min_length,
#         no_repeat_ngram_size=3,
#         truncation=True,
#         max_length=max_length,
#     )
#     return results[0]['summary_text']


class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published: str
    direct_link: str
    pdf_link: str
    category: str

class MetaResponse(BaseModel):
    resultCode: int
    data: List[Paper]

class SaveWeaResponse(BaseModel):
    resultCode: int
    data: Dict[str, str]


class TranslationRequest(BaseModel):
    text: str
    target_lang: str = 'KO'

class DataResponse(BaseModel):
    resultCode: int
    data: str

class TranslationPaperRequest(BaseModel):
    title: str
    text: str
    target_lang: str = 'KO'

class PaperRequset(BaseModel):
    title: str
    text: str

class SummaryRequest(BaseModel):
    text: str

class KeywordRequest(BaseModel):
    title: str
    keyword: list

app = FastAPI()

HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
DEEPL_AUTH_KEY = os.environ["DEEPL_AUTH_KEY"]


# def preprocess(text):
#     # -\n을 빈 문자열로 대체
#     text = text.replace("-\n", "")
#     # \n을 공백으로 대체
#     text = text.replace("\n", " ")
#     # 소문자로 변환
#     text = text.lower()

#     # 불용어 제거 및 토큰화
#     tokens = [word for word in text.split() if word not in stop_words]
#     return ' '.join(tokens)


client = weaviate.connect_to_local(
    headers={
        "X-HuggingFace-Api-Key": HUGGINGFACE_API_KEY,
    }
)

paper_collection = client.collections.get("Paper")
result_collection = client.collections.get("result")


class TranslationResponse(BaseModel):
    resultCode: int
    data: str

@app.get("/getMeta")
async def get_meta(searchword: str = Query(..., description="Search term for arXiv API")) -> Dict[str, Any]:
    text = searchword.replace(" ", "+")
    base_url = f"http://export.arxiv.org/api/query?search_query=ti:{text}+OR+abs:{text}&sortBy=relevance&sortOrder=descending&start=0&max_results=15"

    response = requests.get(base_url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from arXiv API")
    
    feed = feedparser.parse(response.content)
    papers = []

    for entry in feed.entries:
        link = entry.links[0]['href'] if entry.links else None
        pdf_link = entry.links[1]['href'] if len(entry.links) > 1 else None
        category = entry.arxiv_primary_category['term'] if 'arxiv_primary_category' in entry else None

        paper = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "published": entry.published,
            "direct_link": link,
            "pdf_link": pdf_link,
            "category": category
        }
        papers.append(paper)

    return {
        "resultCode": 200,
        "data": papers
    }

@app.post("/saveWea", response_model=SaveWeaResponse)
async def save_wea(meta_response: MetaResponse = Body(...)) -> SaveWeaResponse:
    papers = meta_response.data

    try:
        with paper_collection.batch.fixed_size(5) as batch:
            for paper in papers:
                # title 중복 확인
                response = paper_collection.query.fetch_objects(
                    filters=Filter.by_property("title").equal(paper.title),
                    limit=1
                )
                # object가 있으면 건너뛰기
                if response.objects:
                    continue
                
                properties = {
                    "title": paper.title,
                    "authors": paper.authors,
                    "summary": paper.summary,
                    "published": paper.published,
                    "direct_link": paper.direct_link,
                    "pdf_link": paper.pdf_link,
                    "category": paper.category,
                }

                batch.add_object(
                    properties=properties,
                )
        return SaveWeaResponse(resultCode=200, data={"message": "데이터 저장이 완료되었습니다."})
    except Exception as e:
        return SaveWeaResponse(resultCode=500, data={"message": str(e)})

@app.get("/searchKeyword")
async def search_keyword(searchword: str = Query(..., description="Search term for Weaviate db")) -> Dict[str, Any]:
    try:
        response = paper_collection.query.bm25(
            query=searchword,
            return_metadata=MetadataQuery(score=True),
            query_properties=["title", "authors", "summary"],
            limit=10
        )
        res = []
        # 오브젝트가 있으면
        if response.objects:
            for object in response.objects:
                res.append(object.properties) # 반환 데이터에 추가
            return {"resultCode" : 200, "data" : res}
        else:
            return {"resultCode" : 404, "data" : response}
    
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


@app.post("/translate_text", response_model=DataResponse)
async def translate_text_endpoint(request: TranslationRequest):
    try:
        translator = deepl.Translator(DEEPL_AUTH_KEY)
        result = translator.translate_text(request.text, target_lang=request.target_lang)
        return {"resultCode": 200, "data": result.text}
    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Translation failed")



# 번역해서 저장하는 api (update)
@app.post("/translate_text_paper", response_model=DataResponse)
async def translate_text_paper(request: TranslationPaperRequest):
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(request.title),
                limit=1
            )
        trans_summary = response.objects[0].properties["trans_summary"]
        
        if trans_summary:
            print("trans_summary is found")
            return {"resultCode": 200, "data": trans_summary}
        else:
            uuid = response.objects[0].uuid
            translator = deepl.Translator(DEEPL_AUTH_KEY)
            result = translator.translate_text(request.text, target_lang=request.target_lang)
            
            response.data.update(
                uuid=uuid,
                properties={
                    "title": request.title,
                    "trans_summary": result.text
                }
            )
            return {"resultCode": 200, "data": result.text}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}



# full_text 저장하는 api (add)
@app.post("/store_full_text", response_model=DataResponse)
async def store_full_text(request: PaperRequset):
    try:
        with result_collection.batch.fixed_size(1) as batch:
            response = result_collection.query.fetch_objects(
                    filters=Filter.by_property("title").equal(request.title),
                    limit=1,
                    return_properties=["title", "full_text"]
                )
            
            if response.objects:
                print("full_text is found")
                return {"resultCode": 200, "data": response.objects[0].properties["full_text"]}
            else:
                batch.add_object(
                    properties={
                        "title": request.title,
                        "full_text": request.text
                    }
                )
                return {"resultCode": 200, "data": request.text}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    

# full_text 존재하는지 확인용 api
@app.get("/getFullText", response_model=DataResponse)
async def get_full_text(title: str = Query(..., description="Search title for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "full_text"]
            )
        
        if response.objects:
            return {"resultCode": 200, "data": response.objects[0].properties["full_text"]}
        else:
            return {"resultCode": 404, "data": "full_text is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    

# summary1 존재하는지 확인용 api
@app.get("/getSummary1", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "summary1"]
            )
        summary = response.objects[0].properties["summary1"]
        if summary:
            return {"resultCode": 200, "data": summary}
        else:
            return {"resultCode": 404, "data": "summary is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    

# summary2 존재하는지 확인용 api
@app.get("/getSummary2", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "summary2"]
            )
        summary = response.objects[0].properties["summary2"]
        if summary:
            return {"resultCode": 200, "data": summary}
        else:
            return {"resultCode": 404, "data": "summary is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}



# summary3 존재하는지 확인용 api
@app.get("/getSummary3", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "summary3"]
            )
        summary = response.objects[0].properties["summary3"]
        if summary:
            return {"resultCode": 200, "data": summary}
        else:
            return {"resultCode": 404, "data": "summary is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


# summary1 저장하는 api
@app.post("/saveSummary1")
async def save_summary(request: PaperRequset):
    try:
        check = result_collection.query.fetch_objects(
            filters=Filter.by_property("title").equal(request.title),
            limit=1
        )
        uuid = check.objects[0].uuid

        result_collection.data.update(
            uuid=uuid,
            properties={
                "summary1": request.text
            }
        )
        return {"resultCode": 200, "data": "summary1 저장 성공"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}

# summary2 저장하는 api
@app.post("/saveSummary2")
async def save_summary(request: PaperRequset):
    try:
        check = result_collection.query.fetch_objects(
            filters=Filter.by_property("title").equal(request.title),
            limit=1
        )
        uuid = check.objects[0].uuid

        result_collection.data.update(
            uuid=uuid,
            properties={
                "summary2": request.text
            }
        )
        return {"resultCode": 200, "data": "summary2 저장 성공"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    
# summary1 저장하는 api
@app.post("/saveSummary3")
async def save_summary(request: PaperRequset):
    try:
        check = result_collection.query.fetch_objects(
            filters=Filter.by_property("title").equal(request.title),
            limit=1
        )
        uuid = check.objects[0].uuid

        result_collection.data.update(
            uuid=uuid,
            properties={
                "summary3": request.text
            }
        )
        return {"resultCode": 200, "data": "summary3 저장 성공"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    
@app.get("/delete_object")
async def delete_object(title: str = Query(..., description="Title of the paper to delete")):
    try:
        result_collection.data.delete_many(
            where=Filter.by_property("title").equal(title)
        )
        return {"resultCode": 200, "data": "Object deleted"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), titles: str = Form(...)):
    try:
        # PDF 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.file.read())
            tmp_file.flush()

            # PDF 파일에서 텍스트 추출
            loader = PyPDFLoader(tmp_file.name)
            document = loader.load()

            # 텍스트 분할 및 필터링
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\n\n",  # 분할 기준
                chunk_size=2000,   # 청크 사이즈
                chunk_overlap=500, # 중첩 사이즈
            )

            split_docs = text_splitter.split_documents(document)
            filtered_docs = []
            for doc in split_docs:
                if "References" in doc.page_content:
                    doc.page_content = doc.page_content.split("References")[0]
                    filtered_docs.append(doc.page_content)
                    break
                filtered_docs.append(doc.page_content)

            for i in range(len(filtered_docs)):
                file_name = f"split_doc_{i+1}.txt"

                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(filtered_docs[i])
                print(f"Document saved to {file_name}")

            return {"resultCode": 200, "data": "저장성공"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/getTime1", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "time1"]
            )
        time = response.objects[0].properties["time1"]
        time = str(time)
        if time:
            return {"resultCode": 200, "data": time}
        else:
            return {"resultCode": 404, "data": "time is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
 

@app.get("/getTime2", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "time2"]
            )
        time = response.objects[0].properties["time2"]
        time = str(time)
        if time:
            return {"resultCode": 200, "data": time}
        else:
            return {"resultCode": 404, "data": "time is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
    


@app.get("/getTime3", response_model=DataResponse)
async def get_summary(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "time3"]
            )
        time = response.objects[0].properties["time3"]
        time = str(time)
        if time:
            return {"resultCode": 200, "data": time}
        else:
            return {"resultCode": 404, "data": "time is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}

 # time1 저장하는 api



@app.get("/saveTime1", response_model=DataResponse)
async def save_time1(title: str, time: int):
     try:
         check = result_collection.query.fetch_objects(
             filters=Filter.by_property("title").equal(title),
             limit=1
         )
         uuid = check.objects[0].uuid

         result_collection.data.update(
             uuid=uuid,
             properties={
                 "time1": time
             }
         )
         time = str(time)
         return {"resultCode": 200, "data": time}
     except Exception as e:
         return {"resultCode": 500, "data": str(e)}
     



@app.get("/saveTime2", response_model=DataResponse)
async def save_time1(title: str, time: int):
     try:
         check = result_collection.query.fetch_objects(
             filters=Filter.by_property("title").equal(title),
             limit=1
         )
         uuid = check.objects[0].uuid

         result_collection.data.update(
             uuid=uuid,
             properties={
                 "time2": time
             }
         )
         time = str(time)
         return {"resultCode": 200, "data": time}
     except Exception as e:
         return {"resultCode": 500, "data": str(e)}
     

@app.get("/saveTime3", response_model=DataResponse)
async def save_time1(title: str, time: int):
     try:
         check = result_collection.query.fetch_objects(
             filters=Filter.by_property("title").equal(title),
             limit=1
         )
         uuid = check.objects[0].uuid

         result_collection.data.update(
             uuid=uuid,
             properties={
                 "time3": time
             }
         )
         time = str(time)
         return {"resultCode": 200, "data": time}
     except Exception as e:
         return {"resultCode": 500, "data": str(e)}
     

# keyword 조회
@app.get("/getbertKeyword", response_model=DataResponse)
async def get_keyword(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "bert_keywords"]
            )
        keyword = response.objects[0].properties["bert_keywords"]
        if keyword:
            return {"resultCode": 200, "data": keyword}
        else:
            return {"resultCode": 404, "data": "keyword is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


@app.get("/getrankKeyword", response_model=DataResponse)
async def get_keyword(title: str = Query(..., description="Search summary for Weaviate db")) -> Dict[str, Any]:
    try:
        response = result_collection.query.fetch_objects(
                filters=Filter.by_property("title").equal(title),
                limit=1,
                return_properties=["title", "rank_keywords"]
            )
        keyword = response.objects[0].properties["rank_keywords"]
        if keyword:
            return {"resultCode": 200, "data": keyword}
        else:
            return {"resultCode": 404, "data": "keyword is not found"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


# keyword 저장
@app.post("/savebertKeyword", response_model=DataResponse)
async def save_keyword(request: KeywordRequest):
    try:
        check = result_collection.query.fetch_objects(
            filters=Filter.by_property("title").equal(request.title),
            limit=1
        )
        uuid = check.objects[0].uuid

        result_collection.data.update(
            uuid=uuid,
            properties={
                "bert_keywords": request.keyword
            }
        )
        return {"resultCode": 200, "data": request.keyword}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}


@app.post("/saverankKeyword", response_model=DataResponse)
async def save_keyword(request: KeywordRequest):
    try:
        check = result_collection.query.fetch_objects(
            filters=Filter.by_property("title").equal(request.title),
            limit=1
        )
        uuid = check.objects[0].uuid

        result_collection.data.update(
            uuid=uuid,
            properties={
                "rank_keywords": request.keyword
            }
        )
        return {"resultCode": 200, "data": request.keyword}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}