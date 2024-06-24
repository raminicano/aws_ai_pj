from fastapi import FastAPI, HTTPException, Query, Body
import requests
import feedparser
from typing import List, Dict, Any
from pydantic import BaseModel
import weaviate
import os
import weaviate.classes.config as wc
from weaviate.classes.query import Filter, MetadataQuery
import deepl


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

app = FastAPI()

HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
DEEPL_AUTH_KEY = os.environ["DEEPL_AUTH_KEY"]


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
            propertites={
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
            propertites={
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
            propertites={
                "summary3": request.text
            }
        )
        return {"resultCode": 200, "data": "summary3 저장 성공"}
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}