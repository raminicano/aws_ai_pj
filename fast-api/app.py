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

class TranslationResponse(BaseModel):
    resultCode: int
    data: str

class TranslationPaperRequest(BaseModel):
    title: str
    text: str
    target_lang: str = 'KO'

app = FastAPI()

HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
DEEPL_AUTH_KEY = os.environ["DEEPL_AUTH_KEY"]


client = weaviate.connect_to_local(
    headers={
        "X-HuggingFace-Api-Key": HUGGINGFACE_API_KEY,
    }
)

paper_collection = client.collections.get("Paper")
trans_collection = client.collections.get("Trans")

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


@app.post("/translate_text", response_model=TranslationResponse)
async def translate_text_endpoint(request: TranslationRequest):
    try:
        translator = deepl.Translator(DEEPL_AUTH_KEY)
        result = translator.translate_text(request.text, target_lang=request.target_lang)
        return {"resultCode": 200, "data": result.text}
    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Translation failed")


@app.post("/translate_text_paper", response_model=TranslationResponse)
async def translate_text_paper(request: TranslationPaperRequest):
    try:
        with trans_collection.batch.fixed_size(1) as batch:
            response = trans_collection.query.fetch_objects(
                    filters=Filter.by_property("title").equal(request.title),
                    limit=1
                )

            if response.objects:
                print("trans_summary is found")
                trans_summary = response.objects[0].properties["trans_summary"]
                if trans_summary:
                    return {"resultCode": 200, "data": trans_summary}
                else:
                    return {"resultCode": 404, "data": "trans_summary is not found"}
            else:
                translator = deepl.Translator(DEEPL_AUTH_KEY)
                result = translator.translate_text(request.text, target_lang=request.target_lang)
                batch.add_object(
                    properties={
                        "title": request.title,
                        "trans_summary": result.text
                    }
                )
                return {"resultCode": 200, "data": result.text}
    except deepl.DeepLException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        return {"resultCode": 500, "data": str(e)}
