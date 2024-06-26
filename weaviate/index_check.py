from fastapi import FastAPI, HTTPException, Query, Body
from weaviate.classes.query import Filter, MetadataQuery


import weaviate
import os

HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

client = weaviate.connect_to_local(
    headers={
        "X-HuggingFace-Api-Key": HUGGINGFACE_API_KEY,
    }
)

paper_collection = client.collections.get("Paper")
result_collection = client.collections.get("result")

title = "bert paper6"

response = result_collection.query.fetch_objects(
        filters=Filter.by_property("title").equal(title),
        limit=1,
        return_properties=["title", "time1", "time2"]
    )

if response.objects:
    print("yes")
    print(response.objects[0].properties)
else:
    print("no")
time = response.objects[0].properties["time1"]
time = str(time)
if time == "None":
    print("yes")
else:
    print("no")



client.close()
