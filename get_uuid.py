import weaviate
import os
import weaviate.classes.config as wc


HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

client = weaviate.connect_to_local(
    headers={
        "X-HuggingFace-Api-Key": HUGGINGFACE_API_KEY,
    }
)

# 연결 확인
if client.is_ready():
    print("Weaviate Cloud에 성공적으로 연결되었습니다.")
else:
    print("Weaviate Cloud에 연결할 수 없습니다.")


collection = client.collections.get("Trans")
res = collection.query.fetch_objects(
    limit=1
)
print(res.objects[0].uuid)

client.close()