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


# Paper 클래스의 속성 및 벡터화 설정 정의
client.collections.create(
    name="result",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="trans_summary", data_type=wc.DataType.TEXT),
        wc.Property(name="summary1", data_type=wc.DataType.TEXT),
        wc.Property(name="summary2", data_type=wc.DataType.TEXT),
        wc.Property(name="summary3", data_type=wc.DataType.TEXT),
        wc.Property(name="full_text", data_type=wc.DataType.TEXT),
        wc.Property(name="bert_keywords", data_type=wc.DataType.TEXT_ARRAY),
        wc.Property(name="rank_keywords", data_type=wc.DataType.TEXT_ARRAY),
        wc.Property(name="time1", data_type=wc.DataType.INT),
        wc.Property(name="time2", data_type=wc.DataType.INT),
        wc.Property(name="time3", data_type=wc.DataType.INT),
    ]
)

# 스키마 생성 확인
print("Trans 컬렉션이 성공적으로 생성되었습니다.")
collection = client.collections.get("result")
print(collection)

client.close()