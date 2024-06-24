from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from transformers import pipeline
import torch

loader = PyPDFLoader("./1910.14296v2.pdf")
document = loader.load()

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n",  # 분할 기준
    chunk_size=2000,   # 청크 사이즈
    chunk_overlap=500, # 중첩 사이즈
)


# 분할 실행
split_docs = text_splitter.split_documents(document)
print(f'총 분할된 도큐먼트 수: {len(split_docs)}')


# References 키워드가 포함된 이후 도큐먼트를 무시하고 그 전까지 저장
filtered_docs = []
for i, doc in enumerate(split_docs):
    if "References" in doc.page_content:
        doc.page_content = doc.page_content.split("References")[0]
        filtered_docs.append(doc.page_content)
        break
    filtered_docs.append(doc.page_content)
    


# for i in range(len(filtered_docs)):
#     file_name = f"split_doc_{i+1}.txt"

#     with open(file_name, 'w', encoding='utf-8') as file:
#         file.write(filtered_docs[i])
#     print(f"Document saved to {file_name}")


#----- Map 단계 : 텍스트 요약 -------

summarizer = pipeline("summarization", "jordiclive/flan-t5-3b-summarizer", torch_dtype=torch.bfloat16)

# 텍스트 요약 함수 정의
def summarize_text(text, prompt=f"Here are some of the documents. Please summarize the main contents based on this document list. Answer:", max_length=512, min_length=5):
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


# 각 분할된 도큐먼트에 대해 요약 실행
summaries = []
i = 1
for doc in filtered_docs:
    summary = summarize_text(doc)
    summaries.append(summary)
    # 요약본 저장
    file_name = f"summary_doc_{i}.txt"
    i += 1
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(summary)
    print(f"Summary saved to {file_name}")



##### Reduce 부분 ######
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI

# # Reduce 단계에서 처리할 프롬프트 정의
# reduce_template = """다음은 요약의 집합입니다:
# {doc_summaries}
# 이것들을 바탕으로 통합된 요약을 만들어 주세요.
# 답변:"""
# reduce_prompt = PromptTemplate.from_template(reduce_template)

# # LLMChain 정의
# llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')
# reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# combine_documents_chain = StuffDocumentsChain(
#     llm_chain=reduce_chain,
#     document_variable_name="doc_summaries"
# )

# from langchain.chains import ReduceDocumentsChain
# reduce_documents_chain = ReduceDocumentsChain(
#     combine_documents_chain=combine_documents_chain,
#     collapse_documents_chain=combine_documents_chain,
#     token_max=4000,
# )
# from langchain.chains import MapReduceDocumentsChain

# map_reduce_chain = MapReduceDocumentsChain(
#     llm_chain=reduce_chain,
#     reduce_documents_chain=reduce_documents_chain,
#     document_variable_name="pages",
#     return_intermediate_steps=False,
# )


# # Map-Reduce 체인 실행
# result = map_reduce_chain.run(filtered_docs)
# # 요약 결과 출력
# print(result)
