import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("./1910.14296v2.pdf")
document = loader.load()

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n\n", 
    chunk_size=2000,  
    chunk_overlap=500,
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



nltk.download('stopwords')
stop_words = stopwords.words('english')

# 텍스트 전처리 함수 정의
def preprocess(text):
    # 소문자로 변환
    text = text.lower()
    # 구두점 제거
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 불용어 제거 및 토큰화
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# 각 문서에 대해 전처리 수행
processed_docs = [preprocess(doc.page_content) for doc in filtered_docs]




## 주제 모델링 수행 ###

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(processed_docs)

# LDA 모델 학습
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# 주제 추출 함수 정의
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

# 주제 출력
no_top_words = 10
topics = display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)
for idx, topic in enumerate(topics):
    print(f"Topic {idx}: {topic}")



### 주제별 요약 생성 ###
from transformers import pipeline

# Hugging Face 요약 모델 로드
summarizer = pipeline("summarization", "jordiclive/flan-t5-3b-summarizer", torch_dtype=torch.bfloat16)

# 문서 주제 할당
doc_topic_distributions = lda.transform(tfidf_matrix)
doc_topics = doc_topic_distributions.argmax(axis=1)

# 주제별 문서 그룹화
topic_docs = {i: [] for i in range(lda.n_components)}
for i, topic in enumerate(doc_topics):
    topic_docs[topic].append(filtered_docs[i].page_content)

# 주제별 요약 생성 함수 정의
def summarize_texts(texts, max_length=512, min_length=5):
    summaries = []
    for text in texts:
        summary = summarizer(text, max_length=max_length, min_length=min_length, num_beams=5, truncation=True)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

# 주제별 요약 생성 및 출력
for topic, docs in topic_docs.items():
    if docs:
        topic_summary = summarize_texts(docs)
        print(f"Summary for Topic {topic}: {topic_summary}")



