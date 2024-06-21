from transformers import pipeline
from pdfminer.high_level import extract_text

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text


loader = PyPDFLoader("https://arxiv.org/pdf/1910.14296v2")
document = loader.load()

# pdf 파일을 저장
loader.save_local("1910.14296v2_test.pdf")