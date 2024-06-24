from transformers import pipeline
from pdfminer.high_level import extract_text

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text

# 'References' 뒤의 텍스트 제거
def remove_references(text):
    reference_index = text.find('References')
    if reference_index != -1:
        text = text[:reference_index]
    return text

# 텍스트 파일로 저장
def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# PDF 파일 경로
pdf_path = '/work/aws_ai_pj/1910.14296v2.pdf'

# PDF 파일에서 텍스트 추출
raw_document = extract_text_from_pdf(pdf_path)

# 'References' 뒤의 텍스트 제거
processed_document = remove_references(raw_document)

# 텍스트에서 \n 제거
processed_document = processed_document.replace('\n', '')

# 결과를 텍스트 파일로 저장
output_file_path = '/work/aws_ai_pj/processed_document.txt'
save_text_to_file(processed_document, output_file_path)

# 결과를 텍스트 파일로 저장
output_file_path = '/work/aws_ai_pj/processed_document_test.txt'
save_text_to_file(processed_document, output_file_path)

print(f"Processed document saved to {output_file_path}")
