def extract_text_from_pdf(file_path):
    # 从pdf文件中提取文本
    import PyPDF2
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(file_path):
    # 从txt文件中提取文本
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
def extract_text_from_md(file_path):
    # 从md文件中提取文本
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
def extract_text_from_docx(file_path):
    # 从docx文件中提取文本
    import docx
    doc = docx.Document(file_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text