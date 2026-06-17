from pypdf import PdfReader

def extract_text(pdf):

    reader = PdfReader(pdf)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text