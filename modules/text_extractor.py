import io
import PyPDF2
import docx
import re
from fpdf import FPDF

def extract(file):
    text = ""
    extension = findExtension(file)
    uploaded_file = io.BytesIO(file.read())
    match extension:
        case ".docx":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        case ".pdf":
            pdf = PyPDF2.PdfFileReader(uploaded_file)
            for page in range(pdf.getNumPages()):
                text += pdf.getPage(page).extract_text()
            return text
        case ".txt":
            text = uploaded_file.read()
            return text
    return None

def findExtension(file):
    return re.findall(r"\.+[A-Za-z]*", file.name).pop()

def writeFile(data, extension):
    match extension:
        case ".docx":
            doc = docx.Document()
            doc.add_paragraph(data)
            doc.save("edited.docx")
            return open("edited.docx", "rb").read()
        case ".pdf":
            pdf=FPDF()
            pdf.core_fonts_encoding = 'utf8'
            pdf.add_page(orientation="P")
            pdf.set_font("helvetica")
            pdf.multi_cell(w=0, h=10, text=data, border=0, align="L")
            pdf.output("edited.pdf", "S")
            return open("edited.pdf", "rb").read()
        case ".txt":
            return data
    return None


# modules/text_extractor.py

import pandas as pd
import pdfplumber
from docx import Document


def detect_extension(uploaded_file) -> str:
    """
    Restituisce l'estensione del file caricato,
    es. '.pdf', '.docx', '.txt', '.csv'
    """
    name = uploaded_file.name.lower()
    return '.' + name.split('.')[-1]


def extract_text(uploaded_file) -> str:
    """
    Estrae e restituisce tutto il testo da PDF, DOCX o TXT.
    """
    ext = detect_extension(uploaded_file)

    if ext == '.pdf':
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    elif ext == '.docx':
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == '.txt':
        # file_uploader restituisce un BytesIO
        return uploaded_file.getvalue().decode('utf-8')

    else:
        raise ValueError(f"Estensione non supportata per extract_text: {ext}")
