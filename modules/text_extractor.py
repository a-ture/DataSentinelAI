import pandas as pd
import pdfplumber
from docx import Document


def detect_extension(uploaded_file) -> str:
    """Restituisce l’estensione del file caricato, es. '.pdf', '.docx', '.txt', '.csv'"""
    name = uploaded_file.name.lower()
    return '.' + name.split('.')[-1]


def extract_text(uploaded_file) -> str:
    """
    Estrae tutto il testo da PDF, DOCX, TXT o CSV.
    - per CSV restituisce la to_csv() senza indice
    """
    ext = detect_extension(uploaded_file)
    uploaded_file.seek(0)

    if ext == '.csv':
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False)

    if ext == '.pdf':
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n"
        return text

    if ext == '.docx':
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == '.txt':
        return uploaded_file.getvalue().decode('utf-8', errors='ignore')

    raise ValueError(f"Estensione non supportata: {ext}")
