# modules/text_extractor.py
import logging

import pandas as pd
from docx import Document
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import streamlit as st
from typing import Optional, List


def detect_extension(uploaded_file) -> str:
    """Restituisce l’estensione del file caricato, es. '.pdf', '.docx', '.txt', '.csv'"""
    name = uploaded_file.name.lower()
    if '.' in name:
        return '.' + name.split('.')[-1]
    return ""  # Nessuna estensione trovata


def _ocr_page_image(page_object, lang: str = "ita") -> str:
    """Esegue OCR su un'immagine di pagina (da PyMuPDF o pdfplumber)."""
    try:
        if isinstance(page_object, fitz.Page):  # PyMuPDF page
            pix = page_object.get_pixmap(alpha=False, dpi=300)  # Buona risoluzione per OCR
            img_bytes = pix.tobytes("png")  # o "jpeg"
            pil_img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(pil_img, lang=lang)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Errore durante OCR su pagina: {e}")
    return ""


def extract_text_from_pdf_optimized(
        uploaded_file,
        use_ocr_if_needed: bool = True,
        max_pages_to_sample: Optional[int] = None,
        progress_bar_key: Optional[str] = None  # Chiave per st.session_state per la progress bar
) -> str:
    """
    Estrae testo da un PDF usando PyMuPDF (fitz), con OCR opzionale per pagine basate su immagini
    e possibilità di campionare un numero massimo di pagine.
    Aggiorna una progress bar in Streamlit tramite st.session_state.
    """
    logger = logging.getLogger(__name__)
    pages_text_list: List[str] = []

    try:
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Importante se il file viene letto di nuovo

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages_total = len(doc)

        pages_to_process = num_pages_total
        if max_pages_to_sample is not None and max_pages_to_sample < num_pages_total:
            pages_to_process = max_pages_to_sample
            logger.info(f"Campionamento PDF: verranno processate {pages_to_process} pagine su {num_pages_total}.")

        for i, page in enumerate(doc):
            if max_pages_to_sample is not None and i >= max_pages_to_sample:
                logger.info(f"Raggiunto limite di campionamento di {max_pages_to_sample} pagine.")
                break

            if progress_bar_key and i % 5 == 0 or i == pages_to_process - 1:
                progress_percent = int(((i + 1) / pages_to_process) * 100)
                if progress_bar_key in st.session_state:
                    st.session_state[progress_bar_key].progress(progress_percent,
                                                                text=f"Estrazione PDF: pagina {i + 1}/{pages_to_process}...")
                else:
                    pass

            page_text = page.get_text("text", sort=True)  # Estrae testo, sort=True per ordine di lettura

            if page_text and len(page_text.strip()) > 30:  # Soglia minima per considerare testo valido
                pages_text_list.append(page_text)
            elif use_ocr_if_needed:
                logger.info(
                    f"Pagina {i + 1} del PDF '{uploaded_file.name}' ha poco testo ({len(page_text.strip())} chars). Tento OCR.")
                ocr_text = _ocr_page_image(page, lang="ita")  # Assumendo lingua italiana per OCR
                if ocr_text and ocr_text.strip():
                    pages_text_list.append(ocr_text)
                else:
                    logger.info(f"OCR non ha prodotto testo per pagina {i + 1}.")
                    pages_text_list.append("")
            else:
                pages_text_list.append("")

        doc.close()
    except Exception as e:
        logger.error(f"Errore durante l'estrazione del testo dal PDF '{uploaded_file.name}': {e}", exc_info=True)
        raise ValueError(f"Errore nell'elaborazione del PDF '{uploaded_file.name}': {e}")

    final_text = "\n\n".join(filter(None, pages_text_list))  # Unisci con doppio newline, scarta stringhe vuote
    logger.info(f"Testo estratto da PDF '{uploaded_file.name}'. Lunghezza totale: {len(final_text)} caratteri.")
    return final_text


def extract_text(
        uploaded_file,
        use_ocr_for_pdf: bool = True,  # Nuovo parametro per controllare OCR
        pdf_max_pages_to_sample: Optional[int] = None,  # Nuovo parametro per campionamento PDF
        pdf_progress_bar_key: Optional[str] = None  # Nuovo parametro per progress bar
) -> str:
    """
    Estrae tutto il testo da PDF (ottimizzato con PyMuPDF e OCR opzionale), DOCX, TXT o CSV.
    Per CSV restituisce la rappresentazione testuale (to_csv).
    """
    ext = detect_extension(uploaded_file)

    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)

    if ext == '.csv':
        try:
            df = pd.read_csv(uploaded_file)
            if hasattr(uploaded_file, 'seek'): uploaded_file.seek(0)  # Reset per usi futuri
            return df.to_csv(index=False)
        except Exception as e_csv:
            logger = logging.getLogger(__name__)
            logger.error(f"Errore lettura CSV '{uploaded_file.name}': {e_csv}")
            raise ValueError(f"Impossibile leggere il file CSV: {e_csv}")

    if ext == '.pdf':
        return extract_text_from_pdf_optimized(
            uploaded_file,
            use_ocr_if_needed=use_ocr_for_pdf,
            max_pages_to_sample=pdf_max_pages_to_sample,
            progress_bar_key=pdf_progress_bar_key
        )

    if ext == '.docx':
        try:
            doc = Document(uploaded_file)
            if hasattr(uploaded_file, 'seek'): uploaded_file.seek(0)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e_docx:
            logger = logging.getLogger(__name__)
            logger.error(f"Errore lettura DOCX '{uploaded_file.name}': {e_docx}")
            raise ValueError(f"Impossibile leggere il file DOCX: {e_docx}")

    if ext == '.txt':
        try:
            text_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            if hasattr(uploaded_file, 'seek'): uploaded_file.seek(0)
            return text_content
        except Exception as e_txt:
            logger = logging.getLogger(__name__)
            logger.error(f"Errore lettura TXT '{uploaded_file.name}': {e_txt}")
            raise ValueError(f"Impossibile leggere il file TXT: {e_txt}")

    raise ValueError(f"Estensione non supportata: {ext}")
