# modules/utils.py

import io


def write_file(content: str, extension: str) -> bytes:
    """
    Crea un buffer di byte a partire da una stringa, per il download in Streamlit.

    Args:
        content (str): testo da salvare.
        extension (str): estensione del file (es. '.txt', '.pdf').

    Returns:
        bytes: contenuto in formato byte.
    """
    buffer = io.BytesIO()
    buffer.write(content.encode("utf-8"))
    buffer.seek(0)
    return buffer.getvalue()


def sensitive_informations(report: str, model_name: str) -> str:
    """
    Formatta un report di contesti sensibili da mostrare in Streamlit.

    Args:
        report (str): testo del report generato dal modello.
        model_name (str): chiave del modello usato.

    Returns:
        str: markdown con il titolo del modello e il report.
    """
    header = f"\n\n### {model_name}\n"
    return header + report.strip() + "\n"
