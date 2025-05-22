import os
# Disabilita il file watcher di Streamlit per velocizzare il reload
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import pandas as pd

from modules.config import LLM_MODELS
from modules.text_extractor import detect_extension, extract_text
from modules.generazione_testo import (
    generate_report,
    edit_document,
    extract_entities,
    sensitive_informations,
)
from modules.utils import write_file


def main() -> None:
    """
    Streamlit app per il caricamento, l'analisi e la modifica di documenti.
    """
    st.set_page_config(
        page_title="ProgettoDL Analyzer",
        layout="wide",
    )

    col_left, col_center, col_right = st.columns([0.15, 0.7, 0.15])

    # Sidebar: scelta input utente
    input_type = st.sidebar.radio(
        "Input:",
        ["Testo", "File"],
    )

    document_text = ""
    file_ext = ".txt"

    if input_type == "Testo":
        document_text = st.sidebar.text_area(
            "Inserisci testo", height=300
        )
    else:
        uploaded = st.sidebar.file_uploader(
            "Carica file",
            type=["pdf", "docx", "txt", "csv"],
            accept_multiple_files=False,
        )
        if uploaded:
            file_ext = detect_extension(uploaded)
            if file_ext == ".csv":
                df = pd.read_csv(uploaded)
                st.sidebar.dataframe(df)
                document_text = df.to_csv(index=False)
                file_ext = ".txt"
            else:
                document_text = extract_text(uploaded)

    if document_text:
        # Genera report con ciascun modello LLM
        if col_center.button(
            "Genera report",
            type="primary",
            use_container_width=True,
        ):
            with col_center.status("Analisi in corso...") as status:
                reports = {
                    name: generate_report(document_text, model)
                    for name, model in LLM_MODELS.items()
                }
                status.success("Analisi completata!")
            # Visualizza report
            for name, report in reports.items():
                st.markdown(f"### Report {name}\n{report}")
            st.session_state["reports"] = reports

        # Applica modifiche in base al report
        if (
            "reports" in st.session_state
            and col_center.button(
                "Modifica",
                use_container_width=True,
            )
        ):
            with col_center.status("Modifica in corso...") as status:
                edited_docs = {
                    name: edit_document(
                        document_text,
                        report,
                        LLM_MODELS[name],
                    )
                    for name, report in st.session_state["reports"].items()
                }
                entities = extract_entities(document_text)
                status.success("Modifica completata!")

            # Mostra colonne
            col_left.markdown("## Originale\n" + document_text)
            col_center.markdown(
                "## Modificato\n"
                + "\n\n".join(edited_docs.values())
            )
            col_right.markdown("## Entità riconosciute\n" + entities)

            # Pulsanti di download
            col_center.download_button(
                "Scarica modificato",
                write_file(
                    "\n\n".join(edited_docs.values()), file_ext
                ),
                file_name="edited" + file_ext,
            )
            col_right.download_button(
                "Scarica entità",
                write_file(entities, ".txt"),
                file_name="entities.txt",
            )

            # Mostra contesti sensibili
            for name, report in st.session_state["reports"].items():
                col_left.markdown(
                    sensitive_informations(report, name)
                )


if __name__ == "__main__":
    main()
