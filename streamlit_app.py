# streamlit_app.py
import importlib
import types

try:
    m = importlib.import_module("torch._classes")
    # Fai credere al file‐watcher che esista __path__._path come lista vuota
    m.__path__ = types.SimpleNamespace(_path=[])
except ImportError:
    pass

import streamlit as st
import pandas as pd
import json
import asyncio
import os
import numpy as np
import logging
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
from io import BytesIO
import re

# Import dai tuoi moduli
from modules.text_extractor import detect_extension, extract_text  # extract_text ora potenziato
from modules.generazione_testo import generate_report_on_full_text, edit_document
from modules.analysis_csv import analyze_and_anonymize_csv, get_llm_overall_csv_comment
from modules.privacy_metrics import calculate_k_anonymity, calculate_l_diversity
from modules.config import LLM_MODELS

# Configurazione del Logger
logger_streamlit = logging.getLogger(__name__)
if not logger_streamlit.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s (StreamlitApp): %(message)s")

# Costanti e Definizioni Globali
PII_TYPES = [  # Assicurati che questa lista sia completa per i tuoi casi d'uso
    "PERSON", "PER", "PERS", "NOME", "COGNOME", "FULL_NAME", "NAME",
    "DATE", "DATA", "BIRTHDATE",
    "LOCATION", "LOC", "INDIRIZZO", "ADDRESS", "CITTA", "COMUNE", "LUOGO", "CITY",
    "ORGANIZATION", "ORG", "AZIENDA", "ENTE", "SOCIETA", "COMPANY",
    "ID_NUMBER", "CODICE FISCALE", "CF", "PASSPORT_NUMBER", "IDCARD_NUMBER", "PATIENT_ID",
    "PHONE_NUMBER", "TELEFONO",
    "EMAIL", "EMAIL ADDRESS",
    "CREDIT_CARD_NUMBER", "IBAN", "CREDIT CARD",
    "VEHICLE_REGISTRATION", "TARGA", "LICENSE_PLATE",
    "MEDICAL_RECORD", "HEALTH_CONDITION", "DIAGNOSIS", "MEDICATION", "CONDIZIONE MEDICA",
    "IP_ADDRESS", "USERNAME",
    "CUI", "IMPORTO", "POSTAL CODE", "CAP"
]
PII_TYPES_LOWER_SET = {pii.lower() for pii in PII_TYPES}


# Funzione per badge colorati
def colored_badge(categoria: str) -> str:
    cat_lower = categoria.lower()
    if "diretto" in cat_lower and "identificatore" in cat_lower:
        return (f"<span style='color:#D32F2F; font-weight:bold; padding: 3px 6px; border-radius: 4px; "
                f"background-color: #FFCDD2;'>🔴 Identificatore Diretto</span>")
    elif "quasi" in cat_lower and "identificatore" in cat_lower:
        return (f"<span style='color:#F57C00; font-weight:bold; padding: 3px 6px; border-radius: 4px; "
                f"background-color: #FFE0B2;'>🟠 Quasi-Identificatore</span>")
    elif "attributo sensibile" in cat_lower:
        return (f"<span style='color:#1976D2; font-weight:bold; padding: 3px 6px; border-radius: 4px; "
                f"background-color: #BBDEFB;'>🔵 Attributo Sensibile</span>")
    elif "non sensibile" in cat_lower or "nessuna pii" in cat_lower or "non pii" in cat_lower or "nessuno" == cat_lower or (
            "non" in cat_lower and ("pii" in cat_lower or "sensibile" in cat_lower)):
        return (f"<span style='color:#388E3C; font-weight:bold; padding: 3px 6px; border-radius: 4px; "
                f"background-color: #C8E6C9;'>🟢 Non Sensibile / Non PII</span>")
    else:
        return (f"<span style='color:#546E7A; font-weight:bold; padding: 3px 6px; border-radius: 4px; "
                f"background-color: #CFD8DC;'>⚪ {categoria}</span>")


def redact_pdf_in_memory(pdf_bytes: bytes, sensitive_terms: Dict[str, str],
                         redaction_mode: str = "placeholder") -> bytes:
    """
    Crea una versione redatta di un PDF in memoria, applicando le redazioni.
    """

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        sorted_terms_for_redaction = sorted(sensitive_terms.items(), key=lambda item: len(item[0]), reverse=True)

        for page in doc:
            for term, placeholder_text in sorted_terms_for_redaction:
                if not term.strip():
                    continue

                text_instances_found = page.search_for(term, flags=fitz.TEXT_INHIBIT_SPACES, quads=False)

                for inst_rect in text_instances_found:
                    fill_color_redact = (0.0, 0.0, 0.0) if redaction_mode == "blackbox" else (0.8, 0.8, 0.8)
                    text_for_annot = "" if redaction_mode == "blackbox" else placeholder_text

                    page.add_redact_annot(
                        inst_rect,
                        text=text_for_annot,
                        fill=fill_color_redact,
                        text_color=(0, 0, 0),
                        cross_out=False
                    )

        # ✅ Questa parte è la correzione chiave
        for page in doc:
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
            except Exception as e:
                logger_streamlit.warning(f"Errore durante la redazione della pagina: {e}")

        out_buffer = BytesIO()
        doc.save(out_buffer, garbage=3, deflate=True, clean=True)
        doc.close()
        logger_streamlit.info("PDF redatto salvato in memoria.")
        return out_buffer.getvalue()

    except Exception as e:
        logger_streamlit.error(f"Errore generico durante la redazione del PDF: {e}", exc_info=True)
        if doc:
            doc.close()
        st.error(f"Errore durante la redazione del PDF: {e}. Verrà offerto il download del file originale.")
        return pdf_bytes


def _prepare_sensitive_terms_for_pdf_redaction(reports_text_state: Optional[Dict[str, Any]], _) -> Dict[str, str]:
    """
    Estrae e prepara i termini sensibili per la redazione del PDF, applicando controlli robusti.
    """
    sensitive_terms_map: Dict[str, str] = {}
    placeholder_map = {
        "PERSON": "[NOME]", "EMAIL": "[EMAIL]", "LOCATION": "[LUOGO]", "ADDRESS": "[INDIRIZZO]",
        "PHONE_NUMBER": "[TELEFONO]", "ID_NUMBER": "[ID]", "CODICE FISCALE": "[CF]",
        "ORGANIZATION": "[ORGANIZZAZIONE]", "DATE": "[DATA]",
        "HEALTH_CONDITION": "[INFO_SALUTE]", "DIAGNOSIS": "[DIAGNOSI]", "MEDICATION": "[FARMACO]"
    }

    if reports_text_state:

        for model_name, model_report in reports_text_state.items():
            # 1) Salto subito tutto ciò che non è dict
            if not isinstance(model_report, dict):
                logger_streamlit.warning(
                    f"Skipping report from '{model_name}' of type {type(model_report)}, expected dict.")
                continue

            # 2) Salto se non ho entità trovate o non è una lista
            if not model_report.get("found") or not isinstance(model_report.get("entities"), list):
                continue

            # 3) Itero sulle entità con un'ulteriore protezione
            for entity in model_report.get("entities", []):
                # Controllo di sicurezza interno al loop sulle singole entità
                if not isinstance(entity, dict):
                    logger_streamlit.warning(f"Ignoro elemento entity di tipo {type(entity)} in report '{model_name}'.")
                    continue

                term_text = entity.get("text")
                term_type = entity.get("type")

                if term_text and isinstance(term_text, str) and term_type and isinstance(term_type, str):
                    if term_type.lower() in PII_TYPES_LOWER_SET:
                        placeholder = placeholder_map.get(term_type.upper(), f"[{term_type.upper()}]")
                        sensitive_terms_map[term_text.strip()] = placeholder

    # Ordina per lunghezza per gestire correttamente i termini annidati
    sorted_terms = sorted(sensitive_terms_map.items(), key=lambda item: len(item[0]), reverse=True)
    return dict(sorted_terms)


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """
    Spezza il testo in frammenti di al più max_chars caratteri,
    senza tagliare a metà le parole.
    """
    words = text.split()
    chunks, curr, curr_len = [], [], 0
    for w in words:
        curr.append(w)
        curr_len += len(w) + 1
        if curr_len >= max_chars:
            chunks.append(" ".join(curr))
            curr, curr_len = [], 0
    if curr:
        chunks.append(" ".join(curr))
    return chunks


# --- _perform_text_analysis AGGIORNATA (SENZA CHUNKING INTERNO) ---
def _perform_text_analysis(text_to_analyze: str):
    logger_streamlit.info(f"Avvio _perform_text_analysis su testo di lunghezza {len(text_to_analyze)}.")
    st.info("⏳ Esecuzione analisi PII su testo completo (chunked)…")
    reports_text_single_model_results: Dict[str, Dict[str, Any]] = {}

    if not LLM_MODELS:
        st.error("Nessun modello LLM configurato in modules/config.py.")
        st.session_state["reports_text"] = reports_text_single_model_results
        return

    if not text_to_analyze.strip():
        st.warning("Testo vuoto fornito per l'analisi.")
        st.session_state["reports_text"] = reports_text_single_model_results
        return

    # suddivido in chunk da max 3000 char
    chunks = chunk_text(text_to_analyze, max_chars=3000)

    for model_name, model_api_id in LLM_MODELS.items():
        logger_streamlit.info(f"Analisi PII con {model_name} in {len(chunks)} chunk…")
        combined = {"found": False, "entities": [], "summary": []}

        for chunk in chunks:
            rep = generate_report_on_full_text(chunk, model_api_id)
            if rep.get("entities"):
                combined["entities"].extend(rep["entities"])
                combined["found"] = True
            combined["summary"].append(rep.get("summary", ""))

        # unisco i pezzi di riassunto e lo salvo
        combined["summary"] = "\n".join(s for s in combined["summary"] if s)
        reports_text_single_model_results[model_name] = combined

    st.session_state["reports_text"] = reports_text_single_model_results
    st.success("✅ Analisi PII su testo completo terminata!")
    logger_streamlit.info("_perform_text_analysis (chunked) completata.")


def _perform_csv_analysis(df_to_analyze_full: pd.DataFrame, selected_model_name: Optional[str], num_bins: int,
                          granularity_date: str):
    mod_api_csv_selected = None
    if selected_model_name and selected_model_name in LLM_MODELS:
        mod_api_csv_selected = LLM_MODELS[selected_model_name]

    st.info(f"⏳ Esecuzione analisi CSV potenziata...")
    if selected_model_name and mod_api_csv_selected:
        st.caption(f"(Colonne testuali con {selected_model_name})")
    elif not LLM_MODELS:
        st.caption("(Nessun modello LLM configurato per l'analisi testuale delle colonne)")
    elif not selected_model_name and LLM_MODELS:
        st.caption("(Nessun modello LLM selezionato per l'analisi testuale delle colonne)")

    csv_progress_bar = st.progress(0, text="Avvio analisi CSV dettagliata...")
    try:
        report_cols_df_result, df_anon_result = asyncio.run(
            analyze_and_anonymize_csv(
                df_to_analyze_full, mod_api_csv_selected,
                sample_size_for_preview=5, max_concurrent_requests=3,
                num_bins_numeric=num_bins, granularity_for_dates=granularity_date
            )
        )
        st.session_state["csv_analysis_report_df"] = report_cols_df_result
        st.session_state["csv_anon_df"] = df_anon_result
        csv_progress_bar.progress(40, text="Analisi colonne (testo, numeriche, date) completata.")
    except Exception as e_analyze_cols:
        st.error(f"Errore grave durante l'analisi completa delle colonne CSV: {e_analyze_cols}")
        logger_streamlit.error(f"Errore in analyze_and_anonymize_csv: {e_analyze_cols}", exc_info=True)
        st.session_state["csv_analysis_report_df"] = pd.DataFrame()
        st.session_state["csv_anon_df"] = df_to_analyze_full.copy()
        csv_progress_bar.progress(40, text="Errore analisi colonne.")
        csv_progress_bar.empty()
        st.error("Analisi CSV interrotta.")
        return

    csv_progress_bar.progress(60, text="Passo 2/3: Calcolo metriche di perdita di utilità...")
    loss_metrics_results = {"n_righe_originali": len(df_to_analyze_full),
                            "n_righe_anonimizzate": len(st.session_state["csv_anon_df"]) if st.session_state.get(
                                "csv_anon_df") is not None else len(df_to_analyze_full)}
    original_numeric_cols_for_loss = df_to_analyze_full.select_dtypes(include=["int64", "float64"])
    if not original_numeric_cols_for_loss.empty:
        col_esempio_util = original_numeric_cols_for_loss.columns[0]
        original_series_util = original_numeric_cols_for_loss[col_esempio_util].dropna()
        if pd.api.types.is_numeric_dtype(original_series_util) and not original_series_util.empty and len(
                original_series_util) >= 2:
            var_originale_util = original_series_util.var()
            if st.session_state.get("csv_anon_df") is not None and col_esempio_util in st.session_state[
                "csv_anon_df"].columns:
                anon_series_for_util = st.session_state["csv_anon_df"][col_esempio_util]

                def mid_of_fascia_util(x_fascia):
                    if pd.isna(x_fascia) or not isinstance(x_fascia, str) or '-' not in x_fascia: return np.nan
                    try:
                        low_str, high_str = x_fascia.split("-")
                        return (float(low_str) + float(high_str)) / 2.0
                    except ValueError:
                        return np.nan

                midpoints_util = anon_series_for_util.apply(mid_of_fascia_util)
                var_generalizzata_util = pd.to_numeric(midpoints_util, errors='coerce').dropna().var()
                preserved_util = (var_generalizzata_util / var_originale_util) * 100 if pd.notna(
                    var_originale_util) and var_originale_util > 1e-6 and pd.notna(var_generalizzata_util) else 0.0
                loss_metrics_results[col_esempio_util] = {
                    "var_originale": var_originale_util if pd.notna(var_originale_util) else 'N/D',
                    "var_generalizzata_stimata": var_generalizzata_util if pd.notna(var_generalizzata_util) else 'N/D',
                    "percentuale_preserved_var (%)": round(preserved_util, 1) if preserved_util != 'N/D' else 'N/D'
                }
    st.session_state["loss_of_utility_metrics"] = loss_metrics_results
    csv_progress_bar.progress(70, text="Metriche di perdita di utilità calcolate.")

    report_cols_for_final_metrics = st.session_state.get("csv_analysis_report_df")
    if report_cols_for_final_metrics is not None and not report_cols_for_final_metrics.empty:
        csv_progress_bar.progress(75, text="Passo 3/3: Calcolo metriche di rischio e generazione report finale...")
        qids_final, sas_final = [], []
        if "CategoriaLLM" in report_cols_for_final_metrics.columns:
            mask_qid_final = report_cols_for_final_metrics["CategoriaLLM"].str.lower().str.contains(
                r"quasi[\s\-]?identificatore", na=False, regex=True)
            candidati_qid_final = report_cols_for_final_metrics[mask_qid_final]["Colonna"].tolist()
            df_sample_metrics_final = df_to_analyze_full
            MAX_ROWS_FOR_METRICS_CALC_FINAL = 100_000
            if len(df_sample_metrics_final) > MAX_ROWS_FOR_METRICS_CALC_FINAL:
                df_sample_metrics_final = df_sample_metrics_final.sample(n=MAX_ROWS_FOR_METRICS_CALC_FINAL,
                                                                         random_state=42)
            for c_qid_f in candidati_qid_final:
                if c_qid_f in df_sample_metrics_final.columns:
                    cardinality_ratio_f = df_sample_metrics_final[c_qid_f].nunique(dropna=False) / len(
                        df_sample_metrics_final)
                    if len(df_sample_metrics_final) < 100 or cardinality_ratio_f < 0.90: qids_final.append(c_qid_f)
            sas_mask_final = report_cols_for_final_metrics["CategoriaLLM"].str.lower().str.contains(
                "attributo sensibile", na=False, regex=False)
            sas_candidates_final = report_cols_for_final_metrics[sas_mask_final]["Colonna"].tolist()
            for s_cand_f in sas_candidates_final:
                if s_cand_f in df_sample_metrics_final.columns: sas_final.append(s_cand_f)
        st.session_state["identified_qids_for_summary"] = qids_final
        st.session_state["identified_sas_for_summary"] = sas_final
        current_risk_metrics_final = {"k_anonymity_min": "N/D", "records_singoli": "N/D", "l_diversity": {}}
        if qids_final:
            try:
                k_min_f, rec_sing_f = calculate_k_anonymity(df_sample_metrics_final, qids_final)
                current_risk_metrics_final["k_anonymity_min"] = int(k_min_f) if k_min_f != float('inf') else "N/A (>N)"
                current_risk_metrics_final["records_singoli"] = int(rec_sing_f)
            except Exception as e_k_f:
                st.warning(f"Errore k-anonymity: {e_k_f}")
        if qids_final and sas_final:
            for sa_col_metric_f in sas_final:
                if sa_col_metric_f in df_sample_metrics_final.columns:
                    try:
                        l_min_f = calculate_l_diversity(df_sample_metrics_final, qids_final, sa_col_metric_f)
                        current_risk_metrics_final["l_diversity"][sa_col_metric_f] = {
                            "l_min": int(l_min_f) if l_min_f not in [float('inf'), 0] else (
                                "N/A" if l_min_f == 0 else "N/A (>N)")}
                    except Exception as e_l_f:
                        st.warning(f"Errore l-diversity per {sa_col_metric_f}: {e_l_f}")
        st.session_state["calculated_risk_metrics"] = current_risk_metrics_final

        file_name_for_llm_report = st.session_state.get("last_uploaded_filename", "File CSV")
        try:
            overall_md_final_result = get_llm_overall_csv_comment(
                report_cols_for_final_metrics,
                st.session_state.get("calculated_risk_metrics", {}),
                st.session_state.get("loss_of_utility_metrics"),
                st.session_state.get("identified_qids_for_summary", []),
                st.session_state.get("identified_sas_for_summary", []),
                mod_api_csv_selected,
                file_name_for_llm_report
            )
            st.session_state["overall_csv_comment"] = overall_md_final_result
        except Exception as e_rep_f:
            st.error(f"Errore generazione report finale CSV con LLM: {e_rep_f}")
            logger_streamlit.error(f"Errore in get_llm_overall_csv_comment: {e_rep_f}", exc_info=True)
            st.session_state["overall_csv_comment"] = f"Errore durante la generazione del report LLM: {e_rep_f}"
        csv_progress_bar.progress(100, text="Report finale CSV (o tentativo) generato.")
    else:
        st.info(
            "Report colonne combinato vuoto o non disponibile. Impossibile procedere con metriche di rischio finali e report LLM.")
        csv_progress_bar.progress(100, text="Analisi CSV terminata (dati colonne insufficienti).")

    csv_progress_bar.empty()
    st.success("✅ Analisi CSV potenziata completata!")


st.set_page_config(page_title="🛡️ DataSentinelAI", layout="wide")
st.title("🛡️ DataSentinelAI")
st.markdown(
    """
    Questo strumento analizza il tuo documento (PDF/TXT/DOCX) o dataset CSV per rilevare PII 
    e valutarne i rischi di privacy. Per i CSV, include analisi specifiche per colonne 
    numeriche/data e una stima della perdita di utilità post-generalizzazione. 
    Per i PDF, è possibile generare una versione redatta.
    """
)
with st.expander("ℹ️ Come preparare i dati"):
    st.markdown(
        """
        - **PDF/DOCX/TXT:** Per risultati ottimali, usa testo selezionabile. File molto grandi o basati su immagini potrebbero avere performance ridotte o generare analisi incomplete. L'OCR è tentato per PDF immagine.
        - **CSV:** Usa intestazioni chiare. Le colonne testuali sono analizzate da LLM (se un modello è selezionato). Colonne numeriche e temporali sono analizzate con regole e generalizzate automaticamente se considerate Quasi-Identificatori. Puoi configurare i parametri di generalizzazione.
        - **Testo libero:** Incolla direttamente.
        """
    )

st.subheader("1. Carica i tuoi dati")

default_session_state = {
    "raw_text_input": None, "original_csv_df": None, "current_file_ext": None,
    "last_uploaded_filename": None, "text_area_content": "",
    "original_pdf_bytes": None, "last_uploaded_file_object": None,
    "reports_text": None, "csv_analysis_report_df": None, "csv_anon_df": None,
    "overall_csv_comment": None, "calculated_risk_metrics": None,
    "identified_qids_for_summary": None, "identified_sas_for_summary": None,
    "selected_csv_model_name": None, "loss_of_utility_metrics": None,
    "num_bins_for_numeric": 5, "granularity_for_date": "M",
    "pdf_extraction_progress_object": None, "redacted_pdf_output_bytes": None,
    "pdf_redaction_mode_selector": "placeholder"
}
for k_init, v_val_init in default_session_state.items():
    if k_init not in st.session_state: st.session_state[k_init] = v_val_init

MAX_FILE_SIZE_MB = 50
uploaded_file_obj = st.file_uploader(
    "Carica file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"],
    help=f"Scegli un file. Attenzione a file molto grandi (>{MAX_FILE_SIZE_MB}MB)."
)

input_changed_flag_global = False
if uploaded_file_obj is not None:
    if st.session_state.get("last_uploaded_filename") != uploaded_file_obj.name or \
            st.session_state.get("last_uploaded_file_object") != uploaded_file_obj:
        input_changed_flag_global = True
        st.session_state["last_uploaded_filename"] = uploaded_file_obj.name
        st.session_state["last_uploaded_file_object"] = uploaded_file_obj

        uploaded_file_obj.seek(0)
        file_bytes_content = uploaded_file_obj.read()
        uploaded_file_obj.seek(0)

        if uploaded_file_obj.type == "application/pdf":
            st.session_state["original_pdf_bytes"] = file_bytes_content
        else:
            st.session_state["original_pdf_bytes"] = None

        file_size_mb_val = len(file_bytes_content) / (1024 * 1024)
        if file_size_mb_val > MAX_FILE_SIZE_MB:
            st.warning(
                f"Il file '{uploaded_file_obj.name}' è di circa {file_size_mb_val:.1f} MB. L'analisi potrebbe "
                f"richiedere molto tempo.")

        detected_ext_val = detect_extension(uploaded_file_obj).lower()
        st.session_state["current_file_ext"] = detected_ext_val
        st.success(f"File caricato: **{uploaded_file_obj.name}** (Rilevato come: {detected_ext_val.upper()})")

        current_raw_text = ""
        if detected_ext_val == ".pdf":
            try:
                progress_bar_pdf_key = "pdf_extraction_progress_object"
                st.session_state[progress_bar_pdf_key] = st.progress(0, text="Avvio estrazione PDF...")

                current_raw_text = extract_text(
                    uploaded_file_obj, use_ocr_for_pdf=True,
                    pdf_max_pages_to_sample=None,
                    pdf_progress_bar_key=progress_bar_pdf_key
                )
                current_raw_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', current_raw_text)
                if len(current_raw_text.strip()) < 100 and file_size_mb_val > 0.05:
                    st.warning("Il PDF caricato contiene poco testo estratto. Potrebbe essere scannerizzato.")
                if st.session_state.get(progress_bar_pdf_key) is not None: st.session_state[
                    progress_bar_pdf_key].empty()
            except Exception as e_extract:
                st.error(f"Errore estraendo il testo dal PDF: {e_extract}")
                logger_streamlit.error(f"Errore estrazione PDF: {e_extract}", exc_info=True)
                current_raw_text = ""
                if st.session_state.get("pdf_extraction_progress_object") is not None: st.session_state[
                    "pdf_extraction_progress_object"].empty()

        elif detected_ext_val in [".docx", ".txt"]:
            try:
                current_raw_text = extract_text(uploaded_file_obj)
            except Exception as e_extract_doc_txt:
                st.error(f"Errore estraendo il testo da {detected_ext_val.upper()}: {e_extract_doc_txt}")
                current_raw_text = ""

        elif detected_ext_val == ".csv":
            try:
                uploaded_file_obj.seek(0)
                df_loaded_content = pd.read_csv(uploaded_file_obj)
                st.session_state["original_csv_df"] = df_loaded_content
                uploaded_file_obj.seek(0)
                current_raw_text = extract_text(uploaded_file_obj)
                st.caption("Anteprima CSV (prime 5 righe):")
                st.dataframe(df_loaded_content.head(5), use_container_width=True, height=200)
            except Exception as e_csv_read:
                st.error(f"Errore leggendo il CSV: {e_csv_read}")
                st.session_state["original_csv_df"] = None
                current_raw_text = ""

        st.session_state["raw_text_input"] = current_raw_text
        if detected_ext_val != ".csv": st.session_state["original_csv_df"] = None
        st.session_state["text_area_content"] = ""
    else:
        if st.session_state.get("last_uploaded_filename") and st.session_state.get("current_file_ext"):
            st.success(
                f"File attuale: **{st.session_state.last_uploaded_filename}** (Tipo: {st.session_state.current_file_ext.upper()})")
        if st.session_state.current_file_ext == ".csv" and st.session_state.original_csv_df is not None:
            st.caption("Anteprima dataset CSV (prime 5 righe):")
            st.dataframe(st.session_state.original_csv_df.head(5), use_container_width=True, height=200)

if input_changed_flag_global:
    keys_to_reset = [
        "reports_text", "csv_analysis_report_df", "csv_anon_df", "overall_csv_comment",
        "calculated_risk_metrics", "loss_of_utility_metrics", "identified_qids_for_summary",
        "identified_sas_for_summary", "redacted_pdf_output_bytes", "anonymized_text_output"
    ]
    for key_to_reset in keys_to_reset: st.session_state.pop(key_to_reset, None)
    if st.session_state.last_uploaded_filename is None and not st.session_state.text_area_content.strip():
        st.session_state.current_file_ext = None

active_raw_text_input = st.session_state.get("raw_text_input")
active_df_input = st.session_state.get("original_csv_df")
active_file_type_input = st.session_state.get("current_file_ext")

if active_file_type_input == ".csv" and active_df_input is not None and not active_df_input.empty:
    with st.expander("⚙️ Configurazione Avanzata Analisi CSV", expanded=False):
        if LLM_MODELS:
            csv_model_options = list(LLM_MODELS.keys())
            csv_default_model_idx = 0
            if st.session_state.get("selected_csv_model_name") in csv_model_options:
                csv_default_model_idx = csv_model_options.index(st.session_state.selected_csv_model_name)
            csv_selected_model = st.selectbox(
                "Modello LLM per analisi colonne testuali CSV:",
                csv_model_options, index=csv_default_model_idx, key="sb_csv_model_selector_v4_full",
                help="Modello per analizzare il contenuto delle colonne testuali."
            )
            st.session_state["selected_csv_model_name"] = csv_selected_model
        else:
            st.caption("Nessun modello LLM configurato; l'analisi delle colonne testuali non userà LLM.")
            st.session_state["selected_csv_model_name"] = None

st.subheader("2. Analizza i dati")
analysis_btn_disabled_status = not (
        active_raw_text_input or (active_df_input is not None and not active_df_input.empty))
analysis_btn_label_text = "⚠️ Carica un file o incolla del testo"

if active_file_type_input in [".pdf", ".docx", ".txt"] and active_raw_text_input:
    analysis_btn_label_text = "🚀 Analizza Testo/Documento"
elif active_file_type_input == ".csv" and (active_df_input is not None and not active_df_input.empty):
    csv_selected_model_name_for_btn = st.session_state.get("selected_csv_model_name")
    analysis_btn_label_text = f"📊 Analizza CSV"
    if LLM_MODELS and csv_selected_model_name_for_btn:
        analysis_btn_label_text += f" (testo con {csv_selected_model_name_for_btn})"
    elif LLM_MODELS and not csv_selected_model_name_for_btn:
        analysis_btn_label_text = "📊 Seleziona Modello LLM per colonne testuali CSV"
        analysis_btn_disabled_status = True

main_analyze_button = st.button(analysis_btn_label_text, disabled=analysis_btn_disabled_status,
                                use_container_width=True)
if analysis_btn_disabled_status and active_file_type_input == ".csv" and not st.session_state.get(
        "selected_csv_model_name") and LLM_MODELS:
    st.caption("ℹ️ Seleziona un modello LLM (sopra) per l'analisi delle colonne testuali del CSV.")

if main_analyze_button:
    keys_to_clear = [
        "reports_text", "csv_analysis_report_df", "csv_anon_df", "overall_csv_comment",
        "calculated_risk_metrics", "loss_of_utility_metrics",
        "identified_qids_for_summary", "identified_sas_for_summary", "redacted_pdf_output_bytes"
    ]
    for key_cl in keys_to_clear: st.session_state.pop(key_cl, None)

    if active_raw_text_input and active_file_type_input in [".pdf", ".docx", ".txt"]:
        _perform_text_analysis(active_raw_text_input)
    elif active_df_input is not None and not active_df_input.empty and active_file_type_input == ".csv":
        num_bins_to_pass = st.session_state.get("num_bins_for_numeric", 5)
        granularity_date_to_pass = st.session_state.get("granularity_for_date", "M")
        _perform_csv_analysis(
            active_df_input,
            st.session_state.get("selected_csv_model_name"),
            num_bins=num_bins_to_pass,
            granularity_date=granularity_date_to_pass
        )

# --- Sezione 3: Visualizza Risultati ed Esporta ---
has_text_results = st.session_state.get("reports_text") is not None
csv_report_df_val = st.session_state.get("csv_analysis_report_df")
has_csv_report_cols_val = csv_report_df_val is not None and not csv_report_df_val.empty
csv_anon_df_val = st.session_state.get("csv_anon_df")
has_csv_anon_df_val = csv_anon_df_val is not None and not csv_anon_df_val.empty
has_csv_overall_val = st.session_state.get("overall_csv_comment") is not None
has_loss_metrics_val = st.session_state.get("loss_of_utility_metrics") is not None
has_redacted_pdf_val = st.session_state.get("redacted_pdf_output_bytes") is not None
show_results_ui_section_val = has_text_results or has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val or (
        active_file_type_input == ".pdf" and st.session_state.get("original_pdf_bytes") is not None)

if show_results_ui_section_val:
    st.subheader("3. Visualizza Risultati ed Esporta")
    result_tab_titles_list = []
    if has_text_results or (active_file_type_input == ".pdf" and st.session_state.get("original_pdf_bytes")):
        result_tab_titles_list.append("🔍 Testo/Documento & Redazione PDF")
    if has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val:
        if "📊 CSV Analisi Dettagliata" not in result_tab_titles_list: result_tab_titles_list.append(
            "📊 CSV Analisi Dettagliata")

    if not result_tab_titles_list and main_analyze_button:
        st.info(
            "Analisi completata, ma nessun risultato specifico da visualizzare per i criteri attuali o tipo di file.")
    elif result_tab_titles_list:
        displayed_tabs_list = st.tabs(result_tab_titles_list)
        current_display_tab_idx_val = 0

        if "🔍 Testo/Documento & Redazione PDF" in result_tab_titles_list:
            with displayed_tabs_list[current_display_tab_idx_val]:
                current_display_tab_idx_val += 1
                current_text_reports_val = st.session_state.get("reports_text")
                if current_text_reports_val:
                    st.header("Report PII da LLM (Analisi su Testo/Documento)")
                    # ... (la parte che visualizza i report rimane invariata) ...
                    for model_name, report in current_text_reports_val.items():
                        st.markdown(f"#### Report da: **{model_name}**")

                        # 1) Mostro il riassunto in ogni caso
                        summary = report.get("summary", "")
                        st.markdown(f"**Riassunto LLM:** {summary}")

                        # 2) Se ha trovato entità, le metto in tabella
                        if report.get("found"):
                            ents = report.get("entities", [])
                            if ents:
                                df = pd.DataFrame(ents)
                                st.dataframe(df)
                            else:
                                st.info("Found=True ma nessuna entità da mostrare.")
                        else:
                            st.info("Nessuna PII trovata da questo modello.")

                        st.markdown("---")

                # --- BLOCCO ANONIMIZZAZIONE E REDAZIONE FORTIFICATO ---
                if st.session_state.current_file_ext == ".txt":
                    # --- Inizio blocco Anonimizzazione Testo aggiornato ---
                    st.markdown("---")
                    st.subheader("✉️ Azioni su Documento (Anonimizzazione Testo)")

                    # 1) Prendo solo i report validi (found=True)
                    reports = st.session_state.get("reports_text") or {}
                    valid_reports = {
                        name: rep
                        for name, rep in reports.items()
                        if isinstance(rep, dict) and rep.get("found")
                    }

                    # 2) Pulsante cascade
                    if valid_reports:
                        if st.button("✏️ Anonimizza in cascade con tutti i modelli"):
                            text_to_anon = st.session_state["raw_text_input"]
                            for model_name, report in valid_reports.items():
                                api_id = LLM_MODELS[model_name]
                                text_to_anon = edit_document(text_to_anon, report, api_id)
                            st.session_state["anonymized_text_output"] = text_to_anon
                            st.success("✏️ Testo anonimizzato in cascade!")

                    st.markdown("---")

                    # 3) Pulsanti per ciascun modello singolo
                    for model_name, report in valid_reports.items():
                        api_id = LLM_MODELS[model_name]
                        btn_key = f"anon_{model_name}"
                        if st.button(f"🖊️ Anonimizza con {model_name}", key=btn_key):
                            with st.spinner(f"Anonimizzazione con {model_name}…"):
                                anon = edit_document(
                                    st.session_state["raw_text_input"],
                                    report,
                                    api_id
                                )
                                st.session_state["anonymized_text_output"] = anon
                                st.success(f"Testo anonimizzato con {model_name}!")

                    # 4) Download solo se ho un risultato
                    anon_txt = st.session_state.get("anonymized_text_output")
                    if anon_txt:
                        original_fn = st.session_state.get("last_uploaded_filename", "documento.txt")
                        base, _ = os.path.splitext(original_fn)
                        st.download_button(
                            label="📥 Scarica Documento Anonimizzato (.txt)",
                            data=anon_txt.encode("utf-8"),
                            file_name=f"{base}_anonimizzato.txt",
                            mime="text/plain"
                        )
                    # --- Fine blocco Anonimizzazione Testo aggiornato ---

                # --- BLOCCO REDAZIONE PDF ---
                # --- BLOCCO REDAZIONE PDF ESTESO ---
                if active_file_type_input == ".pdf" and st.session_state.get("original_pdf_bytes"):
                    st.markdown("---")
                    st.subheader("📝 Redazione PDF")

                    # 1) Preparo due mappe di redazione:
                    #    a) cascade: unisco tutte le entità di tutti i modelli
                    cascade_terms = _prepare_sensitive_terms_for_pdf_redaction(
                        st.session_state.get("reports_text"), None
                    )

                    #    b) per‐modello: estraggo una mappa testo→placeholder per ciascun modello
                    per_model_terms: Dict[str, Dict[str, str]] = {}
                    for model_name, report in (st.session_state.get("reports_text") or {}).items():
                        if isinstance(report, dict) and report.get("entities"):
                            m = {}
                            for ent in report["entities"]:
                                txt = ent.get("text", "").strip()
                                typ = ent.get("type", "").upper()
                                placeholder = {
                                    "PERSON": "[NOME]", "EMAIL": "[EMAIL]", "LOCATION": "[LUOGO]",
                                    "ADDRESS": "[INDIRIZZO]", "PHONE_NUMBER": "[TELEFONO]"
                                }.get(typ, f"[{typ}]")
                                if txt: m[txt] = placeholder
                            if m:
                                per_model_terms[model_name] = m

                    # 2) Pulsante “in cascade” (tutti i modelli insieme)
                    # --- PULSANTE CASCADE ANCHE PER PDF ---
                    cascade_terms = _prepare_sensitive_terms_for_pdf_redaction(
                        st.session_state.get("reports_text"), None
                    )
                    if cascade_terms:
                        if st.button("🖊️ Genera PDF Redatto (cascade)", key="btn_pdf_cascade"):
                            with st.spinner("Creazione del PDF redatto in cascade…"):
                                redacted_bytes = redact_pdf_in_memory(
                                    st.session_state["original_pdf_bytes"],
                                    cascade_terms,
                                    redaction_mode=st.session_state.get("pdf_redaction_mode_selector", "placeholder")
                                )
                            base, _ = os.path.splitext(st.session_state.get("last_uploaded_filename", "documento.pdf"))
                            st.download_button(
                                label="📥 Scarica PDF Redatto (cascade)",
                                data=redacted_bytes,
                                file_name=f"{base}_redatto_cascade.pdf",
                                mime="application/pdf",
                                key="dl_redacted_pdf_cascade_btn"
                            )

                    # 3) Pulsanti per ciascun modello
                    for model_name, terms in per_model_terms.items():
                        if st.button(f"🖊️ Redazione PDF con {model_name}", key=f"btn_pdf_{model_name}"):
                            redacted = redact_pdf_in_memory(
                                st.session_state["original_pdf_bytes"],
                                terms,
                                redaction_mode=st.session_state.get("pdf_redaction_mode_selector", "placeholder")
                            )
                            base, _ = os.path.splitext(st.session_state.get("last_uploaded_filename", "documento.pdf"))
                            st.download_button(
                                f"📥 Scarica {model_name} redatto",
                                data=redacted,
                                file_name=f"{base}_redatto_{model_name}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{model_name}"
                            )

                elif not current_text_reports_val and active_file_type_input != ".csv" and main_analyze_button:
                    st.info("Analisi testuale eseguita, ma nessun report PII generato.")
        if "📊 CSV Analisi Dettagliata" in result_tab_titles_list:
            with displayed_tabs_list[current_display_tab_idx_val]:
                csv_report_cols_data_disp_val = st.session_state.get("csv_analysis_report_df")
                csv_overall_md_content_disp_val = st.session_state.get("overall_csv_comment")
                csv_df_anon_content_disp_val = st.session_state.get("csv_anon_df")
                utility_metrics_display_val = st.session_state.get("loss_of_utility_metrics")
                st.header("Risultati Analisi CSV")

                if csv_report_cols_data_disp_val is not None and not csv_report_cols_data_disp_val.empty:
                    st.markdown("#### Dettaglio Analisi per Colonna (Testuali LLM, Numeriche/Date Regole)")
                    df_user_csv_display_val = csv_report_cols_data_disp_val

                    columns_to_show = ["Colonna", "CategoriaLLM", "Esempi", "MetodoSuggerito", "Motivazione"]
                    columns_present = [col for col in columns_to_show if col in df_user_csv_display_val.columns]

                    df_tabella = df_user_csv_display_val[columns_present].rename(columns={
                        "CategoriaLLM": "Categoria",
                        "MetodoSuggerito": "Metodo Suggerito"
                    })

                    # --- NUOVA SOLUZIONE: LAYOUT A SCHEDE PER MASSIMA LEGGIBILITÀ ---
                    for _, row in df_tabella.iterrows():
                        with st.container(border=True):
                            # Riga 1: Nome Colonna e Categoria
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown(f"**Colonna:** `{row['Colonna']}`")
                            with col2:
                                # La funzione colored_badge deve essere definita o importata in streamlit_app.py
                                st.markdown(f"**Categoria:** {colored_badge(row['Categoria'])}", unsafe_allow_html=True)

                            st.divider()

                            # Riga 2: Motivazione e Metodo
                            col3, col4 = st.columns([2, 1])
                            with col3:
                                st.markdown("**Motivazione Dettagliata:**")
                                # Sostituisce il newline con <br> per la resa in HTML e usa il quote block
                                motivation_html = row['Motivazione'].replace('\n', '<br>')
                                st.markdown(f"> {motivation_html}", unsafe_allow_html=True)
                            with col4:
                                st.markdown("**Metodo Suggerito:**")
                                st.info(row['Metodo Suggerito'])

                            st.markdown(f"**Esempi:** `{row['Esempi']}`")
                        # Aggiunge un piccolo spazio tra le schede
                        st.markdown("<br>", unsafe_allow_html=True)
                    # --- FINE NUOVA SOLUZIONE ---
                # --- FINE BLOCCO MODIFICATO ---
                elif csv_report_cols_data_disp_val is not None and csv_report_cols_data_disp_val.empty and main_analyze_button:
                    st.info("L'analisi delle colonne del CSV non ha prodotto un report dettagliato.")

                if csv_overall_md_content_disp_val:
                    st.markdown("---")
                    st.markdown("### 📈 Report Privacy Complessivo (CSV)", unsafe_allow_html=True)
                    st.markdown(csv_overall_md_content_disp_val, unsafe_allow_html=True)
                    if st.button("Rimuovi Report Privacy Complessivo", key="clear_overall_csv_btn_v5"):
                        st.session_state["overall_csv_comment"] = None
                        st.rerun()

                if isinstance(utility_metrics_display_val, dict) and utility_metrics_display_val:
                    st.markdown("---")
                    st.markdown("#### 📉 Metrica di Perdita di Utilità (Post-Generalizzazione)")
                    st.write(f"- Righe originali: **{utility_metrics_display_val.get('n_righe_originali', 'N/D')}**")
                    st.write(
                        f"- Righe generalizzate: **{utility_metrics_display_val.get('n_righe_anonimizzate', 'N/D')}**")
                    for col_metric_util_val, metrics_data_util_val in utility_metrics_display_val.items():
                        if isinstance(metrics_data_util_val, dict):
                            st.write(f"- Colonna esempio per varianza: **{col_metric_util_val}**")
                            var_o = metrics_data_util_val.get('var_originale', 'N/D')
                            var_g = metrics_data_util_val.get('var_generalizzata_stimata', 'N/D')
                            perc_p = metrics_data_util_val.get('percentuale_preserved_var (%)', 'N/D')
                            st.write(
                                f"  - Varianza originale: {'{:.2f}'.format(var_o) if isinstance(var_o, (int, float)) else var_o}")
                            st.write(
                                f"  - Varianza generalizzata: {'{:.2f}'.format(var_g) if isinstance(var_g, (int, float)) else var_g}")
                            st.write(f"  - % Varianza Preservata: {perc_p}%")

                st.markdown("---")
                # --- CSV DOWNLOAD “cascade” e “per-modello” ---
                if isinstance(csv_df_anon_content_disp_val, pd.DataFrame) and not csv_df_anon_content_disp_val.empty:
                    base, ext = os.path.splitext(st.session_state.get("last_uploaded_filename", "dataset.csv"))

                    # 1) Download cascade (tutte le colonne generalizzate insieme)
                    csv_cascade_bytes = csv_df_anon_content_disp_val.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Scarica CSV Generalizzato (cascade)",
                        data=csv_cascade_bytes,
                        file_name=f"{base}_anonimizzato_cascade{ext}",
                        mime="text/csv",
                        key="dl_anon_csv_cascade"
                    )

                    # 2) Download per-modello: per ogni LLM, ricomputi il df anonimizzato solo con le entità di quel modello
                    reports = st.session_state.get("reports_text") or {}
                    for model_name, report in reports.items():
                        if isinstance(report, dict) and report.get("entities"):
                            # ricava mappa text→placeholder per questo modello
                            mapping = {
                                ent["text"].strip(): f"[{ent['type'].upper()}]"
                                for ent in report["entities"]
                                if ent.get("text")
                            }
                            # applica localmente la redazione
                            df_single = csv_df_anon_content_disp_val.copy()
                            for col in df_single.select_dtypes(include=["object"]):
                                df_single[col] = df_single[col].astype(str).replace(mapping, regex=True)

                            csv_single_bytes = df_single.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label=f"📥 Scarica CSV anonimo con {model_name}",
                                data=csv_single_bytes,
                                file_name=f"{base}_anon_{model_name}{ext}",
                                mime="text/csv",
                                key=f"dl_anon_csv_{model_name}"
                            )

else:
    if not (active_raw_text_input or (active_df_input is not None and not active_df_input.empty)):
        st.caption(
            "Carica un file o incolla del testo, seleziona opzioni e clicca 'Analizza' per visualizzare i risultati.")
    elif not main_analyze_button:
        st.caption("Esegui un'analisi per visualizzare i risultati.")
