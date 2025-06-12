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
from modules.generazione_testo import generate_report_on_full_text
from modules.analyisis_cvs import analyze_and_anonymize_csv, get_llm_overall_csv_comment
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

    Args:
        pdf_bytes: Bytes del PDF originale.
        sensitive_terms: Dizionario {termine_da_cercare: placeholder}.
        redaction_mode: "placeholder" per sostituire con testo, "blackbox" per coprire con nero.

    Returns:
        Bytes del PDF redatto.
    """
    logger_streamlit.info(
        f"Avvio redazione PDF in memoria. Termini da redigere: {len(sensitive_terms)}. Modalità: {redaction_mode}")
    if not pdf_bytes or not sensitive_terms:
        logger_streamlit.warning("PDF bytes o termini sensibili mancanti per la redazione.")
        return pdf_bytes

    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_redactions_added = 0
        # Ordina i termini dal più lungo al più corto per evitare redazioni parziali (es. "Mario Rossi" prima di "Mario")
        sorted_terms_for_redaction = sorted(sensitive_terms.items(), key=lambda item: len(item[0]), reverse=True)

        for page_num, page in enumerate(doc):
            page_redactions_count_on_page = 0
            for term, placeholder_text in sorted_terms_for_redaction:
                if not term.strip(): continue
                # Cerca tutte le istanze del termine nella pagina
                text_instances_found = page.search_for(term, flags=fitz.TEXT_INHIBIT_SPACES, quads=False)

                for inst_rect in text_instances_found:
                    # Impostazioni per l'annotazione di redazione
                    fill_color_redact = (0.8, 0.8, 0.8) if redaction_mode == "placeholder" else (0, 0, 0)
                    text_for_annot = placeholder_text if redaction_mode == "placeholder" else ""

                    # Aggiunge l'annotazione di redazione
                    page.add_redact_annot(
                        inst_rect,
                        text=text_for_annot,
                        fill=fill_color_redact,
                        text_color=(0, 0, 0),  # Imposta il testo del placeholder a nero per contrasto
                        cross_out=False
                    )
                    page_redactions_count_on_page += 1

            if page_redactions_count_on_page > 0:
                logger_streamlit.info(
                    f"Pagina PDF {page_num + 1}: Aggiunte {page_redactions_count_on_page} annotazioni di redazione.")
                num_redactions_added += page_redactions_count_on_page

        if num_redactions_added > 0:
            logger_streamlit.info(
                f"Totale annotazioni di redazione aggiunte: {num_redactions_added}. Procedo all'applicazione delle redazioni.")

            # --- APPLICAZIONE DELLE REDAZIONI (MODIFICA CHIAVE) ---
            # Questo passaggio è fondamentale per "imprimere" le redazioni nel PDF.
            try:
                # Metodo preferito per PyMuPDF >= 1.19.0
                doc.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
                logger_streamlit.info("Redazioni applicate con successo usando doc.apply_redactions().")
            except AttributeError:
                # Fallback per versioni più vecchie: applica pagina per pagina
                logger_streamlit.warning(
                    "Metodo 'doc.apply_redactions()' non trovato. Fallback all'applicazione per pagina.")
                for page in doc:
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
                logger_streamlit.info("Redazioni applicate con successo (fallback per pagina).")
        else:
            logger_streamlit.info("Nessuna occorrenza dei termini sensibili trovata nel PDF per la redazione.")

        # Salva il PDF, ora con le redazioni applicate, in un buffer in memoria
        out_buffer = BytesIO()
        doc.save(out_buffer, garbage=3, deflate=True, clean=True)
        doc.close()
        logger_streamlit.info("PDF redatto salvato in memoria.")
        return out_buffer.getvalue()

    except Exception as e:
        logger_streamlit.error(f"Errore generico durante la redazione del PDF: {e}", exc_info=True)
        if doc: doc.close()
        st.error(f"Errore durante la redazione del PDF: {e}. Verrà offerto il download del file originale.")
        return pdf_bytes


def _prepare_sensitive_terms_for_pdf_redaction(reports_text_state: Optional[Dict[str, Any]], _) -> Dict[str, str]:
    # Il secondo parametro (general_report_data_state) non è più usato qui per semplicità
    sensitive_terms_map: Dict[str, str] = {}
    pii_entries: List[Dict[str, str]] = []
    placeholder_map = {
        "PERSON": "[NOME]", "EMAIL": "[EMAIL]", "LOCATION": "[LUOGO]", "ADDRESS": "[INDIRIZZO]",
        "PHONE_NUMBER": "[TELEFONO]", "ID_NUMBER": "[ID]", "CODICE FISCALE": "[CF]",
        "ORGANIZATION": "[ORGANIZZAZIONE]", "DATE": "[DATA]",
        # Aggiungi altri tipi dalla tua lista PII_TYPES
        "HEALTH_CONDITION": "[INFO_SALUTE]", "DIAGNOSIS": "[DIAGNOSI]", "MEDICATION": "[FARMACO]"
    }
    if reports_text_state:
        for model_report in reports_text_state.values():
            if isinstance(model_report, dict) and model_report.get("found") and isinstance(model_report.get("entities"),
                                                                                           list):
                for entity in model_report["entities"]:
                    if isinstance(entity, dict) and entity.get("text") and entity.get("type"):
                        if str(entity["type"]).lower() in PII_TYPES_LOWER_SET:
                            pii_entries.append({"text": str(entity["text"]).strip(),
                                                "type": str(entity["type"]).upper().replace(" ", "_")})
        unique_entities = [dict(t) for t in {tuple(d.items()) for d in pii_entries}]
        unique_entities.sort(key=lambda x: len(x["text"]), reverse=True)
        for entry in unique_entities:
            term, term_type = entry["text"], entry["type"]
            if term and not (term.startswith("[") and term.endswith("]")):
                placeholder = placeholder_map.get(term_type, f"[{term_type}_REDATTO]")
                if term not in sensitive_terms_map: sensitive_terms_map[term] = placeholder
    logger_streamlit.info(f"Termini per redazione PDF: {len(sensitive_terms_map)}")
    return sensitive_terms_map


# --- _perform_text_analysis AGGIORNATA (SENZA CHUNKING INTERNO) ---
def _perform_text_analysis(text_to_analyze: str):
    logger_streamlit.info(f"Avvio _perform_text_analysis su testo di lunghezza {len(text_to_analyze)}.")
    st.info("⏳ Esecuzione analisi PII sul testo completo...")
    reports_text_single_model_results: Dict[str, Dict[str, Any]] = {}

    if not LLM_MODELS:
        st.error("Nessun modello LLM configurato in modules/config.py.")
        logger_streamlit.error("Nessun modello LLM configurato.")
        st.session_state["reports_text"] = reports_text_single_model_results
        return

    if not text_to_analyze or not text_to_analyze.strip():
        st.warning("Il testo fornito per l'analisi è vuoto.")
        logger_streamlit.warning("Testo vuoto fornito a _perform_text_analysis.")
        st.session_state["reports_text"] = reports_text_single_model_results
        return

    # Avviso per testi molto lunghi (da gestire prima della chiamata o nella UI)
    MAX_CHARS_FOR_FULL_TEXT_ANALYSIS = 50000  # Esempio, da adattare
    if len(text_to_analyze) > MAX_CHARS_FOR_FULL_TEXT_ANALYSIS:
        st.warning(
            f"⚠️ Il testo è molto lungo ({len(text_to_analyze)} caratteri). L'analisi completa potrebbe richiedere tempo o superare i limiti del modello LLM. Considera di analizzare porzioni più piccole se l'analisi fallisce o è troppo lenta.")

    text_analysis_progress = st.progress(0, text="Avvio analisi testo completo...")
    total_models = len(LLM_MODELS)

    for model_idx, (model_name, model_api_id) in enumerate(LLM_MODELS.items()):
        progress_percent = int(((model_idx + 1) / total_models) * 100)
        text_analysis_progress.progress(progress_percent,
                                        text=f"Analisi PII con {model_name} ({model_idx + 1}/{total_models})...")
        logger_streamlit.info(f"Analisi testo completo con modello: {model_name}")
        try:
            # Chiamata diretta a generate_report_on_full_text
            report = generate_report_on_full_text(text_to_analyze, model_api_id)
            reports_text_single_model_results[model_name] = report
            if "error_details" in report and report["error_details"]:
                logger_streamlit.error(
                    f"Errore da {model_name}: {report.get('summary')} - Dettagli: {report.get('error_details')}")
        except Exception as e:
            error_msg = f"Errore grave durante l'analisi con {model_name}: {type(e).__name__} - {e}"
            st.error(error_msg)
            logger_streamlit.error(error_msg, exc_info=True)
            reports_text_single_model_results[model_name] = {
                "found": False, "entities": [], "summary": error_msg,
                "error_details": str(e), "raw_output_on_error": ""
            }

    text_analysis_progress.empty()
    st.session_state["reports_text"] = reports_text_single_model_results
    st.success("✅ Analisi PII su testo completo terminata!")
    logger_streamlit.info("_perform_text_analysis (full text) completata.")


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
        csv_progress_bar.empty();
        st.error("Analisi CSV interrotta.");
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
                        low_str, high_str = x_fascia.split("-");
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

st.subheader("1. Carica o incolla i tuoi dati")

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
                f"Il file '{uploaded_file_obj.name}' è di circa {file_size_mb_val:.1f} MB. L'analisi potrebbe richiedere molto tempo.")

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
                st.session_state["original_csv_df"] = None;
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

current_text_area_val = st.session_state.get("text_area_content", "")
text_area_user_input = st.text_area(
    "Oppure incolla qui il tuo testo", value=current_text_area_val,
    placeholder="Copia‐incolla testo...", height=150, key="text_area_main_v3",
    disabled=(uploaded_file_obj is not None)
)
if uploaded_file_obj is None:
    if text_area_user_input != current_text_area_val:
        input_changed_flag_global = True
        st.session_state["text_area_content"] = text_area_user_input
    if text_area_user_input and text_area_user_input.strip():
        if st.session_state.current_file_ext != ".txt" or st.session_state.raw_text_input != text_area_user_input:
            input_changed_flag_global = True
        st.session_state["raw_text_input"] = text_area_user_input;
        st.session_state["current_file_ext"] = ".txt"
        st.session_state["original_csv_df"] = None;
        st.session_state["original_pdf_bytes"] = None
        st.session_state["last_uploaded_filename"] = None;
        st.session_state["last_uploaded_file_object"] = None
    elif not (text_area_user_input and text_area_user_input.strip()) and st.session_state.get("raw_text_input"):
        if st.session_state.current_file_ext == ".txt":
            input_changed_flag_global = True;
            st.session_state["raw_text_input"] = None;
            st.session_state["current_file_ext"] = None

if input_changed_flag_global:
    st.info(
        "Input cambiato/rimosso. Eventuali risultati precedenti sono stati azzerati. Esegui una nuova analisi se necessario.")
    keys_to_reset = [
        "reports_text", "csv_analysis_report_df", "csv_anon_df", "overall_csv_comment",
        "calculated_risk_metrics", "loss_of_utility_metrics", "identified_qids_for_summary",
        "identified_sas_for_summary", "redacted_pdf_output_bytes"
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

        st.markdown("###### Parametri di Generalizzazione (per utilità ML):")
        num_bins_val = st.slider(
            "Numero di bin per generalizzazione numerica (quantili):",
            min_value=2, max_value=20, value=st.session_state.get("num_bins_for_numeric", 5), step=1,
            key="num_bins_slider",
            help="Più bin = meno perdita di informazione, ma potenziale minor privacy."
        )
        st.session_state["num_bins_for_numeric"] = num_bins_val

        granularity_val = st.selectbox(
            "Livello di granularità per generalizzazione date:",
            options=["M", "Q", "Y"], index=["M", "Q", "Y"].index(st.session_state.get("granularity_for_date", "M")),
            format_func=lambda x: {"M": "Mese (YYYY-MM)", "Q": "Trimestre (YYYYQX)", "Y": "Anno (YYYY)"}.get(x),
            key="granularity_selector",
            help="Generalizza le date al livello scelto."
        )
        st.session_state["granularity_for_date"] = granularity_val

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
                    for model_name_txt_rep_val, report_data_txt_rep_val in current_text_reports_val.items():
                        st.markdown(f"#### Report da: **{model_name_txt_rep_val}**")
                        if isinstance(report_data_txt_rep_val, dict):
                            if report_data_txt_rep_val.get("error_details") or (
                                    isinstance(report_data_txt_rep_val.get("summary"),
                                               str) and report_data_txt_rep_val.get("summary", "").startswith(
                                "Error:")):
                                st.error(
                                    f"Errore per {model_name_txt_rep_val}: {report_data_txt_rep_val.get('summary', 'Dettagli non disponibili.')}")
                                if report_data_txt_rep_val.get("error_details"): st.caption(
                                    f"Dettagli errore: {report_data_txt_rep_val.get('error_details')}")
                                if report_data_txt_rep_val.get("raw_output_on_error"):
                                    with st.expander("Mostra output grezzo dell'errore LLM"):
                                        st.text(report_data_txt_rep_val.get("raw_output_on_error"))
                            elif report_data_txt_rep_val.get("found") and isinstance(
                                    report_data_txt_rep_val.get("entities"), list) and report_data_txt_rep_val.get(
                                "entities"):
                                df_entities_text_disp_val = pd.DataFrame(report_data_txt_rep_val["entities"])
                                st.markdown("**Entità sensibili trovate:**")
                                text_cols_to_show_val = ["type", "text", "context", "reasoning", "source_chunk_info"]
                                text_cols_present_val = [col for col in text_cols_to_show_val if
                                                         col in df_entities_text_disp_val.columns]
                                st.dataframe(df_entities_text_disp_val[text_cols_present_val], use_container_width=True,
                                             height=350)
                                with st.expander("🔒 Dettaglio PII (motivazioni)"):
                                    for _, entity_item_row_val in df_entities_text_disp_val.iterrows():
                                        st.markdown(
                                            f"- **{entity_item_row_val.get('type', 'N/A')}**: `{entity_item_row_val.get('text', 'N/A')}`")
                                        st.markdown(f"  - Contesto: _{entity_item_row_val.get('context', 'N/A')}_")
                                        st.markdown(f"  - Motivazione: _{entity_item_row_val.get('reasoning', 'N/A')}_")
                                        if "source_chunk_info" in entity_item_row_val and entity_item_row_val.get(
                                                'source_chunk_info') != "N/A":
                                            st.markdown(
                                                f"  - Provenienza: _{entity_item_row_val['source_chunk_info']}_")
                                        st.markdown("---")
                            else:
                                st.info(
                                    f"Nessuna entità sensibile trovata da {model_name_txt_rep_val} o il report non è valido.")
                            st.markdown(f"**Riassunto (LLM):** {report_data_txt_rep_val.get('summary', 'N/A')}")
                            try:
                                report_json_dl_str_val = json.dumps(report_data_txt_rep_val, ensure_ascii=False,
                                                                    indent=2)
                                st.download_button(
                                    label=f"Scarica Report JSON ({model_name_txt_rep_val})",
                                    data=report_json_dl_str_val,
                                    file_name=f"report_pii_testo_{model_name_txt_rep_val.replace(' ', '_')}.json",
                                    mime="application/json", key=f"dl_json_txt_{model_name_txt_rep_val}_v4",
                                    use_container_width=False)
                            except Exception as e_json_text_dl_val:
                                st.error(f"Errore preparazione JSON download: {e_json_text_dl_val}")
                        else:
                            st.text(str(report_data_txt_rep_val))
                        st.markdown("---")

                if active_file_type_input == ".pdf" and st.session_state.get("original_pdf_bytes"):
                    st.markdown("---")
                    st.subheader("📝 Redazione PDF (Beta)")
                    st.caption(
                        "Questa funzionalità tenta di coprire o sostituire i termini sensibili identificati (dai report LLM) direttamente nel file PDF.")

                    sensitive_terms_for_pdf_val = _prepare_sensitive_terms_for_pdf_redaction(
                        st.session_state.get("reports_text"), None
                    )
                    if not sensitive_terms_for_pdf_val and main_analyze_button:
                        st.warning(
                            "Nessun termine sensibile identificato dai report testuali per la redazione. Assicurati di aver eseguito l'analisi PII sul testo del PDF.")

                    if sensitive_terms_for_pdf_val or not main_analyze_button:
                        st.session_state["pdf_redaction_mode_selector"] = st.radio(
                            "Modalità di redazione PDF:", ("placeholder", "blackbox"),
                            index=["placeholder", "blackbox"].index(
                                st.session_state.get("pdf_redaction_mode_selector", "placeholder")),
                            captions=["Sostituisce con placeholder (es. [NOME]).", "Copre con riquadro nero."],
                            horizontal=True, key="pdf_redaction_mode_radio_v3"
                        )
                        can_redact_now = bool(st.session_state.get("original_pdf_bytes")) and bool(
                            sensitive_terms_for_pdf_val)
                        if st.button("🖊️ Genera PDF Redatto", key="btn_gen_redacted_pdf_v3", use_container_width=True,
                                     disabled=not can_redact_now):
                            with st.spinner("Creazione del PDF redatto in corso..."):
                                redacted_bytes_content = redact_pdf_in_memory(
                                    st.session_state["original_pdf_bytes"],
                                    sensitive_terms_for_pdf_val,
                                    redaction_mode=st.session_state.get("pdf_redaction_mode_selector", "placeholder")
                                )
                                st.session_state["redacted_pdf_output_bytes"] = redacted_bytes_content
                                st.success("PDF redatto generato!")
                        if not can_redact_now and st.session_state.get("original_pdf_bytes") and main_analyze_button:
                            st.caption(
                                "Esegui 'Analizza Testo/Documento' per identificare i termini da redigere nel PDF.")

                    if st.session_state.get("redacted_pdf_output_bytes"):
                        original_fn_pdf_val = st.session_state.get("last_uploaded_filename", "documento.pdf")
                        base_fn_pdf_rd, _ = os.path.splitext(original_fn_pdf_val)
                        redacted_fn_pdf_rd = f"{base_fn_pdf_rd}_redatto.pdf"
                        st.download_button(
                            label="📥 Scarica PDF Redatto", data=st.session_state["redacted_pdf_output_bytes"],
                            file_name=redacted_fn_pdf_rd, mime="application/pdf",
                            key="dl_redacted_pdf_final_btn_v3"
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

                # --- BLOCCO MODIFICATO ---
                if csv_report_cols_data_disp_val is not None and not csv_report_cols_data_disp_val.empty:
                    st.markdown("#### Dettaglio Analisi per Colonna (Testuali LLM, Numeriche/Date Regole)")
                    df_user_csv_display_val = csv_report_cols_data_disp_val

                    # Column list con motivazione inclusa
                    columns_to_show = ["Colonna", "CategoriaLLM", "Esempi", "MetodoSuggerito", "Motivazione"]

                    # Filtra colonne per evitare KeyError se una non fosse presente nel report
                    columns_present = [col for col in columns_to_show if col in df_user_csv_display_val.columns]

                    df_tabella = df_user_csv_display_val[columns_present].rename(columns={
                        "CategoriaLLM": "Categoria",
                        "MetodoSuggerito": "Metodo Suggerito"
                    })

                    # Mostra la tabella
                    st.dataframe(df_tabella, use_container_width=True, height=300)
                # --- FINE BLOCCO MODIFICATO ---
                elif csv_report_cols_data_disp_val is not None and csv_report_cols_data_disp_val.empty and main_analyze_button:
                    st.info("L'analisi delle colonne del CSV non ha prodotto un report dettagliato.")

                if csv_overall_md_content_disp_val:
                    st.markdown("---");
                    st.markdown("### 📈 Report Privacy Complessivo (CSV)", unsafe_allow_html=True)
                    st.markdown(csv_overall_md_content_disp_val, unsafe_allow_html=True)
                    if st.button("Rimuovi Report Privacy Complessivo", key="clear_overall_csv_btn_v5"):
                        st.session_state["overall_csv_comment"] = None;
                        st.rerun()

                if isinstance(utility_metrics_display_val, dict) and utility_metrics_display_val:
                    st.markdown("---");
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
                if isinstance(csv_df_anon_content_disp_val, pd.DataFrame) and not csv_df_anon_content_disp_val.empty:
                    csv_bytes_dl_anon_val = csv_df_anon_content_disp_val.to_csv(index=False).encode("utf-8")
                    base_fn_val, ext_fn_val = os.path.splitext(
                        st.session_state.get('last_uploaded_filename', 'dataset.csv'))
                    dl_fn_anon_val = f"anonimizzato_generalizzato_{base_fn_val}{ext_fn_val if ext_fn_val else '.csv'}"
                    st.download_button(label="📥 Scarica CSV Generalizzato/Anonimizzato", data=csv_bytes_dl_anon_val,
                                       file_name=dl_fn_anon_val, mime="text/csv", use_container_width=True,
                                       key="dl_anon_csv_sugg_btn_v5")
                elif active_file_type_input == ".csv" and main_analyze_button:
                    st.info("Nessun CSV Generalizzato/Anonimizzato disponibile.")

                if not (
                        has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val) and active_file_type_input == ".csv" and main_analyze_button:
                    st.info("Nessun risultato specifico dall'analisi CSV da visualizzare.")
else:
    if not (active_raw_text_input or (active_df_input is not None and not active_df_input.empty)):
        st.caption(
            "Carica un file o incolla del testo, seleziona opzioni e clicca 'Analizza' per visualizzare i risultati.")
    elif not main_analyze_button:
        st.caption("Esegui un'analisi per visualizzare i risultati.")
