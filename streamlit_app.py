import streamlit as st
import pandas as pd
import json
import re
import hashlib
import asyncio
# MODIFICA: Assicurati che get_llm_overall_csv_comment sia importato dalla giusta posizione.
# Se è in generazione_testo.py, modifica l'import qui sotto.
# Se è in analyisis_cvs.py (come sembra dal tuo import), va bene così.
from modules.analyisis_cvs import analyze_and_anonymize_csv, get_llm_overall_csv_comment
from modules.config import LLM_MODELS
from modules.text_extractor import detect_extension, extract_text
from modules.generazione_testo import (
    generate_report,
    edit_document,
    extract_entities,
    sensitive_informations as get_sensitive_contexts,
    # Se get_llm_overall_csv_comment è definito qui, rimuovilo dall'import di analyisis_cvs
)
from modules.utils import write_file
# AGGIUNTA: Import per le funzioni di calcolo delle metriche di privacy
from modules.privacy_metrics import calculate_k_anonymity, calculate_l_diversity

# Tipi di entità da considerare PII
PII_TYPES = [
    "PERSON", "PER", "PERS", "person", "persona", "nome",
    "DATE", "data",
    "LOCATION", "LOC", "location", "luogo", "indirizzo", "address", "full address", "comune",
    "ORGANIZATION", "ORG", "organization", "società", "company",
    "codice fiscale", "national id", "ID_NUMBER",  # Aggiunto ID_NUMBER se l'LLM lo usa
    "phone number", "numero di telefono",
    "email address", "email",
    "credit card number", "numero carta di credito",
    "CUI",  # Assicurati che l'LLM lo riconosca o aggiungi alias se necessario
    "importo",
    "postal code",
    "MEDICATION", "DIAGNOSIS"  # Aggiunti tipi medici se rilevanti
]
PII_TYPES_LOWER = [pii.lower() for pii in PII_TYPES]


def genera_dati_report_generale(report_llm: dict, df_entita_ner: pd.DataFrame, pii_types_list_lower: list) -> dict:
    dati_consolidati = {
        "entita_totali_combinate": [],
        "statistiche_riassuntive": {},
        "riassunti_llm": [],
        "tabella_pii_consolidate": pd.DataFrame(),
        "df_entita_totali_combinate_grezze": pd.DataFrame()
    }

    if report_llm:
        for nome_modello_llm, contenuto_report in report_llm.items():
            if isinstance(contenuto_report, dict) and contenuto_report.get("found") and isinstance(
                    contenuto_report.get("entities"), list):
                for entita in contenuto_report["entities"]:
                    dati_consolidati["entita_totali_combinate"].append({
                        "Testo Entità": entita.get("text", ""),
                        "Tipo": entita.get("type", ""),
                        "Contesto/Dettagli": entita.get("context", "N/A"),
                        "Motivazione (LLM)": entita.get("reasoning", "N/A"),
                        "Fonte Modello": f"LLM: {nome_modello_llm}",
                        "Score": "N/A",
                        "source_chunk_info": entita.get("source_chunk_info", "N/A")
                    })
            if isinstance(contenuto_report, dict) and contenuto_report.get("summary"):
                dati_consolidati["riassunti_llm"].append({
                    "Modello": nome_modello_llm,
                    "Riassunto": contenuto_report["summary"]
                })

    if not df_entita_ner.empty:
        for _, riga in df_entita_ner.iterrows():
            dati_consolidati["entita_totali_combinate"].append({
                "Testo Entità": riga.get("text", ""),
                "Tipo": riga.get("type", ""),
                "Contesto/Dettagli": "N/A (NER Pipeline)",
                "Motivazione (LLM)": "N/A (NER Pipeline)",
                "Fonte Modello": f"NER: {riga.get('model_name', 'N/A')}",
                "Score": f"{riga.get('score', 0.0):.2f}" if isinstance(riga.get('score'), float) else riga.get('score',
                                                                                                               "N/A"),
                "source_chunk_info": "N/A"
            })

    if dati_consolidati["entita_totali_combinate"]:
        df_finale_entita = pd.DataFrame(dati_consolidati["entita_totali_combinate"])
        dati_consolidati["df_entita_totali_combinate_grezze"] = df_finale_entita
        df_entita_uniche_testo_tipo = df_finale_entita.drop_duplicates(subset=['Testo Entità', 'Tipo'], keep='first')
        dati_consolidati["statistiche_riassuntive"]["entita_rilevate_totali_grezze"] = len(df_finale_entita)
        dati_consolidati["statistiche_riassuntive"]["entita_uniche_testo_tipo"] = len(df_entita_uniche_testo_tipo)
        df_pii_consolidate_uniche = df_entita_uniche_testo_tipo[
            df_entita_uniche_testo_tipo["Tipo"].str.lower().isin(pii_types_list_lower)]
        dati_consolidati["statistiche_riassuntive"]["pii_identificate_uniche"] = len(df_pii_consolidate_uniche)
        dati_consolidati["statistiche_riassuntive"]["conteggio_per_tipo_pii"] = df_pii_consolidate_uniche[
            "Tipo"].value_counts().to_dict()

        if not df_pii_consolidate_uniche.empty:
            df_pii_originali_per_tabella = df_finale_entita[
                df_finale_entita.apply(lambda x: (x['Testo Entità'], x['Tipo']) in \
                                                 set(zip(df_pii_consolidate_uniche['Testo Entità'],
                                                         df_pii_consolidate_uniche['Tipo'])), axis=1) & \
                df_finale_entita["Tipo"].str.lower().isin(pii_types_list_lower)
                ]

            def aggrega_fonti_pii(group):
                fonti = sorted(list(set(group["Fonte Modello"])))
                contesti_llm = group[group["Fonte Modello"].str.startswith("LLM:")]["Contesto/Dettagli"].unique()
                contesto_display = " | ".join(c for c in contesti_llm if c != "N/A") if len(contesti_llm) > 0 else "N/A"
                motivazioni_llm = group[group["Fonte Modello"].str.startswith("LLM:")]["Motivazione (LLM)"].unique()
                motivazione_display = " | ".join(m for m in motivazioni_llm if m != "N/A") if len(
                    motivazioni_llm) > 0 else "N/A"
                scores_ner_validi = pd.to_numeric(group[group["Fonte Modello"].str.startswith("NER:")]["Score"],
                                                  errors='coerce').dropna()
                score_display = f"{scores_ner_validi.max():.2f}" if not scores_ner_validi.empty else "N/A"
                return pd.Series({
                    "Fonti Rilevamento": ", ".join(fonti),
                    "Contesto (da LLM)": contesto_display,
                    "Motivazione (da LLM)": motivazione_display,
                    "Score Max (da NER)": score_display
                })

            if not df_pii_originali_per_tabella.empty:
                df_visualizzazione_pii = df_pii_originali_per_tabella.groupby(["Testo Entità", "Tipo"],
                                                                              as_index=False).apply(
                    aggrega_fonti_pii).reset_index(drop=True)
                dati_consolidati["tabella_pii_consolidate"] = df_visualizzazione_pii
    else:
        for key_stat in ["entita_rilevate_totali_grezze", "entita_uniche_testo_tipo", "pii_identificate_uniche"]:
            dati_consolidati["statistiche_riassuntive"][key_stat] = 0
        dati_consolidati["statistiche_riassuntive"]["conteggio_per_tipo_pii"] = {}
    return dati_consolidati


def genera_documento_modificato_consolidato(original_text: str, df_pii_consolidate: pd.DataFrame) -> str:
    # ... (codice invariato)
    if df_pii_consolidate.empty or "Testo Entità" not in df_pii_consolidate.columns:
        return original_text
    modified_text = original_text
    replacements = []
    for _, row in df_pii_consolidate.iterrows():
        entity_text = str(row["Testo Entità"])
        entity_type = str(row.get("Tipo", "PII")).upper().replace(" ", "_")
        placeholder = f"[REDATTO_{entity_type}]"
        if entity_text and not entity_text.isspace():
            replacements.append((entity_text, placeholder))
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for text_to_replace, placeholder in replacements:
        modified_text = re.sub(re.escape(text_to_replace), placeholder, modified_text)
    return modified_text


def genera_csv_modificato_consolidato(original_df: pd.DataFrame, df_pii_consolidate: pd.DataFrame) -> pd.DataFrame:
    # ... (codice invariato)
    if original_df.empty or df_pii_consolidate.empty or "Testo Entità" not in df_pii_consolidate.columns:
        return original_df
    redacted_df = original_df.copy()
    replacements = []
    for _, row in df_pii_consolidate.iterrows():
        entity_text = str(row["Testo Entità"])
        entity_type = str(row.get("Tipo", "PII")).upper().replace(" ", "_")
        placeholder = f"[REDATTO_{entity_type}]"
        if entity_text and not entity_text.isspace():
            replacements.append((entity_text, placeholder))
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for col in redacted_df.columns:
        if pd.api.types.is_numeric_dtype(redacted_df[col]) or pd.api.types.is_datetime64_any_dtype(redacted_df[col]):
            redacted_column_as_str = redacted_df[col].astype(str)
            for text_to_replace, placeholder in replacements:
                redacted_column_as_str = redacted_column_as_str.str.replace(re.escape(text_to_replace), placeholder,
                                                                            regex=True)
            redacted_df[col] = redacted_column_as_str
        else:
            for idx in redacted_df.index:
                cell_value = redacted_df.at[idx, col]
                if pd.isna(cell_value):
                    continue
                cell_text = str(cell_value)
                modified_cell_text = cell_text
                for text_to_replace, placeholder in replacements:
                    modified_cell_text = re.sub(re.escape(text_to_replace), placeholder, modified_cell_text)
                if modified_cell_text != cell_text:
                    redacted_df.at[idx, col] = modified_cell_text
    return redacted_df


def main():
    st.set_page_config(page_title="DataSentinelAI", layout="wide")

    # Inizializzazione session_state (invariata)
    if "reports" not in st.session_state: st.session_state["reports"] = {}
    if "edited_docs" not in st.session_state: st.session_state["edited_docs"] = {}
    if "ner_entities" not in st.session_state: st.session_state["ner_entities"] = pd.DataFrame()
    if "general_report_data" not in st.session_state: st.session_state["general_report_data"] = None
    if "raw_text_input" not in st.session_state: st.session_state["raw_text_input"] = ""
    if "current_file_ext" not in st.session_state: st.session_state["current_file_ext"] = ".txt"
    if "last_uploaded_filename" not in st.session_state: st.session_state["last_uploaded_filename"] = None
    if "general_edited_document" not in st.session_state: st.session_state["general_edited_document"] = None
    if "original_csv_df" not in st.session_state: st.session_state["original_csv_df"] = None
    if "column_reports" not in st.session_state: st.session_state[
        "column_reports"] = {}  # Vecchia chiave, potrebbe non servire più se csv_analysis_report_df la sostituisce
    if "csv_analysis_report_df" not in st.session_state: st.session_state["csv_analysis_report_df"] = pd.DataFrame()
    if "calculated_risk_metrics" not in st.session_state: st.session_state.calculated_risk_metrics = {
        "k_anonymity_min": "N/D", "records_singoli": "N/D", "l_diversity": {}}
    if "identified_qids_for_summary" not in st.session_state: st.session_state.identified_qids_for_summary = []
    if "identified_sas_for_summary" not in st.session_state: st.session_state.identified_sas_for_summary = []
    if "overall_csv_comment" not in st.session_state: st.session_state.overall_csv_comment = None

    st.title("🛡️ DataSentinelAI")

    # Sezione 1: Input Utente (invariata)
    with st.container():
        # ... (codice invariato) ...
        st.subheader("1. Fornisci il Testo o File")
        default_radio_index = 0
        if st.session_state.get("last_uploaded_filename"):
            default_radio_index = 1
        elif st.session_state.get("raw_text_input"):
            default_radio_index = 0

        input_type = st.radio("Scegli tipo di input:", ["Testo libero", "Carica File"],
                              key="input_type_radio_main",
                              horizontal=True,
                              index=default_radio_index)
        raw_text_changed_flag = False
        if input_type == "Testo libero":
            if st.session_state.get("last_uploaded_filename") is not None:
                st.session_state["raw_text_input"] = ""
                st.session_state["last_uploaded_filename"] = None
                st.session_state["original_csv_df"] = None
                raw_text_changed_flag = True
            user_text_area = st.text_area("Inserisci il testo qui:", value=st.session_state["raw_text_input"],
                                          height=150, key="text_area_input_main", label_visibility="collapsed")
            if user_text_area != st.session_state["raw_text_input"]:
                st.session_state["raw_text_input"] = user_text_area
                st.session_state["current_file_ext"] = ".txt"  # Assumiamo .txt per testo libero
                raw_text_changed_flag = True
            if st.session_state.get("original_csv_df") is not None and not st.session_state.get(
                    "last_uploaded_filename"):  # Se prima c'era un CSV e ora testo libero
                st.session_state["original_csv_df"] = None
        else:  # Carica File
            uploaded_file = st.file_uploader("Carica un file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"],
                                             key="file_uploader_main", label_visibility="collapsed")
            if uploaded_file:
                if st.session_state.get("last_uploaded_filename") != uploaded_file.name or not st.session_state.get(
                        "raw_text_input"):  # Nuovo file o cambio da testo a file
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.session_state["current_file_ext"] = detect_extension(uploaded_file)
                    try:
                        st.session_state["raw_text_input"] = extract_text(
                            uploaded_file)  # Estrai testo per analisi generiche
                        raw_text_changed_flag = True
                        if st.session_state["current_file_ext"] == ".csv":
                            uploaded_file.seek(0)  # Riavvolgi il file per pandas
                            try:
                                df_preview = pd.read_csv(uploaded_file)
                                st.session_state["original_csv_df"] = df_preview.copy()
                                st.caption("Anteprima CSV (prime 5 righe):")
                                st.dataframe(df_preview.head(), height=150, use_container_width=True)
                            except Exception as e:
                                st.error(f"Errore lettura/anteprima CSV: {e}")
                                st.session_state["original_csv_df"] = None
                                st.session_state["raw_text_input"] = ""  # Pulisci testo se CSV fallisce
                        else:  # Non CSV
                            st.session_state["original_csv_df"] = None
                    except Exception as e:
                        st.error(f"Errore estrazione testo da file: {e}")
                        st.session_state["raw_text_input"] = ""
                        st.session_state["last_uploaded_filename"] = None
                        st.session_state["original_csv_df"] = None
            elif st.session_state.get(
                    "last_uploaded_filename") is not None and input_type == "Carica File":  # File deselezionato
                st.session_state["raw_text_input"] = ""
                st.session_state["last_uploaded_filename"] = None
                st.session_state["original_csv_df"] = None
                raw_text_changed_flag = True

        if raw_text_changed_flag:  # Se l'input è cambiato, resetta i risultati
            st.session_state["reports"] = {}
            st.session_state["edited_docs"] = {}
            st.session_state["ner_entities"] = pd.DataFrame()
            st.session_state["general_report_data"] = None
            st.session_state["general_edited_document"] = None
            st.session_state["csv_analysis_report_df"] = pd.DataFrame()  # Resetta anche questo
            st.session_state.calculated_risk_metrics = {"k_anonymity_min": "N/D", "records_singoli": "N/D",
                                                        "l_diversity": {}}
            st.session_state.identified_qids_for_summary = []
            st.session_state.identified_sas_for_summary = []
            st.session_state.overall_csv_comment = None
            st.rerun()

    raw_text_to_process = st.session_state.get("raw_text_input", "")
    st.markdown("---")

    # Sezione 2: Azioni di Analisi (invariata fino alla Sezione 2.b)
    with st.container():
        # ... (codice invariato per Sezione 2, Azioni LLM testo completo, NER, Modifica Doc, Report Generale Consolidato) ...
        st.subheader("2. Esegui Azioni di Analisi")
        action_cols_1_2 = st.columns(2)

        with action_cols_1_2[0]:  # Colonna sinistra per azioni principali
            # Bottone Analisi PII con LLM (Testo Completo o CSV come JSON)
            can_analyze_llm = (raw_text_to_process.strip() or \
                               (st.session_state.get("current_file_ext") == ".csv" and \
                                st.session_state.get("original_csv_df") is not None and \
                                not st.session_state.get("original_csv_df").empty))
            if st.button("🚀 Analizza PII con LLM (Testo Completo/CSV come JSON)", key="btn_analyze_pii_llm",
                         use_container_width=True, disabled=not can_analyze_llm):
                is_valid_input_for_llm = False
                if st.session_state.get("current_file_ext") == ".csv" and st.session_state.get(
                        "original_csv_df") is not None:
                    if not st.session_state.get("original_csv_df").empty:
                        is_valid_input_for_llm = True
                elif raw_text_to_process.strip():
                    is_valid_input_for_llm = True

                if is_valid_input_for_llm:
                    active_llm_models = LLM_MODELS
                    if not active_llm_models:
                        st.error("Nessun modello LLM configurato in config.py")
                    else:
                        reports_data_llm_output = {}
                        if st.session_state.get("current_file_ext") == ".csv" and st.session_state.get(
                                "original_csv_df") is not None:
                            with st.spinner("Generazione report PII (CSV come JSON) in corso..."):
                                df_csv_to_analyze = st.session_state["original_csv_df"]
                                chunk_size_csv_json = st.number_input(
                                    "Righe per chunk JSON (Analisi CSV)",
                                    min_value=10, value=100, step=10,
                                    key=f"chunk_size_csv_json_{st.session_state.get('last_uploaded_filename', 'csv_json_mode')}",
                                    help="Numero di righe CSV da inviare come JSON per ogni chunk all'LLM."
                                )
                                num_chunks_csv_json = (len(df_csv_to_analyze) - 1) // chunk_size_csv_json + 1
                                overall_progress_bar = st.progress(0, text="Avvio analisi CSV in JSON...")
                                for i_model, (model_name, model_api_id) in enumerate(active_llm_models.items(),
                                                                                     start=1):
                                    # ... (logica chunking CSV come JSON invariata)
                                    overall_progress_bar.progress((i_model - 1) / len(active_llm_models),
                                                                  text=f"Modello {model_name} (CSV-JSON): Avvio...")
                                    all_entities_current_model_json = []
                                    all_summaries_current_model_json = []
                                    chunk_progress_placeholder = st.empty()
                                    has_errors_in_model_chunks_json = False
                                    first_error_summary_model_json = ""
                                    first_error_raw_output_model_json = ""
                                    for c_idx in range(num_chunks_csv_json):
                                        start_row = c_idx * chunk_size_csv_json
                                        end_row = min((c_idx + 1) * chunk_size_csv_json, len(df_csv_to_analyze))
                                        df_chunk = df_csv_to_analyze.iloc[start_row:end_row]
                                        if df_chunk.empty: continue
                                        chunk_progress_placeholder.text(
                                            f"Modello {model_name}: elaborazione chunk JSON {c_idx + 1}/{num_chunks_csv_json} (righe {start_row + 1}-{end_row})...")
                                        chunk_json_records = df_chunk.to_dict(orient="records")
                                        payload_for_llm = {
                                            "instruction": "Identifica tutte le entità PII (come nome, email, indirizzo, telefono, codice fiscale, data di nascita, ID, etc.) " +
                                                           "in questo array JSON di record CSV. Per ogni entità identificata, restituisci un oggetto con i campi 'text' (il valore dell'entità), " +
                                                           "'type' (il tipo di PII, es. 'PERSON', 'EMAIL', 'ADDRESS'), 'context' (una breve descrizione del record o della colonna dove l'entità è stata trovata), " +
                                                           "e 'reasoning' (una breve spiegazione del perché è considerata PII). Assicurati che l'output sia un singolo oggetto JSON con chiavi 'found', 'entities', 'summary'.",
                                            "data": chunk_json_records
                                        }
                                        prompt_string_for_llm = json.dumps(payload_for_llm, ensure_ascii=False)
                                        report_from_chunk = generate_report(prompt_string_for_llm, model_api_id)
                                        if report_from_chunk:
                                            if report_from_chunk.get("entities") and isinstance(
                                                    report_from_chunk.get("entities"), list):
                                                for entity_obj in report_from_chunk["entities"]:
                                                    if isinstance(entity_obj, dict):
                                                        entity_obj[
                                                            "source_chunk_info"] = f"CSV-JSON righe {start_row + 1}-{end_row}"
                                                        all_entities_current_model_json.append(entity_obj)
                                            chunk_summary_content = report_from_chunk.get("summary", "")
                                            if "Error:" in chunk_summary_content and not has_errors_in_model_chunks_json:
                                                has_errors_in_model_chunks_json = True
                                                first_error_summary_model_json = f"Errore nel chunk JSON {c_idx + 1}: {chunk_summary_content}"
                                                if "raw_output" in report_from_chunk: first_error_raw_output_model_json = report_from_chunk.get(
                                                    "raw_output", "")
                                            elif not has_errors_in_model_chunks_json:
                                                all_summaries_current_model_json.append(
                                                    f"Sommario Chunk JSON {c_idx + 1}: {chunk_summary_content}")
                                        else:
                                            if not has_errors_in_model_chunks_json:
                                                has_errors_in_model_chunks_json = True
                                                first_error_summary_model_json = f"Errore: Nessun report restituito per chunk JSON {c_idx + 1}."
                                    chunk_progress_placeholder.empty()
                                    final_report_for_model_json = {"found": bool(all_entities_current_model_json),
                                                                   "entities": all_entities_current_model_json}
                                    if has_errors_in_model_chunks_json:
                                        final_report_for_model_json["summary"] = first_error_summary_model_json
                                        final_report_for_model_json["raw_output"] = first_error_raw_output_model_json
                                    elif all_summaries_current_model_json:
                                        final_report_for_model_json["summary"] = "\n---\n".join(
                                            all_summaries_current_model_json)
                                    else:
                                        final_report_for_model_json[
                                            "summary"] = "Analisi CSV-JSON completata. Nessun sommario specifico dai chunk."
                                    reports_data_llm_output[model_name] = final_report_for_model_json
                                    overall_progress_bar.progress(i_model / len(active_llm_models),
                                                                  text=f"Modello {model_name} (CSV-JSON) completato.")
                                overall_progress_bar.progress(1.0, text="Analisi PII (CSV come JSON) completata!")
                                st.success("Analisi PII (CSV come JSON) completata!")
                                st.toast("Analisi PII (CSV come JSON) completata!", icon="✅")
                        else:  # Analisi Testo Libero
                            # ... (logica chunking testo invariata) ...
                            with st.spinner("Generazione report PII (Testo) in corso..."):
                                CHUNK_SIZE_FOR_GENERIC_TEXT = 10000
                                num_text_chunks = (len(raw_text_to_process) - 1) // CHUNK_SIZE_FOR_GENERIC_TEXT + 1
                                overall_progress_bar = st.progress(0, text="Avvio analisi Testo...")
                                for i_model, (model_name, model_api_id) in enumerate(active_llm_models.items(),
                                                                                     start=1):
                                    overall_progress_bar.progress((i_model - 1) / len(active_llm_models),
                                                                  text=f"Modello {model_name} (Testo): Avvio...")
                                    all_entities_current_model_text = []
                                    all_summaries_current_model_text = []
                                    text_chunk_progress_placeholder = st.empty()
                                    has_errors_in_model_chunks_text = False
                                    first_error_summary_model_text = ""
                                    first_error_raw_output_model_text = ""
                                    for k_idx in range(num_text_chunks):
                                        start_char_idx = k_idx * CHUNK_SIZE_FOR_GENERIC_TEXT
                                        end_char_idx = min((k_idx + 1) * CHUNK_SIZE_FOR_GENERIC_TEXT,
                                                           len(raw_text_to_process))
                                        current_text_segment_to_analyze = raw_text_to_process[
                                                                          start_char_idx:end_char_idx]
                                        if not current_text_segment_to_analyze.strip(): continue
                                        text_chunk_progress_placeholder.text(
                                            f"Modello {model_name}: elaborazione segmento testuale {k_idx + 1}/{num_text_chunks}...")
                                        segment_report_data = generate_report(current_text_segment_to_analyze,
                                                                              model_api_id)
                                        if segment_report_data:
                                            if segment_report_data.get("entities") and isinstance(
                                                    segment_report_data.get("entities"), list):
                                                for entity_obj_text in segment_report_data["entities"]:
                                                    if isinstance(entity_obj_text, dict):
                                                        entity_obj_text[
                                                            "source_chunk_info"] = f"Testo caratteri {start_char_idx}-{end_char_idx}"
                                                        all_entities_current_model_text.append(entity_obj_text)
                                            segment_summary_content = segment_report_data.get("summary", "")
                                            if "Error:" in segment_summary_content and not has_errors_in_model_chunks_text:
                                                has_errors_in_model_chunks_text = True
                                                first_error_summary_model_text = f"Errore nel segmento testuale {k_idx + 1}: {segment_summary_content}"
                                                if "raw_output" in segment_report_data: first_error_raw_output_model_text = segment_report_data.get(
                                                    "raw_output", "")
                                            elif not has_errors_in_model_chunks_text:
                                                all_summaries_current_model_text.append(
                                                    f"Sommario Segmento Testuale {k_idx + 1}: {segment_summary_content}")
                                        else:
                                            if not has_errors_in_model_chunks_text:
                                                has_errors_in_model_chunks_text = True
                                                first_error_summary_model_text = f"Errore: Nessun report per segmento testuale {k_idx + 1}."
                                    text_chunk_progress_placeholder.empty()
                                    final_report_for_model_text = {"found": bool(all_entities_current_model_text),
                                                                   "entities": all_entities_current_model_text}
                                    if has_errors_in_model_chunks_text:
                                        final_report_for_model_text["summary"] = first_error_summary_model_text
                                        final_report_for_model_text["raw_output"] = first_error_raw_output_model_text
                                    elif all_summaries_current_model_text:
                                        final_report_for_model_text["summary"] = "\n---\n".join(
                                            all_summaries_current_model_text)
                                    else:
                                        final_report_for_model_text[
                                            "summary"] = "Analisi testuale completata. Nessun sommario specifico dai segmenti."
                                    reports_data_llm_output[model_name] = final_report_for_model_text
                                    overall_progress_bar.progress(i_model / len(active_llm_models),
                                                                  text=f"Modello {model_name} (Testo) completato.")
                                overall_progress_bar.progress(1.0, text="Analisi PII (Testo) completata!")
                                st.success("Analisi PII (Testo) completata!")
                                st.toast("Analisi PII (Testo) completata!", icon="✅")
                        st.session_state["reports"] = reports_data_llm_output
                        st.info("ℹ️ Report PII (LLM) generati. Visualizzali nella tab apposita o procedi.")
                else:
                    st.error("Il testo di input è vuoto o il file CSV caricato è vuoto/invalido.")

            if st.button("✨ Esegui NER Dedicata", key="btn_run_ner", use_container_width=True,
                         disabled=not raw_text_to_process.strip()):
                # ... (codice invariato)
                if raw_text_to_process.strip():
                    with st.spinner("Estrazione entità NER dedicata in corso..."):
                        ner_entities_list_loc = extract_entities(raw_text_to_process)
                        st.session_state["ner_entities"] = pd.DataFrame(
                            ner_entities_list_loc) if ner_entities_list_loc else pd.DataFrame()
                        st.success("Analisi NER dedicata completata.")
                        st.toast("Analisi NER completata!", icon="🔖")
                        st.info("ℹ️ Entità NER estratte. Visualizzale nella tab apposita.")

        with action_cols_1_2[1]:
            # ... (codice invariato per Modifica Documento e Report Generale Consolidato) ...
            can_edit_doc = (raw_text_to_process.strip() and \
                            st.session_state.get("reports") and \
                            any(st.session_state["reports"].values()))
            if st.button("✏️ Modifica Documento (basato su Report Testo Completo/CSV-JSON)", key="btn_edit_doc",
                         use_container_width=True, disabled=not can_edit_doc):
                if raw_text_to_process.strip() and st.session_state.get("reports") and any(
                        st.session_state["reports"].values()):
                    if st.session_state.get("current_file_ext") == ".csv":
                        st.warning(
                            "La modifica del documento per i CSV basata su report JSON non è ottimizzata per modificare direttamente il CSV strutturato. Verrà modificata la rappresentazione testuale del CSV.")
                    with st.spinner("Modifica dei documenti in corso..."):
                        edited_docs_data_local = {}
                        active_llm_models_edit = {k: v for k, v in LLM_MODELS.items() if
                                                  k in st.session_state["reports"]}
                        for model_name_edit, report_item_edit in st.session_state["reports"].items():
                            if model_name_edit not in active_llm_models_edit: continue
                            model_api_id_edit = active_llm_models_edit[model_name_edit]
                            report_json_str_for_edit_loc = json.dumps(report_item_edit) if isinstance(report_item_edit,
                                                                                                      dict) else str(
                                report_item_edit)
                            edited_docs_data_local[model_name_edit] = edit_document(raw_text_to_process,
                                                                                    report_json_str_for_edit_loc,
                                                                                    model_api_id_edit)
                        st.session_state["edited_docs"] = edited_docs_data_local
                        st.success("Modifica documenti completata.")
                        st.toast("Documenti modificati dagli LLM!", icon="✏️")
                        st.info("ℹ️ Documenti modificati. Visualizzali e scaricali nella tab apposita.")
                elif not raw_text_to_process.strip():
                    st.error("Il testo di input è vuoto.")
                else:
                    st.warning("Genera prima i 'Report PII con LLM' per poter modificare il documento.")

            can_generate_general_report = (st.session_state.get("reports") or \
                                           not st.session_state.get("ner_entities", pd.DataFrame()).empty)
            if st.button("📊 Genera/Aggiorna Report Generale", key="btn_general_report_main", use_container_width=True,
                         disabled=not can_generate_general_report):
                if st.session_state.get("reports") or not st.session_state.get("ner_entities", pd.DataFrame()).empty:
                    with st.spinner("Creazione del Report Generale in corso..."):
                        st.session_state["general_report_data"] = genera_dati_report_generale(
                            st.session_state["reports"],
                            st.session_state["ner_entities"],
                            PII_TYPES_LOWER
                        )
                    st.success("Report Generale pronto/aggiornato!")
                    st.toast("Report Generale creato/aggiornato.", icon="📄")
                    st.info("ℹ️ Report Generale disponibile nella tab apposita.")
                else:
                    st.warning("Esegui prima un'analisi (Report PII o NER) per avere dati da aggregare.")

        st.markdown("---")
        can_generate_consolidated_edited = st.session_state.get("general_report_data") and \
                                           not st.session_state.get("general_report_data", {}).get(
                                               "tabella_pii_consolidate", pd.DataFrame()).empty
        button_label_consolidated = "✍️ Genera Documento Modificato Consolidato"
        is_csv_mode_for_consolidated = st.session_state.get("current_file_ext") == ".csv" and \
                                       st.session_state.get("original_csv_df") is not None and \
                                       st.session_state.get("general_report_data")
        if is_csv_mode_for_consolidated:
            button_label_consolidated = "✍️ Genera CSV Modificato Consolidato (basato su Report Generale)"
        if st.button(button_label_consolidated, key="btn_generate_general_edited_doc_csv", use_container_width=True,
                     disabled=not can_generate_consolidated_edited):
            general_report_data_consolidated = st.session_state.get("general_report_data")
            if general_report_data_consolidated and not general_report_data_consolidated.get("tabella_pii_consolidate",
                                                                                             pd.DataFrame()).empty:
                with st.spinner(
                        f"Generazione del {('CSV' if is_csv_mode_for_consolidated else 'Documento')} Modificato Consolidato in corso..."):
                    df_pii_for_redaction_consolidated = general_report_data_consolidated["tabella_pii_consolidate"]
                    if is_csv_mode_for_consolidated:
                        original_df_for_redaction_consolidated = st.session_state["original_csv_df"]
                        redacted_df_output_consolidated = genera_csv_modificato_consolidato(
                            original_df_for_redaction_consolidated, df_pii_for_redaction_consolidated)
                        st.session_state["general_edited_document"] = redacted_df_output_consolidated
                        st.success("CSV Modificato Consolidato generato!")
                        st.toast("CSV Modificato Consolidato pronto.", icon="")  # Potrebbe essere un'icona diversa
                    else:
                        original_text_for_consolidation_redaction = st.session_state["raw_text_input"]
                        st.session_state["general_edited_document"] = genera_documento_modificato_consolidato(
                            original_text_for_consolidation_redaction, df_pii_for_redaction_consolidated)
                        st.success("Documento Modificato Consolidato generato!")
                        st.toast("Documento Modificato Consolidato pronto.", icon="📝")
                    st.info(
                        f"ℹ️ Il {('CSV' if is_csv_mode_for_consolidated else 'Documento')} Modificato Consolidato è disponibile nella tab 'Report Generale Consolidato'.")
            else:
                st.warning("Genera prima il 'Report Generale Consolidato' con PII identificate.")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────
    # Sezione 2.b: Analisi CSV, Metriche Re-ID e Report Finale
    # ─────────────────────────────────────────────────────────
    if st.session_state.get("current_file_ext") == ".csv" and st.session_state.get("original_csv_df") is not None:
        st.subheader("2.b Analisi CSV Dettagliata, Rischi di Re-identificazione e Report Finale")

        if not LLM_MODELS:
            st.error("❌ Nessun modello LLM configurato in `modules/config.py`.")
        else:
            model_options = list(LLM_MODELS.keys())
            if not model_options:
                st.error("❌ Dizionario LLM_MODELS configurato ma vuoto.")
            else:
                selected_model_name = st.selectbox(
                    "🤖 Scegli il modello LLM per l'analisi delle colonne e il report finale:",
                    options=model_options, index=0, key="csv_model_selector_reid",
                    help="Modello per analisi colonne, classificazione QID/SA, e generazione report finale."
                )
                model_api_id_for_csv = LLM_MODELS[selected_model_name]
                st.caption(f"Modello selezionato: `{selected_model_name}` (API ID: `{model_api_id_for_csv}`)")

                original_data_df = st.session_state["original_csv_df"]  # DataFrame originale
                text_cols = original_data_df.select_dtypes(include=["object", "string"]).columns

                if not text_cols.empty:
                    if st.button(f"📊 Esegui Analisi Completa del CSV con {selected_model_name}",
                                 key="btn_full_csv_analysis_v2"):
                        # Resetta stati per una nuova analisi completa
                        st.session_state["csv_analysis_report_df"] = pd.DataFrame()
                        st.session_state.calculated_risk_metrics = {"k_anonymity_min": "N/D", "records_singoli": "N/D",
                                                                    "l_diversity": {}}
                        st.session_state.identified_qids_for_summary = []
                        st.session_state.identified_sas_for_summary = []
                        st.session_state.overall_csv_comment = None

                        # 1. Analisi colonne (asincrona)
                        with st.spinner(
                                f"Passo 1/3: Analisi delle colonne del CSV con {selected_model_name}... (max 5 parallele)"):
                            df_text_cols_to_analyze = original_data_df[text_cols].copy()
                            try:
                                report_df_cols, _ = asyncio.run(
                                    analyze_and_anonymize_csv(
                                        df_text_cols_to_analyze,
                                        model_api_id=model_api_id_for_csv,
                                        sample_size_for_preview=5,
                                        default_method_fallback="mask",
                                        max_concurrent_requests=5
                                    )
                                )
                                st.session_state["csv_analysis_report_df"] = report_df_cols
                                st.success("Analisi colonne CSV completata.")
                            except Exception as e:
                                st.error(f"Errore durante l'analisi asincrona delle colonne: {e}")
                                st.stop()  # Interrompi se l'analisi colonne fallisce

                        # 2. Calcolo metriche di re-identificazione (se l'analisi colonne ha prodotto risultati)
                        if not st.session_state["csv_analysis_report_df"].empty:
                            with st.spinner("Passo 2/3: Calcolo delle metriche di rischio di re-identificazione..."):
                                report_df_llm = st.session_state["csv_analysis_report_df"]
                                qids = []
                                sas = []
                                if "CategoriaLLM" in report_df_llm.columns:
                                    qid_mask = report_df_llm["CategoriaLLM"].str.lower().str.contains(
                                        r"quasi[\s\-]?identificatore", na=False, regex=True)
                                    initial_qids = report_df_llm[qid_mask]["Colonna"].tolist()
                                    sas = report_df_llm[
                                        report_df_llm["CategoriaLLM"].str.lower().str.contains("attributo sensibile",
                                                                                               na=False, regex=False)][
                                        "Colonna"].tolist()

                                    temp_qids_filtered = []
                                    for q_col in initial_qids:
                                        if q_col in original_data_df.columns:
                                            cardinality_ratio = original_data_df[q_col].nunique(dropna=False) / len(
                                                original_data_df)
                                            if len(original_data_df) < 100 or cardinality_ratio < 0.90:
                                                temp_qids_filtered.append(q_col)
                                    qids = temp_qids_filtered
                                else:
                                    st.warning(
                                        "Colonna 'CategoriaLLM' non trovata nel report. Impossibile identificare QID/SA per metriche.")

                                st.session_state.identified_qids_for_summary = qids
                                st.session_state.identified_sas_for_summary = sas

                                df_metrics_input = original_data_df
                                if len(original_data_df) > 100000:
                                    df_metrics_input = original_data_df.sample(n=100000, random_state=42)

                                if qids:
                                    try:
                                        k_min, rec_sing = calculate_k_anonymity(df_metrics_input, qids)
                                        st.session_state.calculated_risk_metrics["k_anonymity_min"] = int(
                                            k_min) if k_min != float('inf') else "N/A (>N)"
                                        st.session_state.calculated_risk_metrics["records_singoli"] = int(rec_sing)
                                    except Exception:
                                        pass  # Gestisci errore se necessario

                                    if sas:
                                        for sa_col in sas:
                                            if sa_col in df_metrics_input.columns:
                                                try:
                                                    l_min = calculate_l_diversity(df_metrics_input, qids, sa_col)
                                                    st.session_state.calculated_risk_metrics["l_diversity"][sa_col] = {
                                                        "l_min": int(l_min) if l_min not in [float('inf'), 0] else (
                                                            "N/A" if l_min == 0 else l_min)}
                                                except Exception:
                                                    pass  # Gestisci errore
                                st.success("Metriche di rischio di re-identificazione calcolate.")

                            # 3. Generazione del Report Finale Completo
                            with st.spinner("Passo 3/3: Generazione del Report Privacy completo..."):
                                file_name = st.session_state.get("last_uploaded_filename", "File CSV")
                                try:
                                    st.session_state.overall_csv_comment = get_llm_overall_csv_comment(
                                        st.session_state["csv_analysis_report_df"],
                                        st.session_state.calculated_risk_metrics,
                                        st.session_state.identified_qids_for_summary,
                                        st.session_state.identified_sas_for_summary,
                                        model_api_id_for_csv,
                                        file_name=file_name
                                    )
                                    st.success("Report Privacy completo generato.")
                                except NameError:
                                    st.error("Funzione 'get_llm_overall_csv_comment' non trovata.")
                                    st.session_state.overall_csv_comment = "Errore: Funzione per il report non trovata."
                                except Exception as e_rep:
                                    st.error(f"Errore generazione report: {e_rep}")
                                    st.session_state.overall_csv_comment = f"Errore: {e_rep}"
                        else:  # Se csv_analysis_report_df è vuoto dopo il passo 1
                            st.info(
                                "L'analisi delle colonne non ha prodotto risultati, impossibile procedere con metriche e report finale.")

                    # --- Visualizzazione dei risultati dell'analisi per colonna ---
                    if not st.session_state.get("csv_analysis_report_df", pd.DataFrame()).empty:
                        st.markdown("---")
                        st.markdown("### Dettaglio Analisi per Colonna e Suggerimenti di Anonimizzazione")
                        df_report_to_display = st.session_state["csv_analysis_report_df"]
                        # Filtra per mostrare solo colonne che richiedono attenzione o hanno PII
                        mask_display_cols = (df_report_to_display["LLM_HaTrovatoEntitaPII"] == True) | \
                                            (df_report_to_display["MetodoSuggerito"] != "nessuno")

                        cols_to_show_user = df_report_to_display[mask_display_cols]

                        if cols_to_show_user.empty and not df_report_to_display.empty:
                            st.success(
                                "✅ Nessuna colonna sembra richiedere un intervento di anonimizzazione urgente o contenere PII dirette rilevanti secondo l'analisi LLM.")
                        elif not cols_to_show_user.empty:
                            method_selection_ui = {}
                            available_methods_ui = ["hash", "mask", "generalize_date", "truncate", "nessuno"]
                            for idx, row_data in cols_to_show_user.iterrows():
                                col_name_ui = row_data["Colonna"]
                                problem_desc_ui = row_data["Problematica"]
                                examples_ui = row_data["Esempi"]
                                suggested_method_ui = row_data["MetodoSuggerito"]
                                reasoning_ui = row_data["Motivazione"]

                                st.markdown(f"**Colonna: {col_name_ui}**")
                                st.caption(f"Esempi: {examples_ui}")
                                if "Errore" in problem_desc_ui:
                                    st.error(f"{problem_desc_ui}")
                                elif "non ha rilevato PII specifiche" in problem_desc_ui:
                                    st.info(f"{problem_desc_ui}")
                                else:
                                    st.warning(f"{problem_desc_ui}")
                                st.info(
                                    f"Metodo Anon. Suggerito LLM: **{suggested_method_ui}**\n\n> Motivazione Metodo: _{reasoning_ui}_")

                                default_idx_ui = available_methods_ui.index(
                                    suggested_method_ui) if suggested_method_ui in available_methods_ui else 0
                                method_selection_ui[col_name_ui] = st.selectbox(f"Scegli metodo per «{col_name_ui}»:",
                                                                                available_methods_ui,
                                                                                index=default_idx_ui,
                                                                                key=f"meth_sel_reid_{col_name_ui}")
                                st.markdown("---")

                            if st.button("🔒 Applica Metodi Selezionati e Scarica CSV Anonimizzato",
                                         key="btn_apply_anon_reid"):
                                # ... (Logica per applicare method_selection_ui e scaricare, come prima)
                                df_to_anonymize_final_reid = st.session_state["original_csv_df"].copy()
                                with st.spinner("Applico anonimizzazione selezionata..."):
                                    for col_to_anon, selected_method_for_col in method_selection_ui.items():
                                        if col_to_anon in df_to_anonymize_final_reid.columns:
                                            # ... (copia la logica di if/elif per hash, mask, generalize_date, truncate)
                                            if selected_method_for_col == "hash":
                                                df_to_anonymize_final_reid[col_to_anon] = df_to_anonymize_final_reid[
                                                    col_to_anon].astype(str).apply(
                                                    lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(
                                                        x) else x)
                                            elif selected_method_for_col == "mask":
                                                df_to_anonymize_final_reid[col_to_anon] = df_to_anonymize_final_reid[
                                                    col_to_anon].astype(str).str.replace(r"[a-zA-Z0-9]", "*",
                                                                                         regex=True)
                                            elif selected_method_for_col == "generalize_date":
                                                try:
                                                    parsed_dates_final = pd.to_datetime(
                                                        df_to_anonymize_final_reid[col_to_anon], errors='coerce')
                                                    df_to_anonymize_final_reid[
                                                        col_to_anon] = parsed_dates_final.dt.to_period("M").astype(
                                                        str).replace('NaT', pd.NA)
                                                except Exception:
                                                    st.warning(
                                                        f"Impossibile generalizzare le date per la colonna {col_to_anon}. Lasciata invariata.")
                                            elif selected_method_for_col == "truncate":
                                                df_to_anonymize_final_reid[col_to_anon] = df_to_anonymize_final_reid[
                                                                                              col_to_anon].astype(
                                                    str).str.slice(0, 10) + "..."
                                st.session_state["anonymized_csv_for_download"] = df_to_anonymize_final_reid.copy()
                                st.success("Anonimizzazione completata ✅")
                                st.dataframe(st.session_state["anonymized_csv_for_download"].head(),
                                             use_container_width=True)
                                csv_bytes_dl = st.session_state["anonymized_csv_for_download"].to_csv(
                                    index=False).encode("utf-8")
                                dl_file_name = f"anonimizzato_reid_{st.session_state.get('last_uploaded_filename', 'file').replace('.csv', '')}.csv"
                                st.download_button("📥 Scarica CSV Anonimizzato", csv_bytes_dl, dl_file_name, "text/csv",
                                                   key="dl_anon_csv_reid_btn")
                        # else: # Se cols_to_show_user è vuoto ma il report generale non lo è
                        #    st.caption("Nessuna colonna specifica richiede un intervento manuale di anonimizzazione basato sui suggerimenti LLM.")

                    # --- Visualizzazione del Report Privacy Completo ---
                    if st.session_state.overall_csv_comment:
                        st.markdown("---")  # Separatore prima del report finale
                        st.markdown(st.session_state.overall_csv_comment, unsafe_allow_html=True)
                        if st.button("Rimuovi Report Privacy Completo", key="clear_overall_report_final_v3"):
                            st.session_state.overall_csv_comment = None
                            st.session_state.calculated_risk_metrics = {"k_anonymity_min": "N/D",
                                                                        "records_singoli": "N/D", "l_diversity": {}}
                            st.session_state.identified_qids_for_summary = []
                            st.session_state.identified_sas_for_summary = []
                            st.rerun()
                    elif st.session_state.get("csv_analysis_report_df") is not None and not st.session_state.get(
                            "csv_analysis_report_df").empty:
                        st.caption(
                            "Clicca 'Esegui Analisi Completa del CSV' per generare l'analisi delle colonne, le metriche di rischio e il report finale.")

                else:  # text_cols è vuoto
                    st.info(
                        "ℹ️ Il file CSV caricato non contiene colonne di tipo testuale (object o string) per l'analisi LLM.")
            # Fine del blocco 'else' per 'if not model_options:'
        # Fine del blocco 'else' per 'if not LLM_MODELS:'
    # Fine del blocco principale 'if st.session_state.get("current_file_ext") == ".csv" ...'

    # Questo st.markdown("---") è quello che avevi alla riga 835, che chiude la Sezione 2.b
    # st.markdown("---") # Potrebbe essere ridondante se il container successivo ha già un separatore

    # Logica per visualizzare i risultati o messaggio di attesa (invariata)
    display_results_flag = False
    # ... (codice invariato) ...
    if st.session_state.get("current_file_ext") == ".csv" and st.session_state.get("original_csv_df") is not None:
        if not st.session_state.get("original_csv_df").empty:
            display_results_flag = True
    elif raw_text_to_process.strip():
        display_results_flag = True

    if not display_results_flag:
        st.info("Inserisci del testo o carica un file valido e scegli un'azione per visualizzare i risultati.")
        st.stop()

    # Sezione 3: Visualizza Risultati (invariata)
    with st.container():
        # ... (codice invariato) ...
        st.subheader("3. Visualizza Risultati")
        tab_titles = ["🔎 Report PII (LLM - Testo Completo/CSV-JSON)", "🔖 Entità NER Dedicata",
                      "✏️ Documenti Modificati (LLM)",
                      "📊 Report Generale Consolidato"]
        tab_llm_reports, tab_ner_dedicated, tab_edited_docs, tab_general_report_display = st.tabs(tab_titles)

        with tab_llm_reports:
            # ... (codice invariato)
            header_text_llm_report = "Report PII da LLM"
            if st.session_state.get("current_file_ext") == ".csv" and st.session_state.get(
                    "original_csv_df") is not None:
                header_text_llm_report += " (Analisi su CSV come JSON)"
            else:
                header_text_llm_report += " (Analisi su Testo Completo)"
            st.header(header_text_llm_report)
            if st.session_state.get("reports") and any(st.session_state["reports"].values()):
                report_llm_keys = list(st.session_state["reports"].keys())
                if report_llm_keys:
                    report_llm_tabs_display = st.tabs(report_llm_keys)
                    for r_tab_idx, model_name_display in enumerate(report_llm_keys):
                        with report_llm_tabs_display[r_tab_idx]:
                            report_content_display = st.session_state["reports"][model_name_display]
                            st.subheader(f"Report da: {model_name_display}")
                            if isinstance(report_content_display, dict):
                                if report_content_display.get("found") and isinstance(
                                        report_content_display.get("entities"), list) and report_content_display.get(
                                    "entities"):
                                    df_entities_llm_disp = pd.DataFrame(report_content_display["entities"])
                                    st.markdown("###### Entità Rilevate:")
                                    cols_to_show_llm = ["type", "text", "context", "reasoning", "source_chunk_info"]
                                    cols_present_llm = [col for col in cols_to_show_llm if
                                                        col in df_entities_llm_disp.columns]
                                    st.dataframe(df_entities_llm_disp[cols_present_llm], use_container_width=True,
                                                 height=250)
                                    df_pii_llm_disp = df_entities_llm_disp[
                                        df_entities_llm_disp["type"].str.lower().isin(PII_TYPES_LOWER)]
                                    if not df_pii_llm_disp.empty:
                                        with st.expander("🔒 PII Trovati (dettaglio)", expanded=False):
                                            for _, row_pii_llm in df_pii_llm_disp.iterrows():
                                                st.markdown(f"**{row_pii_llm['type']}**: {row_pii_llm['text']}")
                                                st.markdown(f"  - Contesto: _{row_pii_llm.get('context', 'N/A')}_")
                                                st.markdown(f"  - Motivazione: _{row_pii_llm.get('reasoning', 'N/A')}_")
                                                if "source_chunk_info" in row_pii_llm and row_pii_llm[
                                                    'source_chunk_info'] != "N/A":
                                                    st.markdown(
                                                        f"  - Info Chunk/Segmento: _{row_pii_llm['source_chunk_info']}_")
                                                st.markdown("---")
                                    else:
                                        st.caption(
                                            "Nessuna PII specifica trovata da questo LLM (secondo la lista PII_TYPES).")
                                else:
                                    st.caption("Nessuna entità trovata o report LLM non valido/vuoto.")
                                st.markdown(f"**Riassunto (LLM)**: {report_content_display.get('summary', 'N/A')}")
                                if "raw_output" in report_content_display and report_content_display.get(
                                        "raw_output") and "Error:" in report_content_display.get("summary", ""):
                                    with st.expander("Mostra output grezzo dell'errore LLM"):
                                        st.text(report_content_display["raw_output"])
                                report_json_str_disp = json.dumps(report_content_display, indent=2, ensure_ascii=False)
                                report_bytes_disp = write_file(report_json_str_disp, ".json")
                                st.download_button(label=f"Scarica Report JSON ({model_name_display})",
                                                   data=report_bytes_disp,
                                                   file_name=f"report_pii_{model_name_display.replace(' ', '_')}.json",
                                                   mime="application/json",
                                                   key=f"download_json_tab_{model_name_display}",
                                                   use_container_width=True)
                            else:
                                st.text(str(report_content_display))
                            if isinstance(report_content_display, dict):
                                st.markdown("---")
                                st.markdown("###### Contesti di Sensibilità (da questo report LLM)")
                                model_api_id_ctx_disp = LLM_MODELS.get(model_name_display)
                                if model_api_id_ctx_disp:
                                    report_json_str_for_ctx_disp = json.dumps(report_content_display)
                                    with st.spinner(f"Recupero contesti e motivazioni da {model_name_display}..."):
                                        contexts_output_disp = get_sensitive_contexts(report_json_str_for_ctx_disp,
                                                                                      model_api_id_ctx_disp)
                                        st.markdown(
                                            contexts_output_disp if contexts_output_disp else "Nessun contesto di sensibilità specifico generato.")
                                else:
                                    st.warning(
                                        f"ID Modello API non trovato per {model_name_display} per generare contesti.")
                else:
                    st.info("Nessun report LLM disponibile. Esegui 'Analizza PII con LLM'.")
            else:
                st.info("Esegui 'Analizza PII con LLM' per visualizzare i risultati qui.")

        with tab_ner_dedicated:
            # ... (codice invariato)
            st.header("Entità Rilevate da NER Dedicata (su Testo Completo)")
            if not st.session_state.get("ner_entities", pd.DataFrame()).empty:
                st.dataframe(st.session_state["ner_entities"], use_container_width=True)
                df_ner_pii_disp = st.session_state["ner_entities"][
                    st.session_state["ner_entities"]["type"].str.lower().isin(PII_TYPES_LOWER)]
                if not df_ner_pii_disp.empty:
                    with st.expander("🔒 PII Trovati (NER Dedicata - dettaglio)", expanded=False):
                        for _, row_ner_disp in df_ner_pii_disp.iterrows():
                            st.markdown(
                                f"- **{row_ner_disp['type']}** (da *{row_ner_disp['model_name']}*): {row_ner_disp['text']} (Score: {row_ner_disp.get('score', 0.0):.2f})")
                else:
                    st.caption("Nessuna PII specifica trovata dall'analisi NER dedicata (secondo la lista PII_TYPES).")
                ner_csv_data_disp = st.session_state["ner_entities"].to_csv(index=False).encode('utf-8')
                st.download_button(label="Scarica Entità NER (CSV)", data=ner_csv_data_disp, file_name="entita_ner.csv",
                                   mime="text/csv", key="download_ner_csv_tab", use_container_width=True)
            else:
                st.info("Nessuna entità da NER dedicata disponibile. Esegui 'Esegui NER Dedicata'.")

        with tab_edited_docs:
            # ... (codice invariato)
            st.header("Documenti Modificati (output testuale dagli LLM)")
            if st.session_state.get("edited_docs") and any(st.session_state["edited_docs"].values()):
                for model_name_disp_edit, edited_text_content_disp in st.session_state["edited_docs"].items():
                    with st.expander(f"Documento Modificato da: {model_name_disp_edit}", expanded=True):
                        st.text_area(f"Testo Modificato da {model_name_disp_edit}", edited_text_content_disp,
                                     height=300, key=f"mod_text_disp_tab_{model_name_disp_edit}", disabled=True)
                        edited_doc_bytes = write_file(edited_text_content_disp, ".txt")
                        download_filename_edited = f"doc_modificato_{model_name_disp_edit.replace(' ', '_')}.txt"
                        st.download_button(label=f"Scarica Doc Modificato (.txt) ({model_name_disp_edit})",
                                           data=edited_doc_bytes, file_name=download_filename_edited, mime="text/plain",
                                           key=f"download_edited_txt_tab_{model_name_disp_edit}",
                                           use_container_width=True)
            else:
                st.info(
                    "Nessun documento modificato disponibile. Genera prima i 'Report PII con LLM' e poi esegui "
                    "'Modifica Documento'.")

        with tab_general_report_display:
            # ... (codice invariato)
            st.header("Report Generale Consolidato")
            if st.session_state.get("general_report_data"):
                report_data_gen = st.session_state["general_report_data"]
                st.subheader("Statistiche Riassuntive Globali")
                stats_gen = report_data_gen["statistiche_riassuntive"]
                col1_metric, col2_metric, col3_metric = st.columns(3)
                with col1_metric:
                    st.metric("Entità Grezze Totali (LLM+NER)", stats_gen.get("entita_rilevate_totali_grezze", 0))
                with col2_metric:
                    st.metric("Entità Uniche (Testo+Tipo)", stats_gen.get("entita_uniche_testo_tipo", 0))
                with col3_metric:
                    st.metric("PII Uniche Identificate", stats_gen.get("pii_identificate_uniche", 0))

                if stats_gen.get("conteggio_per_tipo_pii"):
                    st.markdown("#### Conteggio PII Uniche per Tipo:")
                    df_counts = pd.DataFrame(list(stats_gen["conteggio_per_tipo_pii"].items()),
                                             columns=['Tipo PII', 'Conteggio']).sort_values(by='Conteggio',
                                                                                            ascending=False)
                    st.dataframe(df_counts, use_container_width=True)

                st.subheader("Tabella PII Consolidate (Uniche per Testo e Tipo, con Fonti e Motivazioni Aggregate)")
                df_tabella_pii_gen = report_data_gen.get("tabella_pii_consolidate", pd.DataFrame())
                if not df_tabella_pii_gen.empty:
                    cols_display_order = ["Testo Entità", "Tipo", "Contesto (da LLM)", "Motivazione (da LLM)",
                                          "Fonti Rilevamento", "Score Max (da NER)"]
                    cols_presenti = [col for col in cols_display_order if col in df_tabella_pii_gen.columns]
                    st.dataframe(df_tabella_pii_gen[cols_presenti], use_container_width=True, height=300)
                    csv_general_pii_gen = df_tabella_pii_gen.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Scarica Tabella PII Consolidate (CSV)", data=csv_general_pii_gen,
                                       file_name="report_generale_pii_consolidate.csv", mime="text/csv",
                                       key="download_general_pii_csv_tab_main", use_container_width=True)
                else:
                    st.info("Nessuna PII specifica trovata da aggregare per la tabella consolidata.")

                st.subheader("Riassunti del Testo (dagli LLM)")
                if report_data_gen.get("riassunti_llm"):
                    for riassunto_llm_gen in report_data_gen["riassunti_llm"]:
                        with st.expander(f"Riassunto da {riassunto_llm_gen['Modello']}"):
                            st.markdown(riassunto_llm_gen["Riassunto"])
                else:
                    st.info("Nessun riassunto fornito dai modelli LLM.")

                st.markdown("---")
                subheader_label_consolidated_doc_display = "📄 Documento Modificato Consolidato"
                is_general_edited_doc_csv = isinstance(st.session_state.get("general_edited_document"), pd.DataFrame)
                if is_general_edited_doc_csv:
                    subheader_label_consolidated_doc_display = "📄 CSV Modificato Consolidato (basato su Report Generale)"

                st.subheader(subheader_label_consolidated_doc_display)
                if st.session_state.get("general_edited_document") is not None:
                    if is_general_edited_doc_csv:
                        st.dataframe(st.session_state["general_edited_document"], height=300, use_container_width=True)
                        csv_output_bytes_consolidated = st.session_state["general_edited_document"].to_csv(
                            index=False).encode('utf-8')
                        st.download_button(label="Scarica CSV Modificato Consolidato (.csv)",
                                           data=csv_output_bytes_consolidated,
                                           file_name="documento_modificato_consolidato.csv", mime="text/csv",
                                           key="download_general_edited_csv_button_tab", use_container_width=True)
                    else:
                        general_edited_text_content_display = st.session_state["general_edited_document"]
                        st.text_area("Testo Modificato Consolidato", general_edited_text_content_display, height=300,
                                     key="general_edited_text_area_display_tab", disabled=True)
                        general_edited_doc_bytes_display = write_file(general_edited_text_content_display, ".txt")
                        st.download_button(label="Scarica Documento Modificato Consolidato (.txt)",
                                           data=general_edited_doc_bytes_display,
                                           file_name="documento_modificato_consolidato.txt", mime="text/plain",
                                           key="download_general_edited_txt_button_tab", use_container_width=True)
                else:
                    st.info(
                        "Nessun documento o CSV modificato consolidato disponibile. Genera il Report Generale e poi "
                        "clicca sull'azione di generazione del documento/CSV modificato.")
            else:
                st.info(
                    "Clicca su 'Genera/Aggiorna Report Generale' (nella sezione Azioni) dopo aver eseguito almeno "
                    "un'analisi.")


if __name__ == "__main__":
    main()
