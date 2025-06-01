from typing import Optional

import streamlit as st
import pandas as pd
import json
import asyncio
import os
import numpy as np

from modules.text_extractor import detect_extension, extract_text
from modules.generazione_testo import generate_report  # Chiamata diretta
from modules.analyisis_cvs import analyze_and_anonymize_csv, get_llm_overall_csv_comment, logger  # Chiamate dirette
from modules.privacy_metrics import calculate_k_anonymity, calculate_l_diversity
from modules.config import LLM_MODELS


# Le funzioni decorate con @st.cache_data sono state rimosse.

# Funzione per badge colorati (invariata)
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


# ───────────────────────────────────────────────────────────────────
# Funzioni Helper per la Logica di Analisi
# ───────────────────────────────────────────────────────────────────
def _perform_text_analysis(text_to_analyze: str):
    st.info("⏳ Esecuzione analisi PII sul testo...")
    reports_text_results = {}
    if not LLM_MODELS:
        st.error("Nessun modello LLM configurato.")
        return

    text_analysis_progress = st.progress(0, text="Avvio analisi testo...")
    total_models = len(LLM_MODELS)
    for i, (nome_mod, mod_api) in enumerate(LLM_MODELS.items()):
        progress_percent = int(((i + 1) / total_models) * 100)
        text_analysis_progress.progress(progress_percent,
                                        text=f"Analisi PII (Testo) con {nome_mod} ({i + 1}/{total_models})...")
        try:
            # Chiamata diretta alla funzione originale
            rep = generate_report(text_to_analyze, mod_api)
            reports_text_results[nome_mod] = rep
        except Exception as e:
            st.error(f"Errore durante l'analisi con {nome_mod}: {e}")
            reports_text_results[nome_mod] = {"error": str(e), "found": False, "entities": [],
                                              "summary": f"Errore: {e}"}

    text_analysis_progress.empty()
    st.session_state["reports_text"] = reports_text_results
    st.success("✅ Analisi PII su testo completata!")


def _perform_csv_analysis(df_to_analyze_full: pd.DataFrame, selected_model_name: Optional[str]):
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

    csv_progress_bar = st.progress(0, text="Avvio analisi CSV...")

    # --- Chiamata ad analyze_and_anonymize_csv dal modulo modules.analyisis_cvs ---
    # Questa funzione ora riceve il DataFrame completo e gestisce internamente
    # la separazione dei tipi di colonna e il passaggio dei parametri num_bins/granularity_date
    # Assicurati che la firma di analyze_and_anonymize_csv in modules/analyisis_cvs.py
    # accetti num_bins_numeric e granularity_for_dates se li vuoi passare da qui.
    # Dall'ultima iterazione, questi erano parametri di analyze_and_anonymize_csv.
    num_bins_param = st.session_state.get("num_bins_for_numeric", 5)  # Default a 5
    granularity_date_param = st.session_state.get("granularity_for_date", "M")  # Default a "M"

    try:
        # Chiamata diretta alla funzione analyze_and_anonymize_csv (asincrona)
        # Questa funzione ora gestisce l'analisi di tutti i tipi di colonna (testo, num, date)
        report_cols_df_result, df_anon_result = asyncio.run(
            analyze_and_anonymize_csv(
                df_to_analyze_full,
                mod_api_csv_selected,  # Può essere None se nessun modello LLM è selezionato/disponibile
                sample_size_for_preview=5,  # Valore di default o da config
                max_concurrent_requests=3,  # Valore di default o da config
                num_bins_numeric=num_bins_param,
                granularity_for_dates=granularity_date_param
            )
        )
        st.session_state["csv_analysis_report_df"] = report_cols_df_result
        st.session_state["csv_anon_df"] = df_anon_result
        csv_progress_bar.progress(40, text="Analisi colonne (testo, numeriche, date) completata.")

    except Exception as e_analyze_cols:
        st.error(f"Errore grave durante l'analisi completa delle colonne CSV: {e_analyze_cols}")
        logger.error(f"Errore in analyze_and_anonymize_csv: {e_analyze_cols}", exc_info=True)
        st.session_state["csv_analysis_report_df"] = pd.DataFrame()
        st.session_state["csv_anon_df"] = df_to_analyze_full.copy()  # Fallback: df non modificato
        csv_progress_bar.progress(40, text="Errore analisi colonne.")
        # Interrompi qui se l'analisi colonne fallisce gravemente
        csv_progress_bar.empty()
        st.error("Analisi CSV interrotta a causa di un errore nel processamento delle colonne.")
        return

    # --- Calcolo Loss di Utilità (già presente, usa df_to_analyze_full e df_anon_result) ---
    csv_progress_bar.progress(60, text="Passo 2/3: Calcolo metriche di perdita di utilità...")
    loss_metrics_results = {"n_righe_originali": len(df_to_analyze_full),
                            "n_righe_anonimizzate": len(st.session_state["csv_anon_df"]) if st.session_state.get(
                                "csv_anon_df") is not None else len(df_to_analyze_full)}

    # Identifica le colonne numeriche originali per il calcolo della loss of utility
    original_numeric_cols_for_loss = df_to_analyze_full.select_dtypes(include=["int64", "float64"])
    if not original_numeric_cols_for_loss.empty:
        col_esempio_util = original_numeric_cols_for_loss.columns[0]
        original_series_util = original_numeric_cols_for_loss[col_esempio_util].dropna()

        if pd.api.types.is_numeric_dtype(original_series_util) and not original_series_util.empty and len(
                original_series_util) >= 2:
            var_originale_util = original_series_util.var()

            # Assumiamo che df_anon_result (st.session_state["csv_anon_df"]) contenga le colonne generalizzate come stringhe di fascia
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
                var_generalizzata_util = pd.to_numeric(midpoints_util,
                                                       errors='coerce').dropna().var()  # Aggiunto dropna() prima di
                # .var()

                preserved_util = (var_generalizzata_util / var_originale_util) * 100 if pd.notna(
                    var_originale_util) and var_originale_util > 1e-6 and pd.notna(
                    var_generalizzata_util) else 0.0  # Evita divisione per zero o var molto piccole

                loss_metrics_results[col_esempio_util] = {
                    "var_originale": var_originale_util if pd.notna(var_originale_util) else 'N/D',
                    "var_generalizzata_stimata": var_generalizzata_util if pd.notna(var_generalizzata_util) else 'N/D',
                    "percentuale_preserved_var (%)": round(preserved_util, 1) if preserved_util != 'N/D' else 'N/D'
                    # Arrotondato a 1 decimale
                }
            else:
                loss_metrics_results[col_esempio_util] = {
                    "error": "Colonna non trovata nel df anonimizzato o df anonimizzato non presente."}
        else:
            loss_metrics_results[col_esempio_util] = {"error": "Serie originale non numerica o con meno di 2 valori."}

    st.session_state["loss_of_utility_metrics"] = loss_metrics_results
    csv_progress_bar.progress(70, text="Metriche di perdita di utilità calcolate.")

    # --- Passi finali: Metriche di Rischio e Report Finale LLM ---
    report_cols_for_final_metrics = st.session_state.get("csv_analysis_report_df")
    if report_cols_for_final_metrics is not None and not report_cols_for_final_metrics.empty:
        csv_progress_bar.progress(75, text="Passo 3/3: Calcolo metriche di rischio e generazione report finale...")
        qids_final, sas_final = [], []
        if "CategoriaLLM" in report_cols_for_final_metrics.columns:
            mask_qid_final = report_cols_for_final_metrics["CategoriaLLM"].str.lower().str.contains(
                r"quasi[\s\-]?identificatore", na=False, regex=True)
            candidati_qid_final = report_cols_for_final_metrics[mask_qid_final]["Colonna"].tolist()
            df_sample_metrics_final = df_to_analyze_full  # Usa il DataFrame originale completo per le metriche
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

        # --- CHIAMATA CORRETTA A get_llm_overall_csv_comment ---
        try:
            overall_md_final_result = get_llm_overall_csv_comment(
                report_cols_for_final_metrics,  # column_analysis_report
                st.session_state.get("calculated_risk_metrics", {}),  # risk_metrics_calculated
                st.session_state.get("loss_of_utility_metrics"),  # loss_of_utility_metrics
                st.session_state.get("identified_qids_for_summary", []),  # qid_identified_list
                st.session_state.get("identified_sas_for_summary", []),  # sa_identified_list
                mod_api_csv_selected,  # model_api_id (può essere None)
                file_name_for_llm_report  # file_name
            )
            st.session_state["overall_csv_comment"] = overall_md_final_result
        except Exception as e_rep_f:
            st.error(f"Errore generazione report finale CSV con LLM: {e_rep_f}")
            logger.error(f"Errore in get_llm_overall_csv_comment: {e_rep_f}", exc_info=True)
            st.session_state["overall_csv_comment"] = f"Errore durante la generazione del report LLM: {e_rep_f}"
        # --- FINE CHIAMATA CORRETTA ---
        csv_progress_bar.progress(100, text="Report finale CSV (o tentativo) generato.")
    else:
        st.info(
            "Report colonne combinato vuoto o non disponibile. Impossibile procedere con metriche di rischio finali e "
            "report LLM.")
        csv_progress_bar.progress(100, text="Analisi CSV terminata (dati colonne insufficienti).")

    csv_progress_bar.empty()
    st.success("✅ Analisi CSV potenziata completata!")


# ───────────────────────────────────────────────────────────────────
# UI Principale (Titolo, Input, Azioni, Risultati)
# (Il resto dell'UI rimane come nell'ultima versione fornita,
# assicurati solo che la sezione di visualizzazione dei risultati CSV
# includa la stampa di st.session_state.get("loss_of_utility_metrics"))
# ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🛡️ DataSentinelAI", layout="wide")
st.title("🛡️ DataSentinelAI")
st.markdown(
    """
    Questo strumento rileva automaticamente PII e valuta rischi di privacy nel tuo documento (PDF/TXT/DOCX) o dataset CSV,
    ora con analisi estesa per colonne numeriche/data e una stima della perdita di utilità post-generalizzazione.
    """
)
with st.expander("ℹ️ Come preparare i dati"):
    st.markdown(
        """- **PDF/DOCX/TXT:** Preferibilmente testo selezionabile. File grandi o basati su immagini possono avere 
        performance ridotte o fallire. - **CSV:** Intestazioni chiare. Colonne `object`/`string` sono analizzate da 
        LLM (se modello selezionato). Colonne numeriche (`int`, `float`) e temporali (`datetime`) sono analizzate per 
        cardinalità e generalizzate automaticamente se considerate QID. Per analizzare numeri/date come testo, 
        convertili in stringa nel CSV. - **Testo libero:** Incolla direttamente."""
    )

st.subheader("1. Carica o incolla i tuoi dati")

default_session_state = {
    "raw_text_input": None, "original_csv_df": None, "current_file_ext": None,
    "last_uploaded_filename": None, "text_area_content": "",
    "reports_text": None, "csv_analysis_report_df": None, "csv_anon_df": None,
    "overall_csv_comment": None, "calculated_risk_metrics": None,
    "identified_qids_for_summary": None, "identified_sas_for_summary": None,
    "selected_csv_model_name": None, "loss_of_utility_metrics": None
}
for k_init, v_val_init in default_session_state.items():
    if k_init not in st.session_state: st.session_state[k_init] = v_val_init

MAX_FILE_SIZE_MB = 50
uploaded_file = st.file_uploader(
    "Carica file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"],
    help=f"Scegli un file. Attenzione a file molto grandi (>{MAX_FILE_SIZE_MB}MB)."
)

input_changed_flag_global = False
if uploaded_file is not None:
    if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
        input_changed_flag_global = True
        st.session_state["last_uploaded_filename"] = uploaded_file.name
        file_size_mb_val = uploaded_file.size / (1024 * 1024)
        if file_size_mb_val > MAX_FILE_SIZE_MB:
            st.warning(
                f"Il file '{uploaded_file.name}' è di circa {file_size_mb_val:.1f} MB. L'analisi potrebbe richiedere molto tempo.")
        detected_ext_val = detect_extension(uploaded_file).lower()
        st.session_state["current_file_ext"] = detected_ext_val
        st.success(f"File caricato: **{uploaded_file.name}** (Rilevato come: {detected_ext_val.upper()})")
        if detected_ext_val in [".pdf", ".docx", ".txt"]:
            try:
                extracted_text_content = extract_text(uploaded_file)
                if detected_ext_val == ".pdf" and len(extracted_text_content.strip()) < 100 and file_size_mb_val > 0.05:
                    st.warning("Il PDF caricato contiene poco testo. Potrebbe essere scannerizzato/basato su immagini.")
                st.session_state["raw_text_input"] = extracted_text_content
                st.session_state["original_csv_df"] = None
            except Exception as e_extract:
                st.error(f"Errore estraendo il testo: {e_extract}")
                st.session_state["raw_text_input"] = None
        elif detected_ext_val == ".csv":
            try:
                uploaded_file.seek(0)
                df_loaded_content = pd.read_csv(uploaded_file)
                st.session_state["original_csv_df"] = df_loaded_content
                st.session_state["raw_text_input"] = None
                st.caption("Anteprima CSV (prime 5 righe):")
                st.dataframe(df_loaded_content.head(5), use_container_width=True, height=200)
            except Exception as e_csv_read:
                st.error(f"Errore leggendo il CSV: {e_csv_read}")
                st.session_state["original_csv_df"] = None
        st.session_state["text_area_content"] = ""
    else:
        if st.session_state.current_file_ext: st.success(
            f"File attuale: **{st.session_state.last_uploaded_filename}** (Tipo: {st.session_state.current_file_ext.upper()})")
        if st.session_state.current_file_ext == ".csv" and st.session_state.original_csv_df is not None:
            st.caption("Anteprima CSV (prime 5 righe):")
            st.dataframe(st.session_state.original_csv_df.head(5), use_container_width=True, height=200)
else:
    if st.session_state.get("last_uploaded_filename") is not None:
        input_changed_flag_global = True
        st.session_state["last_uploaded_filename"] = None
        st.session_state["raw_text_input"] = None
        st.session_state["original_csv_df"] = None
        st.session_state["current_file_ext"] = None
    current_text_area_val = st.session_state.get("text_area_content", "")
    text_area_user_input = st.text_area(
        "Oppure incolla qui il tuo testo", value=current_text_area_val,
        placeholder="Copia‐incolla testo...", height=150, key="text_area_main_v3"
    )
    if text_area_user_input != current_text_area_val:
        input_changed_flag_global = True
        st.session_state["text_area_content"] = text_area_user_input
    if text_area_user_input and text_area_user_input.strip():
        if st.session_state.current_file_ext != ".txt" or st.session_state.raw_text_input != text_area_user_input:
            input_changed_flag_global = True
        st.session_state["raw_text_input"] = text_area_user_input
        st.session_state["current_file_ext"] = ".txt"
        st.session_state["original_csv_df"] = None
    elif not (text_area_user_input and text_area_user_input.strip()) and st.session_state.raw_text_input:
        input_changed_flag_global = True
        st.session_state["raw_text_input"] = None
        st.session_state["current_file_ext"] = None

if input_changed_flag_global:
    st.info(
        "Input cambiato/rimosso. Eventuali risultati precedenti sono stati azzerati. Esegui una nuova analisi se necessario.")
    keys_to_reset_on_input_change = [
        "reports_text", "csv_analysis_report_df", "csv_anon_df",
        "overall_csv_comment", "calculated_risk_metrics", "loss_of_utility_metrics",
        "identified_qids_for_summary", "identified_sas_for_summary",
    ]
    for key_item_reset in keys_to_reset_on_input_change: st.session_state.pop(key_item_reset, None)
    if st.session_state.last_uploaded_filename is None and not st.session_state.text_area_content.strip():
        st.session_state.current_file_ext = None

active_raw_text_input = st.session_state.get("raw_text_input")
active_df_input = st.session_state.get("original_csv_df")
active_file_type_input = st.session_state.get("current_file_ext")

if active_file_type_input == ".csv" and active_df_input is not None and not active_df_input.empty:
    if LLM_MODELS:
        csv_model_options = list(LLM_MODELS.keys())
        csv_default_model_idx = 0
        if st.session_state.get("selected_csv_model_name") in csv_model_options:
            csv_default_model_idx = csv_model_options.index(st.session_state.selected_csv_model_name)
        csv_selected_model = st.selectbox(
            "Modello LLM per analisi colonne testuali CSV:",
            csv_model_options, index=csv_default_model_idx, key="sb_csv_model_selector_v3"
        )
        st.session_state["selected_csv_model_name"] = csv_selected_model
    else:  # No LLM_MODELS configured
        st.session_state["selected_csv_model_name"] = None  # Ensure it's None
        # Non mostrare warning qui, verrà gestito in _perform_csv_analysis

st.subheader("2. Analizza i dati")
analysis_btn_disabled_status = not (
        active_raw_text_input or (active_df_input is not None and not active_df_input.empty))
analysis_btn_label_text = "⚠️ Carica un file o incolla del testo"

if active_file_type_input in [".pdf", ".docx", ".txt"] and active_raw_text_input:
    analysis_btn_label_text = "🚀 Analizza Testo/Documento"
elif active_file_type_input == ".csv" and (active_df_input is not None and not active_df_input.empty):
    csv_selected_model_name_for_btn = st.session_state.get("selected_csv_model_name")
    if csv_selected_model_name_for_btn or not LLM_MODELS:  # Abilita se non ci sono LLM (analisi numerica/date funziona comunque)
        analysis_btn_label_text = f"📊 Analizza CSV"
        if csv_selected_model_name_for_btn and LLM_MODELS:  # Aggiungi nome modello solo se LLM sono configurati e uno è scelto
            analysis_btn_label_text += f" (testo con {csv_selected_model_name_for_btn})"
    else:  # LLM configurati, ma nessuno selezionato
        analysis_btn_label_text = "📊 Seleziona Modello LLM per colonne testuali CSV"
        analysis_btn_disabled_status = True

main_analyze_button = st.button(analysis_btn_label_text, disabled=analysis_btn_disabled_status,
                                use_container_width=True)
if analysis_btn_disabled_status and active_file_type_input == ".csv" and not st.session_state.get(
        "selected_csv_model_name") and LLM_MODELS:
    st.caption("ℹ️ Seleziona un modello LLM (sopra) per l'analisi delle colonne testuali del CSV.")

if main_analyze_button:
    keys_to_clear_before_analysis = [
        "reports_text", "csv_analysis_report_df", "csv_anon_df", "overall_csv_comment",
        "calculated_risk_metrics", "loss_of_utility_metrics",
        "identified_qids_for_summary", "identified_sas_for_summary"
    ]
    for key_clear in keys_to_clear_before_analysis: st.session_state.pop(key_clear, None)

    if active_raw_text_input and active_file_type_input in [".pdf", ".docx", ".txt"]:
        _perform_text_analysis(active_raw_text_input)
    elif active_df_input is not None and not active_df_input.empty and active_file_type_input == ".csv":
        _perform_csv_analysis(active_df_input, st.session_state.get("selected_csv_model_name"))

# Visualizzazione Risultati
has_text_results = st.session_state.get("reports_text") is not None
csv_report_df_val = st.session_state.get("csv_analysis_report_df")
has_csv_report_cols_val = csv_report_df_val is not None and not csv_report_df_val.empty
csv_anon_df_val = st.session_state.get("csv_anon_df")
has_csv_anon_df_val = csv_anon_df_val is not None and not csv_anon_df_val.empty
has_csv_overall_val = st.session_state.get("overall_csv_comment") is not None
has_loss_metrics_val = st.session_state.get("loss_of_utility_metrics") is not None
show_results_ui_section_val = has_text_results or has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val

if show_results_ui_section_val:
    st.subheader("3. Visualizza Risultati")
    result_tab_titles_list = []
    if has_text_results: result_tab_titles_list.append("🔍 Testo/Documento")
    if has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val:
        if "📊 CSV" not in result_tab_titles_list: result_tab_titles_list.append("📊 CSV")

    if not result_tab_titles_list and main_analyze_button:
        st.info("Analisi completata, ma nessun risultato specifico da visualizzare per i criteri attuali.")
    elif result_tab_titles_list:
        displayed_tabs_list = st.tabs(result_tab_titles_list)
        current_display_tab_idx_val = 0

        if "🔍 Testo/Documento" in result_tab_titles_list:
            with displayed_tabs_list[current_display_tab_idx_val]:
                current_display_tab_idx_val += 1
                current_text_reports_val = st.session_state.get("reports_text")
                if current_text_reports_val:
                    st.header("Report PII da LLM (Analisi su Testo/Documento)")
                    for model_name_txt_rep_val, report_data_txt_rep_val in current_text_reports_val.items():
                        st.markdown(f"#### Report da: **{model_name_txt_rep_val}**")
                        if isinstance(report_data_txt_rep_val, dict):
                            if report_data_txt_rep_val.get("error"):
                                st.error(f"Errore per {model_name_txt_rep_val}: {report_data_txt_rep_val.get('error')}")
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
                                                f"  - Info Chunk/Segmento: _{entity_item_row_val['source_chunk_info']}_")
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
                                    mime="application/json", key=f"dl_json_txt_{model_name_txt_rep_val}_v3",
                                    use_container_width=False)
                            except Exception as e_json_text_dl_val:
                                st.error(f"Errore preparazione JSON download: {e_json_text_dl_val}")
                        else:
                            st.text(str(report_data_txt_rep_val))
                        st.markdown("---")
                else:
                    st.info("Nessun report di analisi testuale disponibile.")

        if "📊 CSV" in result_tab_titles_list:
            with displayed_tabs_list[current_display_tab_idx_val]:
                csv_report_cols_data_disp_val = st.session_state.get("csv_analysis_report_df")
                csv_overall_md_content_disp_val = st.session_state.get("overall_csv_comment")
                csv_df_anon_content_disp_val = st.session_state.get("csv_anon_df")
                utility_metrics_display_val = st.session_state.get("loss_of_utility_metrics")
                st.header("Risultati Analisi CSV")

                if csv_report_cols_data_disp_val is not None and not csv_report_cols_data_disp_val.empty:
                    st.markdown("#### Dettaglio Analisi per Colonna (Testuali LLM, Numeriche/Date Regole)")
                    csv_cols_mask_examine_val = (csv_report_cols_data_disp_val.get("LLM_HaTrovatoEntitaPII",
                                                                                   pd.Series(dtype=bool)) == True) | \
                                                (csv_report_cols_data_disp_val.get("MetodoSuggerito", pd.Series(
                                                    dtype=str)).str.lower() != "nessuno") | \
                                                (csv_report_cols_data_disp_val.get("Problematica",
                                                                                   pd.Series(dtype=str)).str.contains(
                                                    "Errore", na=False, case=False)) | \
                                                (csv_report_cols_data_disp_val.get("CategoriaLLM",
                                                                                   pd.Series(dtype=str)).str.contains(
                                                    "Quasi-Identificatore", na=False, case=False))
                    df_user_csv_display_val = csv_report_cols_data_disp_val[csv_cols_mask_examine_val] if \
                        all(k_csv_check_val in csv_report_cols_data_disp_val for k_csv_check_val in
                            ["LLM_HaTrovatoEntitaPII", "MetodoSuggerito", "Problematica", "CategoriaLLM"]) \
                        else csv_report_cols_data_disp_val
                    if df_user_csv_display_val.empty and not csv_report_cols_data_disp_val.empty:
                        st.success(
                            "✅ Nessuna colonna sembra richiedere un intervento di anonimizzazione specifico o contenere PII dirette rilevanti secondo l'analisi.")
                    elif not df_user_csv_display_val.empty:
                        for _, r_col_csv_item_val in df_user_csv_display_val.iterrows():
                            col_name_csv_disp_val = r_col_csv_item_val.get("Colonna", "N/A")
                            cat_llm_csv_disp_val = r_col_csv_item_val.get("CategoriaLLM", "N/A")
                            st.markdown(f"**Colonna: `{col_name_csv_disp_val}`**")
                            st.markdown(f"Categoria Rilevata: {colored_badge(cat_llm_csv_disp_val)}",
                                        unsafe_allow_html=True)
                            st.caption(f"Esempi: {r_col_csv_item_val.get('Esempi', 'N/A')}")
                            problem_csv_disp_val = r_col_csv_item_val.get("Problematica", "N/A")
                            if "Errore" in problem_csv_disp_val:
                                st.error(problem_csv_disp_val)
                            elif "non ha rilevato PII specifiche" in problem_csv_disp_val.lower() and "Valutazione colonna" in problem_csv_disp_val:
                                st.info(problem_csv_disp_val)
                            else:
                                st.warning(problem_csv_disp_val)
                            st.info(
                                f"Metodo Anon./Gestione Suggerito: **{r_col_csv_item_val.get('MetodoSuggerito', 'N/A')}**\n> Motivazione: _{r_col_csv_item_val.get('Motivazione', 'N/A')}_")
                            st.markdown("---")
                elif csv_report_cols_data_disp_val is not None and csv_report_cols_data_disp_val.empty and main_analyze_button:
                    st.info(
                        "L'analisi delle colonne del CSV non ha prodotto un report dettagliato (es. nessuna colonna analizzabile trovata o errore).")

                if csv_overall_md_content_disp_val:
                    st.markdown("---")
                    st.markdown("### 📈 Report Privacy Complessivo (CSV)", unsafe_allow_html=True)
                    st.markdown(csv_overall_md_content_disp_val, unsafe_allow_html=True)
                    if st.button("Rimuovi Report Privacy Complessivo", key="clear_overall_csv_btn_v4"):
                        st.session_state["overall_csv_comment"] = None
                        st.rerun()

                if isinstance(utility_metrics_display_val, dict) and utility_metrics_display_val:
                    st.markdown("---")
                    st.markdown("#### 📉 Metrica di Perdita di Utilità (Post-Generalizzazione)")
                    st.write(
                        f"- Numero righe dataset originale: **{utility_metrics_display_val.get('n_righe_originali', 'N/D')}**")
                    st.write(
                        f"- Numero righe dataset generalizzato/anonimizzato: **{utility_metrics_display_val.get('n_righe_anonimizzate', 'N/D')}**")
                    for col_metric_util_val, metrics_data_util_val in utility_metrics_display_val.items():
                        if isinstance(metrics_data_util_val, dict):
                            st.write(f"- Colonna di esempio per varianza: **{col_metric_util_val}**")
                            var_orig_disp_val = metrics_data_util_val.get('var_originale', 'N/D')
                            var_gen_disp_val = metrics_data_util_val.get('var_generalizzata_stimata', 'N/D')
                            perc_pres_disp_val = metrics_data_util_val.get('percentuale_preserved_var (%)', 'N/D')
                            st.write(
                                f"  - Varianza originale: {'{:.2f}'.format(var_orig_disp_val) if isinstance(var_orig_disp_val, (int, float)) else var_orig_disp_val}")
                            st.write(
                                f"  - Varianza generalizzata (stimata da fasce): {'{:.2f}'.format(var_gen_disp_val) if isinstance(var_gen_disp_val, (int, float)) else var_gen_disp_val}")
                            st.write(f"  - Percentuale di varianza preservata: {perc_pres_disp_val}%")

                st.markdown("---")
                if isinstance(csv_df_anon_content_disp_val, pd.DataFrame) and not csv_df_anon_content_disp_val.empty:
                    csv_bytes_dl_anon_val = csv_df_anon_content_disp_val.to_csv(index=False).encode("utf-8")
                    base_fn_val, ext_fn_val = os.path.splitext(
                        st.session_state.get('last_uploaded_filename', 'dataset.csv'))
                    dl_fn_anon_val = f"anonimizzato_generalizzato_{base_fn_val}{ext_fn_val if ext_fn_val else '.csv'}"
                    st.download_button(
                        label="📥 Scarica CSV Generalizzato/Anonimizzato",
                        data=csv_bytes_dl_anon_val, file_name=dl_fn_anon_val, mime="text/csv",
                        use_container_width=True, key="dl_anon_csv_sugg_btn_v4")
                elif active_file_type_input == ".csv" and main_analyze_button:
                    st.info("Nessun CSV Generalizzato/Anonimizzato disponibile.")

                if not (
                        has_csv_report_cols_val or has_csv_anon_df_val or has_csv_overall_val or has_loss_metrics_val) and \
                        active_file_type_input == ".csv" and main_analyze_button:
                    st.info("Nessun risultato specifico dall'analisi CSV da visualizzare.")
else:
    if not (active_raw_text_input or (active_df_input is not None and not active_df_input.empty)):
        st.caption(
            "Carica un file (PDF, TXT, DOCX, CSV) o incolla del testo, seleziona opzioni e clicca il bottone di analisi per visualizzare i risultati qui.")
    else:
        st.caption("Esegui un'analisi per visualizzare i risultati.")
