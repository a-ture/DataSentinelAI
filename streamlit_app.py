import streamlit as st
import pandas as pd
import json

from modules.config import LLM_MODELS
from modules.text_extractor import detect_extension, extract_text
from modules.generazione_testo import (
    generate_report,
    edit_document,
    extract_entities,
    sensitive_informations as get_sensitive_contexts,
)
from modules.utils import write_file

# Tipi di entità da considerare PII
PII_TYPES = [
    "PERSON", "PER", "PERS", "person", "persona", "nome",
    "DATE", "data",
    "LOCATION", "LOC", "location", "luogo", "indirizzo", "address", "full address", "comune",
    "ORGANIZATION", "ORG", "organization", "società", "company",
    "codice fiscale", "national id",
    "phone number", "numero di telefono",
    "email address", "email",
    "credit card number", "numero carta di credito",
    "CUI",
    "importo",
    "postal code", "codice postale"
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
                        "Score": "N/A"
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
                                                                                                               "N/A")
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
            df_pii_originali = df_finale_entita[df_finale_entita["Tipo"].str.lower().isin(pii_types_list_lower) & \
                                                df_finale_entita.apply(lambda x: (x['Testo Entità'], x['Tipo']) in \
                                                                                 set(zip(df_pii_consolidate_uniche[
                                                                                             'Testo Entità'],
                                                                                         df_pii_consolidate_uniche[
                                                                                             'Tipo'])), axis=1)]

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

            if not df_pii_originali.empty:
                df_visualizzazione_pii = df_pii_originali.groupby(["Testo Entità", "Tipo"], as_index=False).apply(
                    aggrega_fonti_pii)
                dati_consolidati["tabella_pii_consolidate"] = df_visualizzazione_pii.reset_index(drop=True)
    else:
        for key_stat in ["entita_rilevate_totali_grezze", "entita_uniche_testo_tipo", "pii_identificate_uniche"]:
            dati_consolidati["statistiche_riassuntive"][key_stat] = 0
        dati_consolidati["statistiche_riassuntive"]["conteggio_per_tipo_pii"] = {}

    return dati_consolidati


def main():
    st.set_page_config(page_title="DataSentinelAI", layout="wide")

    if "reports" not in st.session_state:
        st.session_state["reports"] = {}
    if "edited_docs" not in st.session_state:
        st.session_state["edited_docs"] = {}
    if "ner_entities" not in st.session_state:
        st.session_state["ner_entities"] = pd.DataFrame()
    if "general_report_data" not in st.session_state:
        st.session_state["general_report_data"] = None
    if "raw_text_input" not in st.session_state:
        st.session_state["raw_text_input"] = ""
    if "current_file_ext" not in st.session_state:
        st.session_state["current_file_ext"] = ".txt"
    if "last_uploaded_filename" not in st.session_state:
        st.session_state["last_uploaded_filename"] = None
    # Rimuoviamo current_action dallo stato se non necessario per il layout tab
    # if "current_action" not in st.session_state:
    #     st.session_state["current_action"] = "Genera Report PII"

    st.title("🛡️ DataSentinelAI")

    # --- Sezione 1: Input Dati ---
    with st.container():
        st.subheader("1. Fornisci il Testo")
        input_type = st.radio("Scegli tipo di input:", ["Testo libero", "Carica File"], key="input_type_radio_main",
                              horizontal=True)

        raw_text_changed = False
        if input_type == "Testo libero":
            user_text_area = st.text_area("Inserisci il testo qui:", value=st.session_state["raw_text_input"],
                                          height=150, key="text_area_input_main", label_visibility="collapsed")
            if user_text_area != st.session_state["raw_text_input"]:
                st.session_state["raw_text_input"] = user_text_area
                st.session_state["current_file_ext"] = ".txt"
                raw_text_changed = True
                st.session_state["last_uploaded_filename"] = None
        else:
            uploaded_file = st.file_uploader("Carica un file (PDF, DOCX, TXT, CSV)", type=["pdf", "docx", "txt", "csv"],
                                             key="file_uploader_main", label_visibility="collapsed")
            if uploaded_file:
                if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.session_state["current_file_ext"] = detect_extension(uploaded_file)
                    try:
                        st.session_state["raw_text_input"] = extract_text(uploaded_file)
                        raw_text_changed = True
                        if st.session_state["current_file_ext"] == ".csv":
                            uploaded_file.seek(0)
                            try:
                                df_preview = pd.read_csv(uploaded_file)
                                st.caption("Anteprima CSV (prime 5 righe):")
                                st.dataframe(df_preview.head(), height=150, use_container_width=True)
                            except Exception as e:
                                st.error(f"Errore anteprima CSV: {e}")
                    except Exception as e:
                        st.error(f"Errore estrazione testo: {e}")
                        st.session_state["raw_text_input"] = ""
                        st.session_state["last_uploaded_filename"] = None
            elif st.session_state["last_uploaded_filename"] is not None:
                st.session_state["raw_text_input"] = ""
                st.session_state["last_uploaded_filename"] = None
                raw_text_changed = True

        if raw_text_changed:
            st.session_state["reports"] = {}
            st.session_state["edited_docs"] = {}
            st.session_state["ner_entities"] = pd.DataFrame()
            st.session_state["general_report_data"] = None
            st.rerun()

    raw_text_to_process = st.session_state["raw_text_input"]

    st.markdown("---")

    # --- Sezione 2: Azioni di Analisi ---
    with st.container():
        st.subheader("2. Esegui Azioni di Analisi")

        action_cols = st.columns(2)

        with action_cols[0]:
            if st.button("🚀 Analizza PII con LLM", key="btn_analyze_pii_llm", use_container_width=True,
                         disabled=not raw_text_to_process.strip()):
                if raw_text_to_process.strip():
                    with st.spinner("Generazione report PII in corso..."):
                        reports_data_llm = {}
                        active_llm_models = LLM_MODELS
                        total_models = len(active_llm_models)
                        if total_models > 0:
                            progress_bar = st.progress(0, text="Avvio analisi LLM...")
                            for i, (model_name, model_api_id) in enumerate(active_llm_models.items(), start=1):
                                progress_bar.progress(i / total_models, text=f"Analisi con {model_name}...")
                                reports_data_llm[model_name] = generate_report(raw_text_to_process, model_api_id)
                            st.session_state["reports"] = reports_data_llm
                            progress_bar.progress(1.0, text="Report PII dagli LLM completati!")
                            st.success("Report PII dagli LLM completati!")
                            st.toast("Analisi Report PII completata!", icon="✅")
                            st.info(
                                "ℹ️ Report PII generati. Ora puoi visualizzarli nella tab 'Report PII (LLM)' o procedere a modificare il documento.")  # NUOVO FEEDBACK
                        else:
                            st.error("Nessun modello LLM configurato in config.py")

            if st.button("✏️ Modifica Documento (basato su Report LLM)", key="btn_edit_doc", use_container_width=True,
                         disabled=not (raw_text_to_process.strip() and st.session_state.get("reports") and any(
                             st.session_state["reports"].values()))):
                if raw_text_to_process.strip() and st.session_state.get("reports") and any(
                        st.session_state["reports"].values()):
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
                        st.toast("Documenti modificati!", icon="�")
                        st.info(
                            "ℹ️ Documenti modificati. Visualizzali e scaricali nella tab 'Documenti Modificati'.")  # NUOVO FEEDBACK
                elif not raw_text_to_process.strip():
                    st.error("Il testo di input è vuoto.")
                else:
                    st.warning("Genera prima i 'Report PII con LLM' per poter modificare il documento.")

        with action_cols[1]:
            if st.button("✨ Esegui NER Dedicata", key="btn_run_ner", use_container_width=True,
                         disabled=not raw_text_to_process.strip()):
                if raw_text_to_process.strip():
                    with st.spinner("Estrazione entità NER dedicata in corso..."):
                        ner_entities_list_loc = extract_entities(raw_text_to_process)
                        st.session_state["ner_entities"] = pd.DataFrame(
                            ner_entities_list_loc) if ner_entities_list_loc else pd.DataFrame()
                        st.success("Analisi NER dedicata completata.")
                        st.toast("Analisi NER completata!", icon="🔖")
                        st.info(
                            "ℹ️ Entità NER estratte. Visualizzale nella tab 'Entità NER Dedicata'.")  # NUOVO FEEDBACK

            if st.button("📊 Genera/Aggiorna Report Generale", key="btn_general_report_main", use_container_width=True,
                         disabled=not (st.session_state.get("reports") or not st.session_state.get("ner_entities",
                                                                                                   pd.DataFrame()).empty)):
                if st.session_state.get("reports") or not st.session_state.get("ner_entities", pd.DataFrame()).empty:
                    with st.spinner("Creazione del Report Generale in corso..."):
                        st.session_state["general_report_data"] = genera_dati_report_generale(
                            st.session_state["reports"],
                            st.session_state["ner_entities"],
                            PII_TYPES_LOWER
                        )
                    st.success("Report Generale pronto/aggiornato!")
                    st.toast("Report Generale creato/aggiornato.", icon="📄")
                    st.info("ℹ️ Report Generale disponibile nella tab 'Report Generale Consolidato'.")  # NUOVO FEEDBACK
                else:
                    st.warning("Esegui prima un'analisi (Report PII o NER) per avere dati da aggregare.")

    st.markdown("---")

    if not raw_text_to_process.strip():
        st.info("Inserisci del testo o carica un file e scegli un'azione per visualizzare i risultati.")
        return

        # --- Sezione 3: Risultati ---
    with st.container():
        st.subheader("3. Visualizza Risultati")

        # Definisci le tab
        tab_titles = ["🔎 Report PII (LLM)", "🔖 Entità NER Dedicata", "✏️ Documenti Modificati",
                      "📊 Report Generale Consolidato"]
        tab_llm_reports, tab_ner_dedicated, tab_edited_docs, tab_general_report_display = st.tabs(tab_titles)

        with tab_llm_reports:
            # st.header("Report PII dai Modelli LLM") # Rimosso subheader ridondante
            if st.session_state.get("reports") and any(
                    st.session_state["reports"].values()):  # Verifica che ci siano reports non vuoti
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
                                    st.dataframe(df_entities_llm_disp[["type", "text", "context", "reasoning"]],
                                                 use_container_width=True, height=250)

                                    df_pii_llm_disp = df_entities_llm_disp[
                                        df_entities_llm_disp["type"].str.lower().isin(PII_TYPES_LOWER)]
                                    if not df_pii_llm_disp.empty:
                                        with st.expander("🔒 PII Trovati (dettaglio)", expanded=False):
                                            for _, row_pii_llm in df_pii_llm_disp.iterrows():
                                                st.markdown(f"**{row_pii_llm['type']}**: {row_pii_llm['text']}")
                                                st.markdown(f"  - Contesto: _{row_pii_llm.get('context', 'N/A')}_")
                                                st.markdown(f"  - Motivazione: _{row_pii_llm.get('reasoning', 'N/A')}_")
                                                st.markdown("---")
                                    else:
                                        st.caption(
                                            "Nessuna PII specifica trovata da questo LLM (secondo la lista PII_TYPES).")
                                else:
                                    st.caption("Nessuna entità trovata o report LLM non valido/vuoto.")
                                st.markdown(f"**Riassunto (LLM)**: {report_content_display.get('summary', 'N/A')}")
                                if "raw_output" in report_content_display and "Error:" in report_content_display.get(
                                        "summary", ""):
                                    with st.expander("Mostra output grezzo dell'errore LLM"):
                                        st.text(report_content_display["raw_output"])
                                report_json_str_disp = json.dumps(report_content_display, indent=2, ensure_ascii=False)
                                st.download_button(label=f"Scarica Report JSON ({model_name_display})",
                                                   data=write_file(report_json_str_disp, ".json"),
                                                   file_name=f"report_pii_{model_name_display.replace(' ', '_')}.json",
                                                   mime="application/json",
                                                   key=f"download_json_tab_{model_name_display}",
                                                   use_container_width=True)
                            else:
                                st.text(str(report_content_display))

                            if isinstance(report_content_display, dict):
                                st.markdown("---")
                                st.markdown("###### Contesti di Sensibilità (da questo report)")
                                model_api_id_ctx_disp = LLM_MODELS.get(model_name_display)
                                if model_api_id_ctx_disp:
                                    report_json_str_for_ctx_disp = json.dumps(report_content_display)
                                    with st.spinner(f"Recupero contesti e motivazioni da {model_name_display}..."):
                                        contexts_output_disp = get_sensitive_contexts(report_json_str_for_ctx_disp,
                                                                                      model_api_id_ctx_disp)
                                        st.markdown(contexts_output_disp if contexts_output_disp else "N/A")
                                else:
                                    st.warning(
                                        f"ID Modello API non trovato per {model_name_display} per generare contesti.")
                else:
                    st.info("Nessun report LLM disponibile. Esegui 'Analizza PII con LLM'.")
            else:
                st.info("Esegui 'Analizza PII con LLM' per visualizzare i risultati qui.")

        with tab_ner_dedicated:
            # st.header("Entità Rilevate dall'Analisi NER Dedicata")
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
            # st.header("Documenti Modificati (Anonimizzati dagli LLM)")
            if st.session_state.get("edited_docs") and any(st.session_state["edited_docs"].values()):
                for model_name_disp_edit, edited_text_content_disp in st.session_state["edited_docs"].items():
                    with st.expander(f"Documento Modificato da: {model_name_disp_edit}", expanded=True):
                        st.text_area(f"Testo Modificato da {model_name_disp_edit}", edited_text_content_disp,
                                     height=300, key=f"mod_text_disp_tab_{model_name_disp_edit}", disabled=True)
                        st.download_button(label=f"Scarica Doc Modificato ({model_name_disp_edit})",
                                           data=write_file(edited_text_content_disp,
                                                           st.session_state["current_file_ext"]),
                                           file_name=f"doc_modificato_{model_name_disp_edit.replace(' ', '_')}{st.session_state['current_file_ext']}",
                                           mime="text/plain", key=f"download_edited_tab_{model_name_disp_edit}",
                                           use_container_width=True)
            else:
                st.info(
                    "Nessun documento modificato disponibile. Genera prima i 'Report PII con LLM' e poi esegui 'Modifica Documento'.")

        with tab_general_report_display:
            # st.header("Report Generale Consolidato")
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
                    st.dataframe(df_tabella_pii_gen[cols_presenti], use_container_width=True)
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
            else:
                st.info(
                    "Clicca su 'Genera/Aggiorna Report Generale' (nella sezione Azioni) dopo aver eseguito almeno un'analisi.")


if __name__ == "__main__":
    main()
