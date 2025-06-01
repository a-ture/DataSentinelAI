# modules/analyisis_cvs.py

# Importa _init_lmstudio per assicurare che il server sia pronto
# e che openai.api_base e openai.api_key siano impostati.

# Questa costante dovrebbe essere definita qui o importata se usata anche altrove
AVAILABLE_ANON_METHODS = ["hash", "mask", "generalize_date", "truncate", "nessuno"]

# In modules/generazione_testo.py (o dove hai definito get_llm_overall_csv_comment)

import asyncio
import hashlib
import json
import re
from typing import Tuple, Dict, Any

import openai
import pandas as pd

AVAILABLE_ANON_METHODS = ["hash", "mask", "generalize_date", "truncate", "nessuno"]


def get_llm_overall_csv_comment(
        column_analysis_report: pd.DataFrame,
        risk_metrics_calculated: dict,
        qid_identified_list: list,  # Rinominato da qid_identified per chiarezza
        sa_identified_list: list,  # Rinominato da sa_identified per chiarezza
        model_api_id: str,
        file_name: str = "Il file CSV analizzato"
) -> str:
    if column_analysis_report.empty:
        return "### Valutazione Privacy – N/D\n\nNessun report di analisi disponibile per le colonne del CSV."

    num_total_cols_analyzed = len(column_analysis_report)
    pii_mask = pd.Series([False] * num_total_cols_analyzed, index=column_analysis_report.index)

    if "LLM_HaTrovatoEntitaPII" in column_analysis_report.columns:
        pii_mask = column_analysis_report["LLM_HaTrovatoEntitaPII"] == True
    else:
        pii_mask = column_analysis_report["Problematica"].str.contains(
            r"PII.*Rilevat", case=False, na=False, regex=True
        )

    cols_with_pii_df = column_analysis_report[pii_mask]
    num_cols_with_pii = len(cols_with_pii_df)

    risk_pct = 0.0
    if num_total_cols_analyzed > 0:
        risk_pct = round((num_cols_with_pii / num_total_cols_analyzed) * 100, 1)

    k_min_val = risk_metrics_calculated.get("k_anonymity_min", "N/D")
    records_singoli_val = risk_metrics_calculated.get("records_singoli", "N/D")

    risk_level_emoji = "🟢"
    risk_level_text = "Basso"
    if isinstance(records_singoli_val, (int, float)) and records_singoli_val > 0:
        risk_level_emoji = "🛑";
        risk_level_text = "Molto Alto (Record Singoli Presenti)"
    elif isinstance(k_min_val, (int, float)) and k_min_val < 2:
        risk_level_emoji = "🛑";
        risk_level_text = "Alto (Violazione Grave K-Anonymity)"
    elif isinstance(k_min_val, (int, float)) and k_min_val < 5:
        risk_level_emoji = "🔴";
        risk_level_text = "Alto (Violazione K-Anonymity)"
    elif risk_pct >= 50:
        risk_level_emoji = "🔴";
        risk_level_text = "Alto (Diffusione PII)"
    elif risk_pct >= 20 or (isinstance(k_min_val, (int, float)) and k_min_val < 10):
        risk_level_emoji = "🟡";
        risk_level_text = "Medio (Presenza PII / Rischio K-Anonymity)"
    elif num_cols_with_pii > 0:
        risk_level_emoji = "🟡";
        risk_level_text = "Medio-Basso (PII Rilevate)"

    markdown_header = f"### Valutazione Privacy Complessiva – {risk_level_emoji} {risk_level_text} ({risk_pct} %)\n\n"

    if num_cols_with_pii == 0 and not (
            isinstance(records_singoli_val, (int, float)) and records_singoli_val > 0) and not (
            isinstance(k_min_val, (int, float)) and k_min_val < 5):
        sintesi_no_pii = (f"**Sintesi Esecutiva**\nL'analisi per '{file_name}' ({num_total_cols_analyzed} colonne) "
                          "indica che l’LLM **non ha rilevato PII specifiche** nelle colonne e le metriche di re-identificazione (se calcolabili con QID) non mostrano rischi critici. "
                          "Il dataset sembra presentare un rischio di privacy basso basato su questa analisi.")
        return markdown_header + sintesi_no_pii

    reid_risk_table_rows_md = []
    reid_risk_table_rows_md.append(
        f"| k-anonymity (valore minimo k) | {k_min_val} | ≥ 5 | {'✅ OK' if isinstance(k_min_val, (int, float)) and k_min_val >= 5 else ('⚠️ Attenzione!' if isinstance(k_min_val, (int, float)) else 'N/D')} |")
    reid_risk_table_rows_md.append(
        f"| k-anonymity (n. record singoli) | {records_singoli_val} | 0 | {'✅ OK' if isinstance(records_singoli_val, (int, float)) and records_singoli_val == 0 else ('🛑 Rischio Alto!' if isinstance(records_singoli_val, (int, float)) and records_singoli_val > 0 else 'N/D')} |")

    l_diversity_metrics = risk_metrics_calculated.get("l_diversity", {})
    if not l_diversity_metrics and sa_identified_list:
        for sa in sa_identified_list: reid_risk_table_rows_md.append(
            f"| l-diversity ({sa}) | N/D (calcolo non riuscito/SA non valido?) | ≥ 2 | N/D |")
    elif not sa_identified_list:
        reid_risk_table_rows_md.append(
            f"| l-diversity | N/A (Nessun Attributo Sensibile Selezionato/Identificato) | ≥ 2 | N/A |")
    else:
        for sa_col in sa_identified_list:
            l_data = l_diversity_metrics.get(sa_col, {})
            l_val = l_data.get("l_min", "N/D")
            reid_risk_table_rows_md.append(
                f"| l-diversity ({sa_col}) | {l_val} | ≥ 2 | {'✅ OK' if isinstance(l_val, (int, float)) and l_val >= 2 else ('⚠️ Attenzione!' if isinstance(l_val, (int, float)) else 'N/D')} |")
    table_metriche_md_str = "\n".join(reid_risk_table_rows_md)

    display_cols_for_pii_table_df = cols_with_pii_df.head(5)
    pii_cols_table_rows_md = []
    for idx, row in display_cols_for_pii_table_df.iterrows():
        col_name = row['Colonna']
        method_sugg = row['MetodoSuggerito']
        problem_full_text = str(row["Problematica"])
        sensitivity_reason = problem_full_text.split('\n\n')[0]
        if sensitivity_reason.startswith(f"**Valutazione Complessiva Sensibilità Colonna '{col_name}':**"):
            sensitivity_reason = sensitivity_reason.replace(
                f"**Valutazione Complessiva Sensibilità Colonna '{col_name}':**", "").strip()
        else:
            if "PII Specifiche Rilevate" in problem_full_text:
                parts = problem_full_text.split("PII Specifiche Rilevate"); sensitivity_reason = parts[
                    0].strip() if len(parts) > 0 else "Sensibilità rilevata"
            elif problem_full_text:
                sensitivity_reason = problem_full_text.split('.')[0]
            else:
                sensitivity_reason = "Informazione sensibile rilevata"
        sensitivity_reason_short = (sensitivity_reason[:60] + '...') if len(
            sensitivity_reason) > 63 else sensitivity_reason
        pii_cols_table_rows_md.append(f"| {col_name} | {method_sugg} | {sensitivity_reason_short.replace('|', ' ')} |")
    pii_cols_table_md_str = "\n".join(
        pii_cols_table_rows_md) if pii_cols_table_rows_md else "| Nessuna colonna specifica con PII da elencare qui. | - | - |"

    context_summary_for_llm = (
        f"File: '{file_name}'. Colonne testuali totali: {num_total_cols_analyzed}. Colonne con PII: {num_cols_with_pii}.\n"
        f"Livello Rischio Privacy Stimato: {risk_level_text} ({risk_pct}%).\n"
        f"k-anonymity (min k): {k_min_val}, Record Singoli (k=1): {records_singoli_val}.\n"
        f"l-diversity calcolata per SA: {', '.join(l_diversity_metrics.keys()) if l_diversity_metrics else 'Nessuna o non calcolata'}.\n"
        f"QID usati per le metriche: {', '.join(qid_identified_list) if qid_identified_list else 'Nessuno'}.\n"  # MODIFICA: qid_identified -> qid_identified_list
        "Vedi sotto il template Markdown che devi compilare."
    )

    # Pre-formattazione delle stringhe complesse per il prompt_to_llm
    l_diversity_output_str = str(l_diversity_metrics) if l_diversity_metrics else 'N/D'
    qid_list_str_for_prompt = ", ".join(
        qid_identified_list) if qid_identified_list else "Nessun QID specifico identificato"

    prompt_to_llm = f"""
--- CONTESTO SINTETICO DEL FILE ANALIZZATO (NON RIPETERE DIRETTAMENTE QUESTI DATI GREZZI, MA USALI PER INFORMARE LA TUA ANALISI NELLE SEZIONI SOTTOSTANTI) ---
{context_summary_for_llm}
--- FINE CONTESTO SINTETICO ---

Il tuo compito è generare un **Report di Valutazione della Privacy e dei Rischi di Re-identificazione** per il file CSV descritto nel contesto.
Rispondi ESCLUSIVAMENTE in formato Markdown, seguendo scrupolosamente la struttura a **cinque sezioni** qui sotto.
Popola la sezione "Sintesi Esecutiva" e la "Tabella Riepilogativa" con i dati esatti forniti nel template (inclusi i valori di rischio e i dati delle tabelle).
Per le sezioni "Analisi della Sensibilità e Rischi di Re-identificazione", "Principali Rischi Legali..." e "Azioni Correttive...", fornisci la tua analisi esperta e concisa basata sul contesto.

Struttura della risposta Markdown richiesta:

{markdown_header.strip()}
**1. Sintesi Esecutiva e Livello di Rischio**
(Scrivi qui 1-2 frasi di riepilogo che riflettano il livello di rischio {risk_level_text} ({risk_pct}%) e una valutazione generale della pericolosità del file, basata sul CONTESTO fornito, includendo un breve accenno alle metriche di re-identificazione se significative.)

**2. Tabella Riepilogativa delle Principali Colonne con PII e Metriche di Rischio**
| Elemento Analizzato                    | Dettaglio / Valore Osservato | Soglia Consigliata | Esito / Nota        |
|----------------------------------------|------------------------------|--------------------|---------------------|
{table_metriche_md_str}
*Principali Colonne con PII Dirette Identificate (fino a 5):*
| Colonna PII Presunta                   | Metodo Suggerito Iniziale    | Descrizione Sensibilità (Valutazione Colonna) |
|----------------------------------------|------------------------------|-----------------------------------------------|
{pii_cols_table_md_str}

**3. Analisi della Sensibilità e Rischi di Re-identificazione**
(Spiega brevemente perché le colonne PII indicate sono sensibili. Poi, basandoti sulle metriche k-anonymity e l-diversity (valori: k_min={k_min_val}, records_singoli={records_singoli_val}, l-diversity per SA: {l_diversity_output_str}), commenta i rischi specifici di re-identificazione per questo dataset.)

**4. Principali Rischi Legali, Operativi e Reputazionali**
(Elenca come bullet point 2-4 rischi chiave derivanti sia dalla presenza di PII dirette sia dai rischi di re-identificazione.)

**5. Azioni Correttive e Raccomandazioni Prioritarie**
(Elenca come bullet point azioni concrete e gerarchizzate. Includi:
 - Applicazione dei metodi di anonimizzazione per le PII dirette.
 - Suggerimenti specifici per mitigare i rischi di k-anonymity e l-diversity (es. generalizzazione dei QID '{qid_list_str_for_prompt}', suppression, ecc.), specialmente se le soglie sono violate.
 - Valutazione della necessità di una DPIA.)

Sii analitico e orientato all'azione. Non aggiungere altre sezioni.
"""

    try:
        response = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un esperto consulente di data privacy e re-identificazione. Genera un report Markdown strutturato, analitico e con raccomandazioni chiare, basato sui dati forniti nel contesto e sul template di risposta."},
                {"role": "user", "content": prompt_to_llm.strip()}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        structured_markdown_comment = response.choices[0].message.content.strip()
        return structured_markdown_comment
    except Exception as e:
        error_message = (f"{markdown_header}"
                         f"**1. Sintesi Esecutiva e Livello di Rischio**\nImpossibile generare il commento finale a causa di un errore con l'LLM: {type(e).__name__}.\n"
                         f"Il file '{file_name}' presenta un livello di rischio stimato preliminare '{risk_level_text}' ({risk_pct}%), con {num_cols_with_pii} su {num_total_cols_analyzed} colonne testuali contenenti PII rilevate. Le metriche di re-identificazione calcolate erano: k-anonymity (min k = {risk_metrics_calculated.get('k_anonymity_min', 'N/D')}, record singoli = {risk_metrics_calculated.get('records_singoli', 'N/D')}). Si raccomanda cautela e revisione manuale approfondita.")
        return error_message


async def _inspect_column_async(
        col_name: str,
        sample_preview: str,
        model_api_id: str,
        methods_list_str: str
) -> Tuple[str, Dict[str, Any]]:
    # ... (codice come fornito precedentemente, corretto e con gestione errori migliorata)
    prompt_unificato_con_classificazione = f"""
Analizza la colonna CSV chiamata '{col_name}', i cui valori di esempio sono: "{sample_preview}".
Considera che i dati potrebbero essere usati per Machine Learning, quindi cerca di bilanciare privacy e utilità.

Fornisci la tua analisi ESCLUSIVAMENTE in formato JSON con la seguente struttura:
{{
  "is_sensitive_column": boolean, 
  "column_sensitivity_assessment": "stringa", 
  "pii_entities_found": [ {{ "text": "stringa_pii", "type": "tipo_pii", "reasoning_is_pii": "stringa" }} ],
  "suggested_anonymization_method": "stringa", 
  "method_reasoning": "stringa", 
  "column_privacy_category": "stringa" 
}}
Assicurati che "suggested_anonymization_method" sia uno da: {methods_list_str}.
E che "column_privacy_category" sia una SOLO tra: 'Identificatore Diretto', 'Quasi-Identificatore', 'Attributo Sensibile', 'Attributo Non Sensibile'.
"""
    default_error_response = {
        "error": "Errore generico", "is_sensitive_column": False,
        "column_sensitivity_assessment": "Analisi fallita.", "pii_entities_found": [],
        "suggested_anonymization_method": "nessuno",
        "method_reasoning": "Analisi LLM fallita o risposta non valida.",
        "column_privacy_category": "Errore Classificazione"
    }
    try:
        response = await openai.ChatCompletion.acreate(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un esperto di analisi dati, privacy e anonimizzazione. Rispondi sempre e solo con l'oggetto JSON richiesto, includendo tutti i campi specificati."},
                {"role": "user", "content": prompt_unificato_con_classificazione.strip()}
            ],
            temperature=0.1, max_tokens=1500
        )
        raw_llm_output = response.choices[0].message.content.strip()
        json_str_to_parse = None
        match_md_json = re.search(r"```json\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL | re.IGNORECASE)
        if match_md_json:
            json_str_to_parse = match_md_json.group(1)
        else:
            start_index = raw_llm_output.find('{');
            end_index = raw_llm_output.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index: json_str_to_parse = raw_llm_output[
                                                                                                      start_index: end_index + 1]

        if json_str_to_parse:
            parsed_data = json.loads(json_str_to_parse)
            parsed_data.setdefault("is_sensitive_column", False)
            parsed_data.setdefault("column_sensitivity_assessment", "Valutazione non fornita.")
            parsed_data.setdefault("pii_entities_found", [])
            parsed_data.setdefault("method_reasoning", "Motivazione non fornita.")
            parsed_data.setdefault("column_privacy_category", "Non Classificato")
            sugg_method = parsed_data.get("suggested_anonymization_method")
            if sugg_method not in AVAILABLE_ANON_METHODS:
                parsed_data[
                    "method_reasoning"] += f" (Avviso: Metodo LLM '{sugg_method}' non standard, usato fallback 'nessuno')"
                parsed_data["suggested_anonymization_method"] = "nessuno"
            return col_name, parsed_data
        else:
            error_detail = f"Nessun JSON valido per colonna '{col_name}'. Output: {raw_llm_output[:200]}..."
            default_error_response["error"] = error_detail;
            default_error_response["method_reasoning"] = error_detail
            return col_name, default_error_response
    except Exception as e:
        error_detail = f"Errore API/parsing per colonna '{col_name}': {type(e).__name__} - {e}"
        default_error_response["error"] = error_detail;
        default_error_response["method_reasoning"] = error_detail
        return col_name, default_error_response


async def analyze_and_anonymize_csv(
        df_text_columns: pd.DataFrame,
        model_api_id: str,
        sample_size_for_preview: int = 5,
        default_method_fallback: str = "mask",
        max_concurrent_requests: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ... (codice come fornito precedentemente, corretto)
    report_rows = []
    df_out = df_text_columns.copy()
    _init_lmstudio(model_api_id)

    methods_list_str = ", ".join([f"'{m}'" for m in AVAILABLE_ANON_METHODS])
    tasks = []
    for col_name in df_text_columns.columns:
        unique_vals = df_text_columns[col_name].dropna().astype(str).unique().tolist()
        sample_preview = "; ".join(unique_vals[:sample_size_for_preview]) + (
            "…" if len(unique_vals) > sample_size_for_preview else "")
        tasks.append(_inspect_column_async(col_name, sample_preview, model_api_id, methods_list_str))

    sem = asyncio.Semaphore(max_concurrent_requests)

    async def run_with_semaphore_limit(task):
        async with sem: return await task

    llm_results_for_columns = await asyncio.gather(*(run_with_semaphore_limit(t) for t in tasks))

    for col_name, llm_data in llm_results_for_columns:
        llm_found_pii_entities_flag = False
        actual_problem_description = f"**Valutazione Sensibilità Colonna '{col_name}':** Analisi fallita o risposta LLM non interpretabile."
        actual_suggested_method = default_method_fallback
        actual_method_reasoning = "Fallback a causa di errore o risposta LLM non valida."
        llm_column_category = "Errore Classificazione"

        if "error" not in llm_data:
            column_assessment = llm_data.get("column_sensitivity_assessment",
                                             "Nessuna valutazione contestuale fornita.")
            pii_entities = llm_data.get("pii_entities_found", [])

            if llm_data.get("is_sensitive_column", False):
                llm_found_pii_entities_flag = True
                if pii_entities:
                    detailed_entity_descriptions = []
                    for ent in pii_entities[:2]:
                        text_val = ent.get('text', 'N/A');
                        ent_type = ent.get('type', 'N/A')
                        pii_reason = ent.get('reasoning_is_pii', 'N/D')
                        detailed_entity_descriptions.append(
                            f"'{text_val}' (tipo: {ent_type}). Motivo sensibilità PII: *{pii_reason}*")
                    problem_details_str = "\n- ".join(
                        detailed_entity_descriptions) if detailed_entity_descriptions else "Nessun dettaglio specifico per le PII rilevate, ma la colonna è marcata come sensibile."
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}\n\n"
                        f"**PII Specifiche Rilevate ({len(pii_entities)} entità totali):**\n- {problem_details_str}")
                else:
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}\n\n"
                        f"L'LLM indica che la colonna è sensibile ma non sono state dettagliate PII specifiche negli esempi forniti.")
            else:
                actual_problem_description = (f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}")

            actual_suggested_method = llm_data.get("suggested_anonymization_method", default_method_fallback)
            actual_method_reasoning = llm_data.get("method_reasoning", "Nessuna motivazione fornita per il metodo.")
            llm_column_category = llm_data.get("column_privacy_category", "Non Classificato")
        else:
            actual_problem_description = f"**Errore Analisi Colonna '{col_name}':** {llm_data['error']}"

        current_col_data = df_out[col_name].copy()
        if actual_suggested_method == "hash":
            df_out[col_name] = current_col_data.astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(x) else x)
        elif actual_suggested_method == "mask":
            df_out[col_name] = current_col_data.astype(str).str.replace(r"[a-zA-Z0-9]", "*", regex=True)
        elif actual_suggested_method == "generalize_date":
            try:
                parsed_dates = pd.to_datetime(current_col_data, errors='coerce'); df_out[
                    col_name] = parsed_dates.dt.to_period("M").astype(str).replace('nan', pd.NA).replace('NaT', pd.NA)
            except Exception:
                df_out[col_name] = current_col_data
        elif actual_suggested_method == "truncate":
            df_out[col_name] = current_col_data.astype(str).str.slice(0, 10) + "..."

        unique_vals_for_report = df_text_columns[col_name].dropna().astype(str).unique().tolist()
        report_sample_preview = "; ".join(unique_vals_for_report[:sample_size_for_preview]) + (
            "…" if len(unique_vals_for_report) > sample_size_for_preview else "")

        report_rows.append({
            "Colonna": col_name, "Esempi": report_sample_preview,
            "Problematica": actual_problem_description, "MetodoSuggerito": actual_suggested_method,
            "Motivazione": actual_method_reasoning, "LLM_HaTrovatoEntitaPII": llm_found_pii_entities_flag,
            "CategoriaLLM": llm_column_category
        })
    report_df = pd.DataFrame(report_rows)
    return report_df, df_out


def get_llm_anonymization_method_suggestion(
        column_name: str,
        sample_preview: str,
        pii_summary: str,  # Es. "Rilevati: Nomi di persona, Email. Esempi: Mario Rossi, m.rossi@test.com"
        available_methods: list,
        model_api_id: str,
        for_ml_context: bool = True
) -> dict:
    """
    Chiama l'LLM per ottenere un suggerimento sul metodo di anonimizzazione.
    """
    ml_guidance = ("Considera che l'obiettivo è anonimizzare i dati preservando il più possibile la loro utilità per "
                   "analisi successive, come il Machine Learning. Scegli il metodo che offre il miglior compromesso "
                   "tra privacy e mantenimento del segnale informativo.") if for_ml_context else ""

    methods_list_str = ", ".join([f"'{m}'" for m in available_methods])

    prompt = f"""
Analisi per suggerimento metodo di anonimizzazione:
Colonna: "{column_name}"
Esempi di valori: "{sample_preview}"
PII Rilevate (tipi ed esempi): "{pii_summary}"

{ml_guidance}

Dalla seguente lista di metodi di anonimizzazione disponibili [{methods_list_str}], quale ritieni sia il più appropriato per questa colonna?
Fornisci anche una breve motivazione per la tua scelta.

Rispondi ESCLUSIVAMENTE in formato JSON con le seguenti chiavi:
- "suggested_method": stringa (uno dei metodi dalla lista: {methods_list_str})
- "method_reasoning": stringa (la tua motivazione)
"""
    try:

        response = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un esperto di anonimizzazione dati. Fornisci suggerimenti concisi e pertinenti in formato JSON."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2,  # Bassa per risposte più consistenti
            max_tokens=300  # Spazio sufficiente per JSON e motivazione
        )
        raw_llm_output = response.choices[0].message.content.strip()

        # Estrazione JSON (semplificata, adattabile da quella in generate_report)
        json_str_to_parse = None
        match_md_json = re.search(r"```json\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL)
        if match_md_json:
            json_str_to_parse = match_md_json.group(1)
        else:
            start_index = raw_llm_output.find('{')
            end_index = raw_llm_output.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str_to_parse = raw_llm_output[start_index: end_index + 1]

        if json_str_to_parse:
            parsed_json = json.loads(json_str_to_parse)
            # Validazione dell'output
            if "suggested_method" in parsed_json and "method_reasoning" in parsed_json:
                if parsed_json["suggested_method"] not in available_methods:
                    # Se l'LLM suggerisce un metodo non valido, usa un fallback e segnalalo
                    parsed_json[
                        "method_reasoning"] += f" [Avviso: Metodo '{parsed_json['suggested_method']}' non standard, usato fallback]"
                    parsed_json["suggested_method"] = available_methods[
                        0] if available_methods else "nessuno"  # o un default più sicuro
                return parsed_json
            else:
                return {"suggested_method": "error",
                        "method_reasoning": "JSON dall'LLM incompleto (mancano chiavi attese)."}
        else:
            return {"suggested_method": "error",
                    "method_reasoning": f"Nessun JSON valido trovato nella risposta dell'LLM: {raw_llm_output}"}

    except Exception as e:
        return {"suggested_method": "error",
                "method_reasoning": f"Errore durante la chiamata LLM per suggerimento metodo: {str(e)}"}


# modules/analyisis_cvs.py
# ... (import esistenti) ...
from modules.generazione_testo import _init_lmstudio  # Assicurati che sia importato


async def _inspect_column_async(
        col_name: str,
        sample_preview: str,
        model_api_id: str,
        methods_list_str: str  # Stringa formattata dei metodi disponibili
) -> Tuple[str, Dict[str, Any]]:
    """
    Funzione helper asincrona per analizzare una singola colonna con l'LLM.
    Chiede classificazione, sensibilità, PII, metodo suggerito e motivazioni.
    """
    prompt_unificato_con_classificazione = f"""
Analizza la colonna CSV chiamata '{col_name}', i cui valori di esempio sono: "{sample_preview}".
Considera che i dati potrebbero essere usati per Machine Learning, quindi cerca di bilanciare privacy e utilità.

Fornisci la tua analisi ESCLUSIVAMENTE in formato JSON con la seguente struttura:
{{
  "is_sensitive_column": boolean, // La colonna nel suo complesso contiene dati sensibili o PII basati sugli esempi?
  "column_sensitivity_assessment": "stringa", // Spiegazione concisa del perché la colonna, basata sugli esempi, è (o non è) sensibile nel suo contesto.
  "pii_entities_found": [ // Lista delle PII specifiche trovate NEGLI ESEMPI FORNITI. Lascia vuota se nessuna PII specifica è identificabile negli esempi.
    {{
      "text": "stringa_pii", // Il testo della PII rilevata
      "type": "tipo_pii", // Il tipo di PII (es. PERSON, DATE, EMAIL, MEDICATION, ID_NUMBER)
      "reasoning_is_pii": "stringa" // Breve motivo per cui questo specifico testo è una PII o è sensibile
    }}
  ],
  "suggested_anonymization_method": "stringa", // Uno da: {methods_list_str}. Scegli 'nessuno' se is_sensitive_column è false o se non è necessario anonimizzare per ML.
  "method_reasoning": "stringa", // Motivazione per il metodo di anonimizzazione suggerito.
  "column_privacy_category": "stringa" // Classifica la colonna come uno SOLO tra: 'Identificatore Diretto', 'Quasi-Identificatore', 'Attributo Sensibile', 'Attributo Non Sensibile'
}}
Assicurati che "suggested_anonymization_method" sia uno da: {methods_list_str}.
E che "column_privacy_category" sia una delle quattro categorie specificate.
"""
    default_error_response = {
        "error": "Errore generico",
        "is_sensitive_column": False,  # Default a non sensibile in caso di errore grave
        "column_sensitivity_assessment": "Analisi fallita.",
        "pii_entities_found": [],
        "suggested_anonymization_method": "nessuno",  # Default sicuro
        "method_reasoning": "Analisi LLM fallita o risposta non valida.",
        "column_privacy_category": "Errore Classificazione"
    }
    try:
        response = await openai.ChatCompletion.acreate(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un esperto di analisi dati, privacy e anonimizzazione. Rispondi sempre e solo con l'oggetto JSON richiesto, includendo tutti i campi specificati."},
                {"role": "user", "content": prompt_unificato_con_classificazione.strip()}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        raw_llm_output = response.choices[0].message.content.strip()

        json_str_to_parse = None
        match_md_json = re.search(r"```json\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL | re.IGNORECASE)
        if match_md_json:
            json_str_to_parse = match_md_json.group(1)
        else:
            start_index = raw_llm_output.find('{')
            end_index = raw_llm_output.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str_to_parse = raw_llm_output[start_index: end_index + 1]

        if json_str_to_parse:
            parsed_data = json.loads(json_str_to_parse)
            # Assicura che i campi chiave siano presenti con valori di default se mancano
            parsed_data.setdefault("is_sensitive_column", False)
            parsed_data.setdefault("column_sensitivity_assessment", "Valutazione non fornita.")
            parsed_data.setdefault("pii_entities_found", [])
            parsed_data.setdefault("method_reasoning", "Motivazione non fornita.")
            parsed_data.setdefault("column_privacy_category", "Non Classificato")

            # Valida suggested_anonymization_method
            sugg_method = parsed_data.get("suggested_anonymization_method")
            if sugg_method not in AVAILABLE_ANON_METHODS:
                parsed_data[
                    "method_reasoning"] += f" (Avviso: Metodo LLM '{sugg_method}' non standard, usato fallback 'nessuno')"
                parsed_data["suggested_anonymization_method"] = "nessuno"  # Fallback sicuro

            return col_name, parsed_data
        else:
            error_detail = f"Nessun JSON valido per colonna '{col_name}'. Output: {raw_llm_output[:200]}..."
            default_error_response["error"] = error_detail
            default_error_response["method_reasoning"] = error_detail
            return col_name, default_error_response

    except Exception as e:
        error_detail = f"Errore API/parsing per colonna '{col_name}': {type(e).__name__} - {e}"
        default_error_response["error"] = error_detail
        default_error_response["method_reasoning"] = error_detail
        return col_name, default_error_response


async def analyze_and_anonymize_csv(  # Nome mantenuto per coerenza con l'import dell'utente
        df_text_columns: pd.DataFrame,
        model_api_id: str,
        sample_size_for_preview: int = 5,
        default_method_fallback: str = "mask",
        max_concurrent_requests: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_rows = []
    df_out = df_text_columns.copy()
    _init_lmstudio(model_api_id)

    methods_list_str = ", ".join([f"'{m}'" for m in AVAILABLE_ANON_METHODS])
    tasks = []
    for col_name in df_text_columns.columns:
        unique_vals = df_text_columns[col_name].dropna().astype(str).unique().tolist()
        sample_preview = "; ".join(unique_vals[:sample_size_for_preview]) + (
            "…" if len(unique_vals) > sample_size_for_preview else "")
        tasks.append(_inspect_column_async(col_name, sample_preview, model_api_id, methods_list_str))

    sem = asyncio.Semaphore(max_concurrent_requests)

    async def run_with_semaphore_limit(task):
        async with sem:
            return await task

    llm_results_for_columns = await asyncio.gather(*(run_with_semaphore_limit(t) for t in tasks))

    for col_name, llm_data in llm_results_for_columns:
        llm_found_pii_entities_flag = False
        actual_problem_description = f"**Valutazione Sensibilità Colonna '{col_name}':** Analisi fallita o risposta LLM non interpretabile."
        actual_suggested_method = default_method_fallback
        actual_method_reasoning = "Fallback a causa di errore o risposta LLM non valida."
        llm_column_category = "Errore Classificazione"

        if "error" not in llm_data:  # Se non c'è stato un errore grave nella chiamata o parsing iniziale
            column_assessment = llm_data.get("column_sensitivity_assessment",
                                             "Nessuna valutazione contestuale fornita.")
            pii_entities = llm_data.get("pii_entities_found", [])

            if llm_data.get("is_sensitive_column", False):  # Basati su quanto dice l'LLM per la colonna
                llm_found_pii_entities_flag = True  # Anche se pii_entities fosse vuota, ma l'LLM dice che la colonna è sensibile
                if pii_entities:  # Ci sono entità PII specifiche
                    detailed_entity_descriptions = []
                    for ent in pii_entities[:2]:
                        text_val = ent.get('text', 'N/A')
                        ent_type = ent.get('type', 'N/A')
                        pii_reason = ent.get('reasoning_is_pii', 'N/D')
                        detailed_entity_descriptions.append(
                            f"'{text_val}' (tipo: {ent_type}). Motivo sensibilità PII: *{pii_reason}*"
                        )
                    problem_details_str = "\n- ".join(
                        detailed_entity_descriptions) if detailed_entity_descriptions else "Nessun dettaglio specifico per le PII rilevate, ma la colonna è marcata come sensibile."
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}\n\n"
                        f"**PII Specifiche Rilevate ({len(pii_entities)} entità totali):**\n- {problem_details_str}"
                    )
                else:  # is_sensitive_column è True, ma pii_entities è vuota
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}\n\n"
                        f"L'LLM indica che la colonna è sensibile ma non sono state dettagliate PII specifiche negli esempi forniti."
                    )
            else:  # is_sensitive_column è False
                actual_problem_description = (
                    f"**Valutazione Sensibilità Colonna '{col_name}':** {column_assessment}"
                )

            actual_suggested_method = llm_data.get("suggested_anonymization_method", default_method_fallback)
            actual_method_reasoning = llm_data.get("method_reasoning", "Nessuna motivazione fornita per il metodo.")
            llm_column_category = llm_data.get("column_privacy_category", "Non Classificato")
        else:  # C'è stata una chiave "error" nel dict llm_data
            actual_problem_description = f"**Errore Analisi Colonna '{col_name}':** {llm_data['error']}"
            # actual_suggested_method e actual_method_reasoning rimangono i fallback

        # Anonimizzazione preliminare di df_out
        current_col_data = df_out[col_name].copy()
        if actual_suggested_method == "hash":
            df_out[col_name] = current_col_data.astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(x) else x)
        elif actual_suggested_method == "mask":
            df_out[col_name] = current_col_data.astype(str).str.replace(r"[a-zA-Z0-9]", "*", regex=True)
        elif actual_suggested_method == "generalize_date":
            try:
                parsed_dates = pd.to_datetime(current_col_data, errors='coerce')
                df_out[col_name] = parsed_dates.dt.to_period("M").astype(str).replace('nan', pd.NA).replace('NaT',
                                                                                                            pd.NA)
            except Exception:
                df_out[col_name] = current_col_data
        elif actual_suggested_method == "truncate":
            df_out[col_name] = current_col_data.astype(str).str.slice(0, 10) + "..."

        unique_vals_for_report = df_text_columns[col_name].dropna().astype(str).unique().tolist()
        report_sample_preview = "; ".join(unique_vals_for_report[:sample_size_for_preview]) + (
            "…" if len(unique_vals_for_report) > sample_size_for_preview else "")

        report_rows.append({
            "Colonna": col_name,
            "Esempi": report_sample_preview,
            "Problematica": actual_problem_description,
            "MetodoSuggerito": actual_suggested_method,
            "Motivazione": actual_method_reasoning,
            "LLM_HaTrovatoEntitaPII": llm_found_pii_entities_flag,
            "CategoriaLLM": llm_column_category
        })

    report_df = pd.DataFrame(report_rows)
    return report_df, df_out