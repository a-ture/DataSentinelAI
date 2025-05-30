# modules/analyisis_cvs.py
import pandas as pd
import hashlib
from typing import Dict, Any, Tuple
from modules.generazione_testo import generate_report  # Assicurati che sia importato

# All'inizio di modules/analyisis_cvs.py, aggiungi/verifica questi import:
import json
import openai  # Già usato implicitamente tramite generazione_testo
import re  # Per l'estrazione del JSON dalla risposta LLM


# Aggiungi questa funzione, ad esempio, in modules/generazione_testo.py (o dove hai get_llm_anonymization_method_suggestion)
# Assicurati che openai, json, re siano importati in quel file.

def get_llm_overall_csv_comment(
        column_analysis_report: pd.DataFrame,  # Il DataFrame prodotto da analyze_and_anonymize_csv
        model_api_id: str,
        file_name: str = "Il file CSV analizzato"
) -> str:
    """
    Genera un commento generale sulla sensibilità dell'intero file CSV
    basandosi sul report di analisi delle singole colonne.
    """
    if column_analysis_report.empty:
        return "Nessun report di analisi disponibile per le colonne del CSV."

    num_total_cols_analyzed = len(column_analysis_report)

    # Filtra per colonne che hanno PII o richiedono azione
    problematic_cols = column_analysis_report[
        (column_analysis_report["Problematica"].str.contains("PII rilevate", case=False, na=False)) &
        (column_analysis_report["MetodoSuggerito"] != "nessuno")
        ]
    num_problematic_cols = len(problematic_cols)

    if num_problematic_cols == 0:
        return (f"Analisi completata per '{file_name}' ({num_total_cols_analyzed} colonne testuali). "
                "Non sembrano esserci PII significative che richiedono anonimizzazione immediata secondo l'LLM.")

    intro = (f"Il file '{file_name}' è stato analizzato ({num_total_cols_analyzed} colonne testuali) "
             f"e sono state identificate problematiche di sensibilità in {num_problematic_cols} colonna/e.")

    details_for_prompt = [intro]
    details_for_prompt.append("\nDettagli dalle colonne più rilevanti:")
    for idx, row in problematic_cols.head(min(3, num_problematic_cols)).iterrows():  # Dettagli per max 3 colonne
        col = row["Colonna"]
        problem = row["Problematica"].split('\n\n')[0]  # Prendi la parte della valutazione generale della colonna
        method = row["MetodoSuggerito"]
        details_for_prompt.append(f"- Colonna '{col}': {problem} (Metodo suggerito: {method})")

    # Conteggio dei metodi suggeriti (escluso 'nessuno')
    suggested_methods_counts = problematic_cols[problematic_cols["MetodoSuggerito"] != "nessuno"][
        "MetodoSuggerito"].value_counts()
    if not suggested_methods_counts.empty:
        details_for_prompt.append("\nMetodi di anonimizzazione suggeriti con più frequenza:")
        for method, count in suggested_methods_counts.items():
            details_for_prompt.append(f"  - {method}: {count} volta/e")

    aggregated_report_summary = "\n".join(details_for_prompt)

    prompt_to_llm = f"""
Basandoti sul seguente riepilogo dell'analisi di un file CSV:
---
{aggregated_report_summary}
---

Fornisci un commento generale conciso (1-2 paragrafi) sulla sensibilità complessiva di questo file CSV. 
Considera la natura e la frequenza delle PII rilevate e dei metodi di anonimizzazione suggeriti.
Quali sono le implicazioni generali per la privacy e la gestione di questo dataset?
"""
    try:
        # Assicurati che _init_lmstudio sia stato chiamato se necessario (es. da generate_report)
        # o che il server LLM sia già attivo e il modello caricato.
        response = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un consulente per la privacy dei dati. Fornisci un commento finale sulla sensibilità di un dataset CSV."},
                {"role": "user", "content": prompt_to_llm.strip()}
            ],
            temperature=0.4,
            max_tokens=400
        )
        comment = response.choices[0].message.content.strip()
        return comment
    except Exception as e:
        return f"Errore durante la generazione del commento generale sul file: {str(e)}"

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
    ml_guidance = "Considera che l'obiettivo è anonimizzare i dati preservando il più possibile la loro utilità per analisi successive, come il Machine Learning. Scegli il metodo che offre il miglior compromesso tra privacy e mantenimento del segnale informativo." if for_ml_context else ""

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
        # Nota: _init_lmstudio da generazione_testo.py viene chiamato da generate_report.
        # Se questa è la prima chiamata LLM per questo model_api_id o se il server non è attivo,
        # potrebbe essere necessario un meccanismo simile a _init_lmstudio qui,
        # oppure assicurarsi che generate_report sia stato chiamato prima.
        # Per ora, assumiamo che il server sia pronto se generate_report è stato chiamato.

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
# ... (import esistenti e funzione get_llm_anonymization_method_suggestion come definite precedentemente) ...

def analyze_and_anonymize_csv(
        df: pd.DataFrame,
        model_api_id: str,
        sample_size: int = 50,
        default_method_fallback: str = "mask"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_rows = []
    df_out = df.copy()

    AVAILABLE_ANON_METHODS = ["hash", "mask", "generalize_date", "truncate", "nessuno"]

    for col in df.columns:  # Itera sulle colonne del DataFrame fornito
        # ... (definizione di vals e sample_preview come prima) ...
        vals = df[col].dropna().astype(str).unique().tolist()[:sample_size]
        sample_preview = "; ".join(vals[:5]) + ("…" if len(vals) > 5 else "")

        prompt_for_pii_detection = f"""
Analizza il seguente testo per identificare informazioni personali identificabili (PII).
Il testo proviene da una colonna CSV chiamata '{col}' e questi sono alcuni esempi di valori:
{sample_preview}
Restituisci un JSON con chiavi "found" (boolean), "entities" (lista di PII trovate con type, text, context, reasoning), e "summary" (un breve sommario che includa una valutazione generale della sensibilità del campione di dati fornito).
"""
        llm_resp_pii_detection = generate_report(prompt_for_pii_detection.strip(), model_api_id)

        actual_problem_description = ""
        actual_suggested_method = default_method_fallback
        actual_method_reasoning = "In attesa di analisi e suggerimento LLM."

        if isinstance(llm_resp_pii_detection, dict):
            pii_found_by_llm = llm_resp_pii_detection.get("found", False)
            entities = llm_resp_pii_detection.get("entities", [])
            summary_pii_detection = llm_resp_pii_detection.get("summary",
                                                               "")  # Questo è il sommario contestuale per la colonna

            # Costruisci la descrizione del problema
            column_sensitivity_context = summary_pii_detection if summary_pii_detection else "Nessun sommario contestuale sulla sensibilità della colonna fornito dall'LLM."

            if pii_found_by_llm and entities:
                detailed_entity_descriptions = []
                for ent in entities[:2]:
                    text_val = ent.get('text', 'N/A')
                    ent_type = ent.get('type', 'N/A')
                    pii_sensitivity_reasoning = ent.get('reasoning', 'Nessuna motivazione specifica fornita.')
                    detailed_entity_descriptions.append(
                        f"'{text_val}' (tipo: {ent_type}). Motivo sensibilità PII: *{pii_sensitivity_reasoning}*"
                    )

                problem_details_str = "\n- ".join(
                    detailed_entity_descriptions) if detailed_entity_descriptions else "Nessun dettaglio specifico sulle PII rilevate."

                actual_problem_description = (
                    f"**Valutazione Complessiva Sensibilità Colonna '{col}':** {column_sensitivity_context}\n\n"
                    f"**PII Specifiche Rilevate ({len(entities)} entità totali):**\n- {problem_details_str}"
                )

                # Fase 2: Chiamata LLM per suggerimento metodo (come prima)
                simple_entity_examples_list = [f"'{ent.get('text', 'N/A')[:50]}' (tipo: {ent.get('type', 'N/A')})" for
                                               ent in entities[:2]]
                problem_entities_preview = "; ".join(
                    simple_entity_examples_list) if simple_entity_examples_list else "Nessun esempio specifico di entità PII disponibile"
                pii_types_str = ", ".join(list(set(str(ent.get("type", "")).lower() for ent in entities)))
                pii_summary_for_method_prompt = f"Tipi di PII: {pii_types_str}. Esempi: {problem_entities_preview}."

                suggestion_result = get_llm_anonymization_method_suggestion(
                    # ... (parametri come prima) ...
                    column_name=col,
                    sample_preview=sample_preview,
                    pii_summary=pii_summary_for_method_prompt,
                    available_methods=AVAILABLE_ANON_METHODS,
                    model_api_id=model_api_id,
                    for_ml_context=True
                )
                actual_suggested_method = suggestion_result.get("suggested_method", default_method_fallback)
                actual_method_reasoning = suggestion_result.get("method_reasoning",
                                                                "Errore nel ricevere la motivazione dall'LLM per il metodo.")
                if actual_suggested_method == "error":
                    actual_suggested_method = default_method_fallback

            elif pii_found_by_llm and not entities:
                actual_problem_description = f"**Valutazione Complessiva Sensibilità Colonna '{col}':** {column_sensitivity_context}\n\nL'LLM ha indicato PII generiche, ma senza fornire dettagli specifici."
                actual_suggested_method = default_method_fallback
                actual_method_reasoning = f"Applicato fallback: {default_method_fallback}."

            elif not pii_found_by_llm:
                actual_problem_description = f"**Valutazione Complessiva Sensibilità Colonna '{col}':** L'analisi LLM non ha rilevato PII specifiche in questi esempi. {summary_pii_detection if summary_pii_detection else ''}"
                actual_suggested_method = "nessuno"
                actual_method_reasoning = "Nessuna PII specifica trovata."

            if "Error:" in summary_pii_detection and not actual_problem_description.startswith(
                    f"**Valutazione Complessiva Sensibilità Colonna '{col}':**"):
                actual_problem_description = f"**Valutazione Complessiva Sensibilità Colonna '{col}':** Errore da rilevamento PII: {summary_pii_detection}"
                actual_method_reasoning = "Dettagli errore nel sommario. Applicato fallback."
                actual_suggested_method = default_method_fallback
        else:
            actual_problem_description = f"**Valutazione Complessiva Sensibilità Colonna '{col}':** Risposta non valida da rilevamento PII."
            actual_method_reasoning = "Impossibile determinare PII, applicato fallback."
            actual_suggested_method = default_method_fallback

        # ... (Logica di anonimizzazione per df_out) ...
        current_col_data = df_out[col].copy()
        if actual_suggested_method == "hash":
            df_out[col] = current_col_data.astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(x) else x)
        elif actual_suggested_method == "mask":
            df_out[col] = current_col_data.astype(str).str.replace(r"[a-zA-Z0-9]", "*", regex=True)
        elif actual_suggested_method == "generalize_date":
            try:
                parsed_dates = pd.to_datetime(current_col_data, errors='coerce')
                df_out[col] = parsed_dates.dt.to_period("M").astype(str).replace('NaT', pd.NA)
            except Exception:
                df_out[col] = current_col_data
        elif actual_suggested_method == "truncate":
            df_out[col] = current_col_data.astype(str).str.slice(0, 10) + "..."

        report_rows.append({
            "Colonna": col,
            "Esempi": sample_preview,
            "Problematica": actual_problem_description,  # Ora include la valutazione contestuale + dettagli PII
            "MetodoSuggerito": actual_suggested_method,
            "Motivazione": actual_method_reasoning
        })

    report_df = pd.DataFrame(report_rows)
    return report_df, df_out