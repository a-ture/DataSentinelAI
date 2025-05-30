import pandas as pd
import hashlib
import json
import openai
import re
from typing import Tuple
from modules.generazione_testo import _init_lmstudio


# Nel file dove hai definito get_llm_overall_csv_comment
# (es. modules/generazione_testo.py o modules/analyisis_cvs.py)
# Assicurati che pandas, openai, json, re siano importati in quel file.

# Nel file dove hai definito get_llm_overall_csv_comment
# (es. modules/generazione_testo.py o modules/analyisis_cvs.py)
# Assicurati che pandas, openai, json, re siano importati in quel file.

# Nel file dove hai definito get_llm_overall_csv_comment
# (es. modules/generazione_testo.py o modules/analyisis_cvs.py)
import pandas as pd
import openai
import json
import re


# Nel file dove hai definito get_llm_overall_csv_comment
# (es. modules/generazione_testo.py o modules/analyisis_cvs.py)
# Assicurati che pandas, openai, json, re siano importati in quel file.

# Nel file dove hai definito get_llm_overall_csv_comment
# (es. modules/generazione_testo.py o modules/analyisis_cvs.py)
# Assicurati che pandas, openai, json, re siano importati in quel file.

def get_llm_overall_csv_comment(
        column_analysis_report: pd.DataFrame,
        model_api_id: str,
        file_name: str = "Il file CSV analizzato"
) -> str:
    if column_analysis_report.empty:
        return "Nessun report di analisi disponibile per le colonne del CSV."

    # --- 1. CALCOLA PRIVACY RISK SCORE ---
    num_total_cols_analyzed = len(column_analysis_report)
    pii_mask = pd.Series([False] * num_total_cols_analyzed, index=column_analysis_report.index)  # Default a False

    if "LLM_HaTrovatoEntitaPII" in column_analysis_report.columns:
        pii_mask = column_analysis_report["LLM_HaTrovatoEntitaPII"] == True
    else:  # Fallback robusto
        pii_mask = column_analysis_report["Problematica"].str.contains(
            r"PII.*Rilevat", case=False, na=False, regex=True
        )

    cols_with_pii_detected_df = column_analysis_report[pii_mask]
    num_cols_with_pii = len(cols_with_pii_detected_df)

    risk_pct = 0.0
    if num_total_cols_analyzed > 0:
        risk_pct = round((num_cols_with_pii / num_total_cols_analyzed) * 100, 1)

    if risk_pct == 0:
        risk_level_emoji = "🟢"
        risk_level_text = "Basso"
    elif risk_pct < 30:
        risk_level_emoji = "🟡"
        risk_level_text = "Medio"
    else:
        risk_level_emoji = "🛑"
        risk_level_text = "Alto"

    risk_score_full_text = f"{risk_level_emoji} {risk_level_text} ({risk_pct} % delle colonne testuali analizzate contengono PII)"

    if num_cols_with_pii == 0:
        return (f"### Valutazione Privacy – {risk_score_full_text}\n\n"
                f"**Sintesi**\nL'analisi per '{file_name}' ({num_total_cols_analyzed} colonne) "
                "indica che l’LLM **non ha rilevato PII specifiche**. Il dataset sembra presentare un rischio di privacy basso.")

    # --- 2. COSTRUISCI MINI-TABELLA RIEPILOGATIVA (DATI PER L'LLM) ---
    # Seleziona le colonne da mostrare nella tabella (es. quelle con PII e metodo non "nessuno", o le più rilevanti)
    # display_cols_for_table_df = cols_with_pii_detected_df[cols_with_pii_detected_df["MetodoSuggerito"] != "nessuno"].head(5)
    # if display_cols_for_table_df.empty:
    display_cols_for_table_df = cols_with_pii_detected_df.head(5)  # Mostra fino a 5 colonne con PII

    table_data_for_llm = []
    for idx, row in display_cols_for_table_df.iterrows():
        col_name = row['Colonna']
        method_sugg = row['MetodoSuggerito']

        problem_full_text = row["Problematica"]
        sensitivity_reason = problem_full_text.split('\n\n')[0]  # Prendi la prima parte (valutazione generale colonna)
        if sensitivity_reason.startswith(f"**Valutazione Complessiva Sensibilità Colonna '{col_name}':**"):
            sensitivity_reason = sensitivity_reason.replace(
                f"**Valutazione Complessiva Sensibilità Colonna '{col_name}':**", "").strip()
        else:  # Fallback
            pii_details_match = re.search(r"\*\*PII Specifiche Rilevate.*?\*\*:\s*(.*)", problem_full_text, re.DOTALL)
            if pii_details_match:
                sensitivity_reason = pii_details_match.group(1).split('\n-')[1][
                                     :100].strip()  # Prendi il primo dettaglio PII
            else:
                sensitivity_reason = "Informazione sensibile rilevata"

        sensitivity_reason_short = (sensitivity_reason[:70] + '...') if len(
            sensitivity_reason) > 73 else sensitivity_reason
        table_data_for_llm.append({
            "Colonna": col_name,
            "MetodoConsigliato": method_sugg,
            "PercheSensibile": sensitivity_reason_short.replace('|', ' ')  # Evita problemi con Markdown
        })

    # Prepara il contesto informativo per l'LLM
    context_for_llm_prompt = [
        f"File Analizzato: {file_name}",
        f"Colonne Testuali Totali Analizzate: {num_total_cols_analyzed}",
        f"Colonne con PII Rilevate: {num_cols_with_pii}",
        f"Percentuale Colonne con PII: {risk_pct}%",
        f"Livello di Rischio Privacy Stimato: {risk_level_text}",
        "\nDettaglio delle principali colonne con PII (fino a 5):"
    ]
    if not table_data_for_llm:
        context_for_llm_prompt.append(
            "- Nessuna colonna specifica da dettagliare ulteriormente oltre le statistiche generali.")
    else:
        for item in table_data_for_llm:
            context_for_llm_prompt.append(
                f"  - Colonna: '{item['Colonna']}', Metodo Suggerito: '{item['MetodoConsigliato']}', Motivo Sensibilità Principale: \"{item['PercheSensibile']}\""
            )

    aggregated_report_summary_for_llm = "\n".join(context_for_llm_prompt)

    # --- 3. RAFFORZA IL PROMPT ALL'LLM ---
    prompt_to_llm = f"""
Basandoti sul seguente contesto derivato dall'analisi di un file CSV:
--- CONTESTO DEL FILE ANALIZZATO ---
{aggregated_report_summary_for_llm}
--- FINE CONTESTO ---

Il tuo compito è generare un report di valutazione della privacy per questo file.
Rispondi ESCLUSIVAMENTE in formato Markdown, seguendo scrupolosamente la struttura a quattro sezioni qui sotto. Usa le informazioni del contesto fornito dove appropriato (es. per il punteggio di rischio e la tabella). Per i rischi e le azioni, fornisci la tua analisi esperta.

Struttura della risposta Markdown richiesta:

### Valutazione Privacy – {risk_level_emoji} {risk_level_text} ({risk_pct} %)

**Sintesi**
(Scrivi qui 1-2 frasi di riepilogo che includano il punteggio di rischio e una valutazione generale della pericolosità del file basata sul contesto fornito.)

**Tabella Riepilogativa delle Colonne Principali con PII**
| Colonna | Metodo Consigliato | Perché Sensibile (Valutazione Colonna) |
|---------|--------------------|------------------------------------------|
{''.join([f"| {item['Colonna']} | {item['MetodoConsigliato']} | {item['PercheSensibile']} |" for item in table_data_for_llm]) if table_data_for_llm else "| Nessuna colonna specifica da elencare qui. | - | - |"}

**Perché le colonne sopra indicate (e altre simili nel dataset) sono sensibili?**
(Per ogni colonna listata nella tabella o per gruppi di colonne simili menzionate nel contesto, fornisci un bullet point conciso - max 1 riga per bullet - che spieghi la natura della loro sensibilità. Ad Esempio: "- La colonna 'NomeCompleto' è sensibile perché identifica direttamente una persona.")

**Rischi Legali e Operativi Principali**
(Elenca qui come bullet point un massimo di 3 rischi principali, legali o operativi, derivanti dalla presenza delle PII identificate nel contesto. Esempio: "- Possibile violazione art. 32 GDPR per mancanza di pseudonimizzazione adeguata.")

**Azioni Immediate Raccomandate**
(Elenca qui come bullet point azioni concrete e prioritarie da intraprendere per mitigare i rischi. Esempio: "- Applicare i metodi di anonimizzazione suggeriti o equivalenti per tutte le colonne con PII.")

Non aggiungere altre sezioni o testo al di fuori di questa struttura.
"""
    try:
        response = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": "Sei un esperto consulente di data privacy. Il tuo compito è generare un report Markdown strutturato e informativo basato sui dati di analisi di un file CSV che ti vengono forniti."},
                {"role": "user", "content": prompt_to_llm.strip()}
            ],
            temperature=0.1,  # Molto bassa per seguire la struttura
            max_tokens=1200  # Aumentato per permettere un output Markdown più completo
        )
        structured_markdown_comment = response.choices[0].message.content.strip()

        # Piccolo controllo per assicurarsi che l'output inizi come atteso (opzionale)
        if not structured_markdown_comment.startswith("### Valutazione Privacy"):
            # Fallback o aggiunta dell'intestazione se mancante, anche se l'LLM dovrebbe produrla
            header_fallback = f"### Valutazione Privacy – {risk_score_full_text}\n\n"
            # Potrebbe essere necessario un logging qui se l'LLM non segue la struttura
            # print("WARN: L'LLM non ha prodotto l'intestazione attesa, aggiunta di fallback.")
            if risk_pct == 0 and num_cols_with_pii == 0:  # Se il file è effettivamente a basso rischio
                return (f"### Valutazione Privacy – {risk_score_full_text}\n\n"
                        f"**Sintesi**\nL'analisi per '{file_name}' ({num_total_cols_analyzed} colonne) "
                        "indica che l’LLM **non ha rilevato PII specifiche**. Il dataset sembra presentare un rischio di privacy basso e non richiede azioni di anonimizzazione immediate basate su questa analisi.")
            # Se ci sono PII, ma l'LLM non ha formattato bene, almeno ritorna il contesto e chiedi all'LLM di riprovare o mostra un errore.
            # Per ora, proviamo a restituire quello che l'LLM dà, sperando sia vicino.
            # O si potrebbe tentare di forzare l'intestazione:
            # if not structured_markdown_comment.strip().startswith("###"):
            #    structured_markdown_comment = header_fallback + structured_markdown_comment

        return structured_markdown_comment

    except Exception as e:
        # print(f"DEBUG: Errore in get_llm_overall_csv_comment: {e}") # Per debug
        error_message = (f"### Valutazione Privacy – Errore\n\n"
                         f"**Sintesi**\nImpossibile generare il commento finale a causa di un errore con l'LLM: {type(e).__name__}.\n"
                         f"Dati preliminari: Rischio stimato '{risk_level_text}' ({risk_pct}%), {num_cols_with_pii}/{num_total_cols_analyzed} colonne con PII rilevate per '{file_name}'.")
        return error_message

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


def analyze_and_anonymize_csv(
        df: pd.DataFrame,
        model_api_id: str,
        sample_size: int = 50,
        default_method_fallback: str = "mask"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_rows = []
    df_out = df.copy()
    _init_lmstudio(model_api_id)

    AVAILABLE_ANON_METHODS = ["hash", "mask", "generalize_date", "truncate", "nessuno"]
    methods_list_str = ", ".join([f"'{m}'" for m in AVAILABLE_ANON_METHODS])

    for col in df.columns:
        # ... (vals, sample_preview, prompt_unificato come nella versione precedente) ...
        vals = df[col].dropna().astype(str).unique().tolist()[:sample_size]
        sample_preview = "; ".join(vals[:5]) + ("…" if len(vals) > 5 else "")

        prompt_unificato = f"""
Analizza la colonna CSV chiamata '{col}', i cui valori di esempio sono: "{sample_preview}".
Considera che i dati potrebbero essere usati per Machine Learning, quindi cerca di bilanciare privacy e utilità.
Fornisci la tua analisi ESCLUSIVAMENTE in formato JSON con la seguente struttura:
{{
  "is_sensitive_column": boolean,
  "column_sensitivity_assessment": "stringa",
  "pii_entities_found": [ {{ "text": "stringa_pii", "type": "tipo_pii", "reasoning_is_pii": "stringa" }} ],
  "suggested_anonymization_method": "stringa", 
  "method_reasoning": "stringa"
}}
"""  # Ho semplificato un po' la struttura JSON nel commento per chiarezza, ma il prompt è quello

        actual_problem_description = f"Analisi per la colonna '{col}' non ancora completata o fallita."
        actual_suggested_method = default_method_fallback
        actual_method_reasoning = "Fallback a causa di errore o risposta LLM non interpretabile."
        llm_found_pii_entities_flag = False  # NUOVA FLAG, inizializzata a False

        try:
            response = openai.ChatCompletion.create(
                model=model_api_id,
                messages=[
                    {"role": "system",
                     "content": "Sei un esperto di analisi dati e privacy. Rispondi sempre e solo con l'oggetto JSON richiesto."},
                    {"role": "user", "content": prompt_unificato.strip()}
                ],
                temperature=0.2,
                max_tokens=1024
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
                llm_data = json.loads(json_str_to_parse)
                column_assessment = llm_data.get("column_sensitivity_assessment",
                                                 "Nessuna valutazione contestuale fornita.")
                pii_entities = llm_data.get("pii_entities_found", [])

                # Imposta la flag se l'LLM dice che la colonna è sensibile E ha trovato entità PII
                if llm_data.get("is_sensitive_column", False) and pii_entities:
                    llm_found_pii_entities_flag = True  # <--- IMPOSTA LA FLAG QUI

                # Costruzione di actual_problem_description (la logica può rimanere la stessa)
                if llm_found_pii_entities_flag:  # Usa la flag per coerenza
                    detailed_entity_descriptions = []
                    for ent in pii_entities[:2]:
                        text_val = ent.get('text', 'N/A')
                        ent_type = ent.get('type', 'N/A')
                        pii_reason = ent.get('reasoning_is_pii', 'N/D')
                        detailed_entity_descriptions.append(
                            f"'{text_val}' (tipo: {ent_type}). Motivo sensibilità PII: *{pii_reason}*"
                        )
                    problem_details_str = "\n- ".join(
                        detailed_entity_descriptions) if detailed_entity_descriptions else "Dettagli PII non specificati."
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col}':** {column_assessment}\n\n"
                        f"**PII Specifiche Rilevate ({len(pii_entities)} entità totali):**\n- {problem_details_str}"
                    )
                elif llm_data.get("is_sensitive_column", False):
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col}':** {column_assessment}\n\n"
                        f"L'LLM indica che la colonna è sensibile ma non ha dettagliato PII specifiche."
                    )
                else:
                    actual_problem_description = (
                        f"**Valutazione Sensibilità Colonna '{col}':** {column_assessment}"
                    # Qui non ci sono "PII Rilevate"
                    )

                suggested_method_from_llm = llm_data.get("suggested_anonymization_method")
                if suggested_method_from_llm in AVAILABLE_ANON_METHODS:
                    actual_suggested_method = suggested_method_from_llm
                else:
                    actual_suggested_method = default_method_fallback
                actual_method_reasoning = llm_data.get("method_reasoning", "Nessuna motivazione fornita per il metodo.")
            else:
                actual_problem_description = f"Risposta LLM per colonna '{col}' non conteneva un JSON valido: {raw_llm_output}"
        except Exception as e:
            actual_problem_description = f"Errore durante l'analisi LLM della colonna '{col}': {str(e)}"

        # ... (Logica di anonimizzazione per df_out come prima) ...
        current_col_data = df_out[col].copy()
        if actual_suggested_method == "hash":
            df_out[col] = current_col_data.astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(x) else x)
        elif actual_suggested_method == "mask":
            df_out[col] = current_col_data.astype(str).str.replace(r"[a-zA-Z0-9]", "*", regex=True)
        elif actual_suggested_method == "generalize_date":
            try:
                parsed_dates = pd.to_datetime(current_col_data, errors='coerce')
                df_out[col] = parsed_dates.dt.to_period("M").astype(str).replace('nan', pd.NA).replace('NaT', pd.NA)
            except Exception:
                df_out[col] = current_col_data
        elif actual_suggested_method == "truncate":
            df_out[col] = current_col_data.astype(str).str.slice(0, 10) + "..."

        report_rows.append({
            "Colonna": col,
            "Esempi": sample_preview,
            "Problematica": actual_problem_description,
            "MetodoSuggerito": actual_suggested_method,
            "Motivazione": actual_method_reasoning,
            "LLM_HaTrovatoEntitaPII": llm_found_pii_entities_flag  # <--- AGGIUNGI LA NUOVA COLONNA QUI
        })

    report_df = pd.DataFrame(report_rows)
    return report_df, df_out