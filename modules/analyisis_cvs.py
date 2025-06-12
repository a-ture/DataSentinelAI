# modules/analyisis_cvs.py

import asyncio
import hashlib
import json
import re
from typing import Tuple, Dict, Any, List, Optional

import openai
import pandas as pd
import numpy as np
import logging

from modules.generazione_testo import _init_lmstudio

# from scipy.stats import variation # Non usata direttamente qui se evaluate_numeric_info_loss è altrove o non usata

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

LLM_TEXT_ANON_METHODS = ["hash", "mask", "truncate", "pseudonymization", "nessuno"]
VALID_PRIVACY_CATEGORIES = {"Identificatore Diretto", "Quasi-Identificatore", "Attributo Sensibile",
                            "Attributo Non Sensibile"}


# --- Funzioni Helper per Generalizzazione, Loss Utilità (definite precedentemente, rimangono) ---
def evaluate_numeric_info_loss(original_series: pd.Series, generalized_series_str: pd.Series) -> Optional[float]:
    numeric_original = pd.to_numeric(original_series.dropna(), errors="coerce")
    if len(numeric_original) < 2: return None
    var_orig = numeric_original.var()
    if pd.isna(var_orig) or var_orig < 1e-9: return None

    def midpoint_from_range_str(range_str: str) -> Optional[float]:
        if not isinstance(range_str, str) or '-' not in range_str: return np.nan
        try:
            low, high = map(float, range_str.split('-'));
            return (low + high) / 2.0
        except ValueError:
            return np.nan

    midpoints_numeric = pd.to_numeric(generalized_series_str.dropna().apply(midpoint_from_range_str).dropna(),
                                      errors='coerce')
    if len(midpoints_numeric) < 2: return 0.0
    var_gen = midpoints_numeric.var()
    if pd.isna(var_gen): return 0.0
    return round((var_gen / var_orig) * 100, 1)


def generalize_numeric_series(series: pd.Series, num_bins: int = 5) -> pd.Series:
    cleaned_series = pd.to_numeric(series, errors="coerce")
    non_null_values = cleaned_series.dropna()
    if len(non_null_values) < num_bins or non_null_values.nunique() < 2:
        logger.info(
            f"Generalizzazione numerica: pochi valori unici ({non_null_values.nunique()}) per la colonna '{series.name}'. Restituisco come stringa.")
        return series.astype(str)
    try:
        binned_series_qcut, bin_edges_qcut = pd.qcut(non_null_values, q=num_bins, duplicates="drop", retbins=True,
                                                     precision=2)
        effective_categories = binned_series_qcut.cat.categories
        bin_labels_qcut_effective = [f"{interval.left}-{interval.right}" for interval in effective_categories]
        category_map_qcut = {code: label for code, label in
                             zip(range(len(effective_categories)), bin_labels_qcut_effective)}
        mapped_values_qcut = binned_series_qcut.cat.codes.map(category_map_qcut)
        generalized_output_series_qcut = pd.Series([pd.NA] * len(series), index=series.index, dtype=object)
        generalized_output_series_qcut.loc[non_null_values.index] = mapped_values_qcut
        return generalized_output_series_qcut.astype(str)
    except Exception as e_qcut:
        logger.warning(f"Errore pd.qcut per colonna '{series.name}': {e_qcut}. Tentativo con pd.cut.")
        try:
            effective_bins_fallback = min(num_bins, non_null_values.nunique() if non_null_values.nunique() >= 1 else 1)
            if non_null_values.nunique() == 1:
                generalized_output_series_fallback = pd.Series([pd.NA] * len(series), index=series.index, dtype=object)
                generalized_output_series_fallback.loc[non_null_values.index] = non_null_values.astype(str)
                return generalized_output_series_fallback.astype(str)
            if effective_bins_fallback < 1: effective_bins_fallback = 1
            binned_series_cut_fallback, bin_edges_fallback = pd.cut(non_null_values, bins=effective_bins_fallback,
                                                                    retbins=True, precision=2, duplicates="drop")
            effective_categories_cut = binned_series_cut_fallback.cat.categories
            bin_labels_fallback_effective = [f"{interval.left}-{interval.right}" for interval in
                                             effective_categories_cut]
            category_map_fallback = {code: label for code, label in
                                     zip(range(len(effective_categories_cut)), bin_labels_fallback_effective)}
            mapped_values_fallback = binned_series_cut_fallback.cat.codes.map(category_map_fallback)
            generalized_output_series_fallback = pd.Series([pd.NA] * len(series), index=series.index, dtype=object)
            generalized_output_series_fallback.loc[non_null_values.index] = mapped_values_fallback
            return generalized_output_series_fallback.astype(str)
        except Exception as e_cut_fallback:
            logger.error(
                f"Anche pd.cut fallito per colonna '{series.name}': {e_cut_fallback}. Restituisco serie come stringa.")
            return series.astype(str)


def generalize_date_series(series: pd.Series, granularity: str = "M") -> pd.Series:
    try:
        parsed_dates = pd.to_datetime(series, errors="coerce")
        if parsed_dates.isna().all(): return series.astype(str)
        if granularity == "Y":
            return parsed_dates.dt.to_period("Y").astype(str).replace("NaT", pd.NA)
        elif granularity == "Q":
            return parsed_dates.dt.to_period("Q").astype(str).replace("NaT", pd.NA)
        elif granularity == "M":
            return parsed_dates.dt.to_period("M").astype(str).replace("NaT", pd.NA)
        else:
            logger.warning(
                f"Granularità data '{granularity}' non riconosciuta per colonna '{series.name}', fallback a 'M'.")
            return parsed_dates.dt.to_period("M").astype(str).replace("NaT", pd.NA)
    except Exception as e:
        logger.error(f"Errore generalizzazione data per colonna '{series.name}': {e}. Restituisco serie come stringa.")
        return series.astype(str)


def _infer_parse_dates(df: pd.DataFrame, threshold: float = 0.7) -> Tuple[pd.DataFrame, List[str]]:
    df_copy = df.copy()
    inferred_datetime_col_names: List[str] = []
    potential_date_cols = df_copy.select_dtypes(include=["object", "string"]).columns
    for col in potential_date_cols:
        if col not in df_copy.columns: continue
        try:
            original_dtype_kind = df_copy[col].dtype.kind
            if original_dtype_kind in ['M', 'm', 'i', 'u', 'f']: continue
            temp_series = df_copy[col].copy()
            if temp_series.dtype == 'object':
                temp_series = temp_series.str.replace(r'\s\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$', '',
                                                      regex=True)
            parsed_series = pd.to_datetime(temp_series, errors="coerce", dayfirst=True, infer_datetime_format=True)
            if not parsed_series.empty:
                num_original_non_na = df_copy[col].notna().sum()
                num_parsed_non_na = parsed_series.notna().sum()
                if num_original_non_na > 0 and \
                        (num_parsed_non_na / num_original_non_na) >= threshold and \
                        (num_parsed_non_na / len(df_copy[col]) if len(df_copy[col]) > 0 else 0) >= threshold / 2.0:
                    logger.info(
                        f"Colonna '{col}' inferita come datetime e convertita (dtype originale: {original_dtype_kind}).")
                    df_copy[col] = parsed_series
                    inferred_datetime_col_names.append(col)
        except Exception as e:
            logger.warning(f"Errore inferenza date per colonna '{col}': {e}")
    return df_copy, inferred_datetime_col_names


# --- Funzione _inspect_column_async_text AGGIORNATA ---
async def _inspect_column_async_text(
        col_name: str,
        # series_for_fallback_checks non è più necessaria perché i fallback sono rimossi
        sample_preview_for_llm: str,  # Stringa di esempi per il prompt LLM
        model_api_id: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Analizza colonna testuale con LLM. LLM decide sensibilità e metodo.
    Non ci sono più fallback rule-based per email, gender, city qui.
    """
    # RIMOSSI tutti i fallback “rule‐based” da questa funzione.
    logger.info(f"Analisi LLM per colonna testuale: '{col_name}'...")

    valid_text_methods_for_prompt = ", ".join([f'"{m}"' for m in LLM_TEXT_ANON_METHODS])
    prompt_unificato = f"""
Analizza la colonna CSV testuale chiamata "{col_name}", i cui valori di esempio sono:
"{sample_preview_for_llm}".

1) Classifica la colonna in base alla sua **sensibilità** (usa ESATTAMENTE una di queste quattro categorie, in italiano):
   - "Identificatore Diretto"
   - "Quasi-Identificatore"
   - "Attributo Sensibile"
   - "Attributo Non Sensibile"

   Spiega in breve perché hai scelto quella categoria, tenendo presente **il contesto di Machine Learning**:
   - Se la colonna contiene nomi o e‐mail, potrebbe essere Identificatore Diretto.
   - Se la colonna è qualcosa come genere, città, codice postale ecc., valuta se in quel dataset
     è davvero un QID oppure se è “Attributo Non Sensibile” (ad esempio “Gender” da solo non è sempre QID, mentre una “Città” dettagliata spesso è QID).

2) Elenca eventuali entità o pattern PII presenti negli esempi (es. indirizzi email, nomi completi, 
   codici paziente, ecc.) e per ognuna riportami:
   - "text": il valore PII rilevato
   - "type": il tipo (es. “PERSON”, “EMAIL”, “ID_NUMBER”, “LOCATION”, “HEALTH_CONDITION”, “CREDIT_CARD_NUMBER”, etc.)
   - "reasoning": perché consideri quella stringa come PII.

   Se non trovi PII specifiche negli esempi, lascia la lista "pii_entities_found" vuota.

3) Suggerisci **il miglior metodo di anonimizzazione** da applicare a questa colonna testuale in un contesto di Machine Learning.
   Scegli ESATTAMENTE uno tra i seguenti metodi validi per le colonne testuali:
   {valid_text_methods_for_prompt}

   Spiega brevemente perché quel metodo è il più adatto per questa colonna e il suo impatto ML. Considera:
   - "hash": garantisce forte anonimato ma perde relazione di stringhe (utile se non serve matching o pattern interni alla stringa).
   - "mask": preserva parte dell'informazione (es. lunghezza, presenza di caratteri) ma è meno robusto.
   - "truncate": conserva i primi N caratteri (utile se i prefissi sono informativi).
   - "pseudonymization": sostituisce con ID unici mantenendo la distinguibilità dei valori originali (utile se serve mantenere unicità ma non il valore originale).
   - "nessuno": se la colonna non necessita di anonimizzazione per ML (es. label, descrizioni generiche non sensibili).

Rispondi **ESCLUSIVAMENTE** in formato JSON, con questa struttura (niente testo extra):
{{
  "is_sensitive_column": boolean,
  "column_privacy_category": "stringa_categoria_scelta",
  "column_sensitivity_assessment": "testo breve spiega perché",
  "pii_entities_found": [
    {{"text": "stringa_PII", "type": "tipo_PII", "reasoning": "perché è PII"}}
  ],
  "suggested_anonymization_method": "stringa_metodo_scelto_dalla_lista",
  "method_reasoning": "breve spiegazione dell’impatto ML e privacy"
}}
"""
    default_error_response = {
        "error": "Errore non specificato.", "is_sensitive_column": False,
        "column_privacy_category": "Errore Analisi LLM",
        "column_sensitivity_assessment": "Analisi LLM fallita.",
        "pii_entities_found": [], "suggested_anonymization_method": "nessuno",
        "method_reasoning": "Analisi LLM fallita."
    }
    try:
        response = await openai.ChatCompletion.acreate(
            model=model_api_id,
            messages=[
                {"role": "system",
                 "content": """Sei un esperto consulente di data privacy. Quando classifichi una colonna CSV testuale:
- Usa ESATTAMENTE una delle quattro categorie (in italiano) richieste.
- Per il metodo di anonimizzazione, scegli ESATTAMENTE uno dalla lista fornita.
- Valuta il contesto ML: spiega brevemente l’impatto di quel metodo.
Rispondi solo con un oggetto JSON valido (niente testo fuori dal JSON)."""},
                {"role": "user", "content": prompt_unificato.strip()}
            ],
            temperature=0.1, max_tokens=1500, request_timeout=45
        )
        raw_llm_output = response.choices[0].message.content.strip()
        json_str_to_parse = None
        match_md_json = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_llm_output, re.DOTALL | re.IGNORECASE)
        if match_md_json:
            json_str_to_parse = match_md_json.group(1)
        else:
            first_brace = raw_llm_output.find('{')
            last_brace = raw_llm_output.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                potential_json = raw_llm_output[first_brace: last_brace + 1]
                try:
                    json.loads(potential_json);
                    json_str_to_parse = potential_json
                except json.JSONDecodeError:
                    logger.warning(f"Potenziale JSON per '{col_name}' non valido: {potential_json[:200]}...")

        if json_str_to_parse:
            try:
                parsed = json.loads(json_str_to_parse)
                parsed.setdefault("is_sensitive_column", bool(parsed.get("pii_entities_found")))
                parsed.setdefault("column_sensitivity_assessment", "Valutazione LLM non fornita.")
                parsed.setdefault("pii_entities_found", [])
                parsed.setdefault("method_reasoning", "Motivazione LLM non fornita.")

                cat_llm = parsed.get("column_privacy_category", "Attributo Non Sensibile")
                assessment_adj_info = ""
                # Normalizzazione più precisa basata sulla lista esatta
                if cat_llm not in VALID_PRIVACY_CATEGORIES:
                    cat_llm_lower_norm = str(cat_llm).lower().replace('-', ' ').replace('_', ' ')
                    # Tentativo di mappare comunque
                    if "identificatore diretto" in cat_llm_lower_norm:
                        parsed["column_privacy_category"] = "Identificatore Diretto"
                    elif "quasi identificatore" in cat_llm_lower_norm:
                        parsed["column_privacy_category"] = "Quasi-Identificatore"
                    elif "attributo sensibile" in cat_llm_lower_norm:
                        parsed["column_privacy_category"] = "Attributo Sensibile"
                    else:  # Se non mappabile, usa default e logga
                        parsed["column_privacy_category"] = "Attributo Non Sensibile"  # Default sicuro
                        assessment_adj_info = f" [Cat. LLM non standard: '{cat_llm}', default a Non Sensibile]"
                if assessment_adj_info: parsed["column_sensitivity_assessment"] += assessment_adj_info

                meth_llm = parsed.get("suggested_anonymization_method", "nessuno").lower().strip()
                if meth_llm not in LLM_TEXT_ANON_METHODS:
                    parsed[
                        "method_reasoning"] += f" [Avviso: metodo LLM '{meth_llm}' non valido per testo, usato 'nessuno']"
                    parsed["suggested_anonymization_method"] = "nessuno"
                else:
                    parsed["suggested_anonymization_method"] = meth_llm  # Già validato/mappato

                return col_name, parsed
            except json.JSONDecodeError as json_e:
                error_detail = f"Errore JSONDecodeError per colonna '{col_name}': {json_e}. Stringa (primi 500char):\n'{json_str_to_parse[:500]}...'"
                default_error_response["error"] = error_detail
                default_error_response["method_reasoning"] = error_detail
                logger.error(f"JSONDecodeError in _inspect_column_async_text ({col_name}): {error_detail}")
                return col_name, default_error_response
        else:
            error_detail = f"Nessun blocco JSON identificabile per colonna '{col_name}'. Output LLM (primi 500char):\n'{raw_llm_output[:500]}...'"
            default_error_response["error"] = error_detail
            default_error_response["method_reasoning"] = error_detail
            logger.error(f"Nessun JSON in _inspect_column_async_text ({col_name}): {error_detail}")
            return col_name, default_error_response
    except openai.error.Timeout as to_e:
        error_detail = f"Timeout API OpenAI per colonna '{col_name}': {str(to_e)}"
        default_error_response["error"] = error_detail
        default_error_response["method_reasoning"] = error_detail
        logger.error(error_detail)
        return col_name, default_error_response
    except openai.error.APIError as api_e:
        error_detail = f"APIError OpenAI per colonna '{col_name}': {type(api_e).__name__} - {str(api_e)}"
        default_error_response["error"] = error_detail
        default_error_response["method_reasoning"] = error_detail
        logger.error(error_detail)
        return col_name, default_error_response
    except Exception as e:
        error_detail = f"Errore imprevisto in _inspect_column_async_text ({col_name}'): {type(e).__name__} - {str(e)}"
        default_error_response["error"] = error_detail
        default_error_response["method_reasoning"] = error_detail
        logger.error(error_detail, exc_info=True)
        return col_name, default_error_response


async def _analyze_text_columns_llm(
        df_text_cols_to_analyze: pd.DataFrame,
        model_api_id_for_text: str,
        sample_size: int,
        max_req: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_rows_text = []
    df_out_text_anonymized = df_text_cols_to_analyze.copy()
    if df_text_cols_to_analyze.empty:
        return pd.DataFrame(report_rows_text), df_out_text_anonymized

    tasks_text_to_run = []
    for col_name_txt in df_text_cols_to_analyze.columns:
        unique_vals_txt = df_text_cols_to_analyze[col_name_txt].dropna().astype(str).unique().tolist()
        sample_preview_txt_str = "; ".join(unique_vals_txt[:sample_size]) + (
            "…" if len(unique_vals_txt) > sample_size else "")
        # Non passiamo più series_for_fallback_checks perché i fallback sono stati rimossi da _inspect_column_async_text
        tasks_text_to_run.append(
            _inspect_column_async_text(col_name_txt, sample_preview_txt_str, model_api_id_for_text))

    sem_text_limiter = asyncio.Semaphore(max_req)

    async def run_with_sem_and_delay(task, task_idx, total_tasks_count):
        if task_idx > 0 and task_idx % max_req == 0:
            delay_duration = 0.2
            logger.info(
                f"Rallentamento: pausa di {delay_duration}s prima del batch successivo di richieste LLM (task {task_idx}/{total_tasks_count}).")
            await asyncio.sleep(delay_duration)
        async with sem_text_limiter:
            return await task

    llm_results_for_text_cols_gathered = await asyncio.gather(
        *(run_with_sem_and_delay(t, i, len(tasks_text_to_run)) for i, t in enumerate(tasks_text_to_run))
    )

    for col_name_res_txt, llm_data_res_txt in llm_results_for_text_cols_gathered:
        has_error_txt = "error" in llm_data_res_txt
        is_sensitive_col_llm_txt = llm_data_res_txt.get("is_sensitive_column", False)
        pii_entities_llm_txt = llm_data_res_txt.get("pii_entities_found", [])
        llm_found_overall_pii_flag_txt = is_sensitive_col_llm_txt or bool(pii_entities_llm_txt)
        actual_problem_description_txt = ""
        if has_error_txt:
            actual_problem_description_txt = f"**Errore Analisi Colonna '{col_name_res_txt}':** {llm_data_res_txt['error']}"
        else:
            column_assessment_llm_txt = llm_data_res_txt.get("column_sensitivity_assessment", "Nessuna valutazione.")
            actual_problem_description_txt = f"**Valutazione Sensibilità Colonna '{col_name_res_txt}' (LLM):** {column_assessment_llm_txt}"  # Modificato
            if pii_entities_llm_txt:
                detailed_entity_descriptions_txt = [
                    f"'{ent.get('text', 'N/A')}' (tipo: {ent.get('type', 'N/A')}). Motivo: *{ent.get('reasoning', 'N/D')}*"
                    for ent in pii_entities_llm_txt[:3]]
                problem_details_str_txt = "\n- ".join(detailed_entity_descriptions_txt)
                actual_problem_description_txt += f"\n\n**PII Specifiche Rilevate da LLM ({len(pii_entities_llm_txt)} totali):**\n- {problem_details_str_txt}"
            elif is_sensitive_col_llm_txt:
                actual_problem_description_txt += "\n\nL'LLM considera la colonna sensibile."

        final_applied_method_txt = llm_data_res_txt.get("suggested_anonymization_method", "nessuno")
        actual_method_reasoning_llm_txt = llm_data_res_txt.get("method_reasoning", "Nessuna motivazione.")
        llm_column_category_val_txt = llm_data_res_txt.get("column_privacy_category", "Non Classificato")

        current_col_data_for_anon_txt = df_out_text_anonymized[col_name_res_txt].copy()

        if final_applied_method_txt == "hash":
            df_out_text_anonymized[col_name_res_txt] = current_col_data_for_anon_txt.astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest() if pd.notna(x) else x)
        elif final_applied_method_txt == "pseudonymization":
            df_out_text_anonymized[col_name_res_txt] = current_col_data_for_anon_txt.astype(str).apply(
                lambda x: f"PSEUDO_{hashlib.md5(x.encode()).hexdigest()[:16].upper()}" if pd.notna(x) else x)
        elif final_applied_method_txt == "mask":
            df_out_text_anonymized[col_name_res_txt] = current_col_data_for_anon_txt.astype(str).str.replace(
                r"[a-zA-Z0-9\.\@\-\_]", "*", regex=True)
        elif final_applied_method_txt == "truncate":
            df_out_text_anonymized[col_name_res_txt] = current_col_data_for_anon_txt.astype(str).str.slice(0,
                                                                                                           10) + "..."

        report_rows_text.append({
            "Colonna": col_name_res_txt,
            "Esempi": "; ".join(
                df_text_cols_to_analyze[col_name_res_txt].dropna().astype(str).unique()[:sample_size].tolist()),
            "Problematica": actual_problem_description_txt, "MetodoSuggerito": final_applied_method_txt,
            "Motivazione": actual_method_reasoning_llm_txt, "LLM_HaTrovatoEntitaPII": llm_found_overall_pii_flag_txt,
            "CategoriaLLM": llm_column_category_val_txt,
            "VarianzaPreservata(%)": "N/A (Testuale)"
        })
    return pd.DataFrame(report_rows_text), df_out_text_anonymized


def _process_numeric_columns_rules(df_numeric: pd.DataFrame, num_bins: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_numeric_list_rules = []
    df_out_num_rules = df_numeric.copy()
    if df_numeric.empty: return pd.DataFrame(report_numeric_list_rules), df_out_num_rules

    for col_num_rules in df_numeric.columns:
        series_num = df_numeric[col_num_rules]
        uniq_ratio_num = series_num.nunique(dropna=False) / len(series_num) if len(series_num) > 0 else 0
        metodo_num_rules, motivo_num_rules, cat_num_rules = "nessuno", "", ""
        info_loss_pct_num: Optional[float] = 100.0

        if uniq_ratio_num > 0.85 and series_num.nunique(dropna=False) > num_bins:
            gen_candidate_num = generalize_numeric_series(series_num, num_bins=num_bins)
            info_loss_pct_num = evaluate_numeric_info_loss(series_num, gen_candidate_num)
            df_out_num_rules[col_num_rules] = gen_candidate_num
            metodo_num_rules = f"generalize_numeric (quantili={num_bins})"
            cat_num_rules = "Quasi-Identificatore"
            motivo_num_rules = f"Alta cardinalità: generalizzato in {num_bins} quantili. Varianza preservata stimata: {info_loss_pct_num if info_loss_pct_num is not None else 'N/D'}%."
        else:
            metodo_num_rules = "nessuno"
            cat_num_rules = "Attributo Non Sensibile"
            motivo_num_rules = "Bassa cardinalità o pochi valori unici: non richiede generalizzazione numerica."

        report_numeric_list_rules.append({
            "Colonna": col_num_rules,
            "Esempi": "; ".join(map(str, series_num.dropna().unique()[:3])) + (
                "…" if series_num.nunique(dropna=False) > 3 else ""),
            "CategoriaLLM": cat_num_rules, "LLM_HaTrovatoEntitaPII": False,
            "Problematica": f"Valutazione colonna '{col_num_rules}': Numerica, rapporto unicità {uniq_ratio_num:.2f}.",
            "MetodoSuggerito": metodo_num_rules, "Motivazione": motivo_num_rules,
            "VarianzaPreservata(%)": round(info_loss_pct_num, 1) if info_loss_pct_num is not None else "N/D"
        })
    return pd.DataFrame(report_numeric_list_rules), df_out_num_rules


def _process_date_columns_rules(df_date: pd.DataFrame, granularity_date: str = "M") -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    report_date_list_rules = []
    df_out_date_rules = df_date.copy()
    if df_date.empty: return pd.DataFrame(report_date_list_rules), df_out_date_rules

    for col_date_rules in df_date.columns:
        series_date = df_date[col_date_rules]
        metodo_date_rules, motivo_date_rules, cat_date_rules = "nessuno", "", "Quasi-Identificatore"

        if pd.api.types.is_datetime64_any_dtype(series_date):
            try:
                gen_date_series_val = generalize_date_series(series_date, granularity=granularity_date)
                df_out_date_rules[col_date_rules] = gen_date_series_val
                metodo_date_rules = f"generalize_date ({granularity_date})"
                motivo_date_rules = {"Y": "Generalizzazione a livello Anno.",
                                     "Q": "Generalizzazione a livello Trimestre.",
                                     "M": "Generalizzazione a livello Anno-Mese."}.get(granularity_date,
                                                                                       "Generalizzazione data.")
                esempi_date_rules = "; ".join(series_date.dt.strftime("%Y-%m-%d").dropna().unique()[:3]) + (
                    "…" if series_date.nunique(dropna=False) > 3 else "")
            except Exception as e_date_gen:
                logger.error(f"Errore generalizzazione data per colonna '{col_date_rules}': {e_date_gen}")
                metodo_date_rules = "nessuno"
                cat_date_rules = "Errore Generalizzazione"
                motivo_date_rules = f"Errore durante la generalizzazione: {e_date_gen}"
                esempi_date_rules = "; ".join(series_date.dropna().astype(str).unique()[:3]) + (
                    "…" if series_date.nunique(dropna=False) > 3 else "")
        else:
            metodo_date_rules = "nessuno"
            cat_date_rules = "Attributo Non Sensibile (Tipo Inatteso)"
            motivo_date_rules = "Colonna non riconosciuta come tipo datetime valido per generalizzazione automatica."
            esempi_date_rules = "; ".join(series_date.dropna().astype(str).unique()[:3]) + (
                "…" if series_date.nunique(dropna=False) > 3 else "")

        report_date_list_rules.append({
            "Colonna": col_date_rules, "Esempi": esempi_date_rules,
            "CategoriaLLM": cat_date_rules, "LLM_HaTrovatoEntitaPII": False,
            "Problematica": f"Valutazione colonna '{col_date_rules}': {motivo_date_rules}",
            # Modificato per più chiarezza
            "MetodoSuggerito": metodo_date_rules, "Motivazione": motivo_date_rules,
            "VarianzaPreservata(%)": "N/A (Data)"
        })
    return pd.DataFrame(report_date_list_rules), df_out_date_rules


async def analyze_and_anonymize_csv(
        df_full_input: pd.DataFrame,
        model_api_id: Optional[str],
        sample_size_for_preview: int = 5,
        max_concurrent_requests: int = 3,
        num_bins_numeric: int = 5,
        granularity_for_dates: str = "M"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analizza e anonimizza le colonne di un DataFrame CSV.
    Le colonne testuali sono inviate all'LLM, numeriche e datetime sono generalizzate con regole.
    """
    _init_lmstudio(model_api_id if model_api_id else "local_model_generic_placeholder")
    logger.info("Avvio analyze_and_anonymize_csv...")

    logger.info("Fase 1: Inferenza e parsing delle date iniziali...")
    df_processed_initial_dates, inferred_date_col_names_list = _infer_parse_dates(df_full_input.copy(), threshold=0.7)

    df_datetime_cols_final = df_processed_initial_dates.select_dtypes(include=["datetime64[ns]", "datetimetz"])
    df_remaining_for_text_numeric = df_processed_initial_dates.drop(columns=df_datetime_cols_final.columns,
                                                                    errors='ignore')
    df_numeric_cols_final = df_remaining_for_text_numeric.select_dtypes(include=["int64", "float64"])
    df_text_columns_for_llm = df_remaining_for_text_numeric.select_dtypes(include=["object", "string"])
    df_text_columns_for_llm = df_text_columns_for_llm.drop(columns=df_numeric_cols_final.columns, errors='ignore')

    logger.info(f"Colonne testuali per LLM: {df_text_columns_for_llm.columns.tolist()}")
    logger.info(f"Colonne numeriche per Regole: {df_numeric_cols_final.columns.tolist()}")
    logger.info(f"Colonne datetime per Regole: {df_datetime_cols_final.columns.tolist()}")

    report_text_df_res = pd.DataFrame()
    df_out_text_res = pd.DataFrame()
    if not df_text_columns_for_llm.empty and model_api_id:
        logger.info(f"Fase 2a: Analisi LLM per {len(df_text_columns_for_llm.columns)} colonne testuali...")
        report_text_df_res, df_out_text_res = await _analyze_text_columns_llm(
            df_text_columns_for_llm, model_api_id, sample_size_for_preview, max_concurrent_requests
        )
    elif not df_text_columns_for_llm.empty and not model_api_id:
        logger.warning("Nessun model_api_id fornito, le colonne testuali non saranno analizzate dall'LLM.")
        report_text_df_res = pd.DataFrame([{
            "Colonna": col, "Esempi": "; ".join(map(str, df_text_columns_for_llm[col].dropna().unique()[:3])),
            "CategoriaLLM": "Non Analizzato (Testuale)", "LLM_HaTrovatoEntitaPII": False,
            "Problematica": "Analisi LLM non eseguita (modello API non specificato).",
            "MetodoSuggerito": "nessuno", "Motivazione": "-", "VarianzaPreservata(%)": "N/A"
        } for col in df_text_columns_for_llm.columns])
        df_out_text_res = df_text_columns_for_llm.copy()

    logger.info(f"Fase 2b: Processamento regole per {len(df_numeric_cols_final.columns)} colonne numeriche...")
    report_num_df_res, df_out_num_res = _process_numeric_columns_rules(df_numeric_cols_final, num_bins=num_bins_numeric)

    logger.info(f"Fase 2c: Processamento regole per {len(df_datetime_cols_final.columns)} colonne datetime...")
    report_date_df_res, df_out_date_res = _process_date_columns_rules(df_datetime_cols_final,
                                                                      granularity_date=granularity_for_dates)

    logger.info("Fase 3: Unione dei risultati...")
    list_of_reports_to_concat = []
    if not report_text_df_res.empty: list_of_reports_to_concat.append(report_text_df_res)
    if not report_num_df_res.empty: list_of_reports_to_concat.append(report_num_df_res)
    if not report_date_df_res.empty: list_of_reports_to_concat.append(report_date_df_res)

    report_df_final_concat_res = pd.concat(list_of_reports_to_concat, ignore_index=True,
                                           sort=False) if list_of_reports_to_concat else pd.DataFrame()

    if not report_df_final_concat_res.empty and report_df_final_concat_res["Colonna"].duplicated().any():
        dup_cols_final_report = report_df_final_concat_res["Colonna"][
            report_df_final_concat_res["Colonna"].duplicated()].unique().tolist()
        logger.warning(
            f"Colonne duplicate trovate nel report finale: {dup_cols_final_report}. Rimozione duplicati mantenendo la prima.")
        report_df_final_concat_res = report_df_final_concat_res.drop_duplicates(subset=["Colonna"], keep="first")

    df_out_full_final_res = df_full_input.copy()
    # Applica le modifiche dai DataFrame specifici al DataFrame completo di output
    # Colonne testuali anonimizzate
    if not df_out_text_res.empty:
        for col_to_update in df_out_text_res.columns:
            if col_to_update in df_out_full_final_res.columns:
                df_out_full_final_res[col_to_update] = df_out_text_res[col_to_update]
    # Colonne numeriche generalizzate
    if not df_out_num_res.empty:
        for col_to_update in df_out_num_res.columns:
            if col_to_update in df_out_full_final_res.columns:
                df_out_full_final_res[col_to_update] = df_out_num_res[col_to_update]
    # Colonne datetime generalizzate
    if not df_out_date_res.empty:
        for col_to_update in df_out_date_res.columns:
            if col_to_update in df_out_full_final_res.columns:
                df_out_full_final_res[col_to_update] = df_out_date_res[col_to_update]

    logger.info("Fase 4: Sanificazione finale dei DataFrame di output...")
    # Sanificazione: converti tutte le colonne object a stringa e riempi NA con stringa vuota per report.
    # Per df_out, riempi NA con np.nan per coerenza tipi se possibile, poi converti object a stringa.
    if not report_df_final_concat_res.empty:
        for col_obj_rep in report_df_final_concat_res.select_dtypes(include=['object']).columns:
            try:
                report_df_final_concat_res[col_obj_rep] = report_df_final_concat_res[col_obj_rep].fillna("").astype(str)
            except Exception as e_final_san_rep:
                logger.warning(f"San. report_df col '{col_obj_rep}': {e_final_san_rep}")

    if df_out_full_final_res is not None:
        for col_obj_out in df_out_full_final_res.select_dtypes(include=['object']).columns:
            try:
                # Non convertire a stringa se è una colonna numerica/data generalizzata che è object per le fasce ma i cui valori sono ok
                # Se una colonna è object e contiene misto di stringhe e pd.NA/np.nan, fillna("") e astype(str) è ok.
                # Se contiene oggetti complessi, astype(str) è una buona sanificazione.
                if not pd.api.types.is_numeric_dtype(df_out_full_final_res[col_obj_out]) and \
                        not pd.api.types.is_datetime64_any_dtype(df_out_full_final_res[col_obj_out]):
                    df_out_full_final_res[col_obj_out] = df_out_full_final_res[col_obj_out].fillna("").astype(str)
            except Exception as e_final_san_out:
                logger.warning(f"San. df_out col '{col_obj_out}': {e_final_san_out}")

    logger.info("analyze_and_anonymize_csv completata.")
    return report_df_final_concat_res, df_out_full_final_res


# Copia qui la funzione get_llm_overall_csv_comment aggiornata
def get_llm_overall_csv_comment(
        column_analysis_report: pd.DataFrame,
        risk_metrics_calculated: dict,
        loss_of_utility_metrics: Optional[dict],
        qid_identified_list: list,
        sa_identified_list: list,
        model_api_id: Optional[str],
        file_name: str = "Il file CSV analizzato"
) -> str:
    if column_analysis_report.empty and not risk_metrics_calculated and not loss_of_utility_metrics:
        return "### Valutazione Privacy – N/D\n\nNessun dato di analisi disponibile per generare un report complessivo."

    num_total_cols_analyzed_rep = len(column_analysis_report)
    pii_mask_overall_rep = pd.Series([False] * num_total_cols_analyzed_rep, index=column_analysis_report.index)
    if "LLM_HaTrovatoEntitaPII" in column_analysis_report.columns and "CategoriaLLM" in column_analysis_report.columns:
        pii_mask_overall_rep = (column_analysis_report["LLM_HaTrovatoEntitaPII"] == True) | \
                               (column_analysis_report["CategoriaLLM"].str.contains(
                                   "Identificatore Diretto|Attributo Sensibile", case=False, na=False, regex=True))
    elif "Problematica" in column_analysis_report.columns:
        pii_mask_overall_rep = column_analysis_report["Problematica"].str.contains(r"PII.*Rilevat|colonna è sensibile",
                                                                                   case=False, na=False, regex=True)

    cols_with_pii_df_overall_rep = column_analysis_report[pii_mask_overall_rep]
    num_cols_with_pii_overall_rep = len(cols_with_pii_df_overall_rep)
    risk_pct_overall_rep = round((num_cols_with_pii_overall_rep / num_total_cols_analyzed_rep) * 100,
                                 1) if num_total_cols_analyzed_rep > 0 else 0.0
    k_min_val_overall_rep = risk_metrics_calculated.get("k_anonymity_min", "N/D")
    records_singoli_val_overall_rep = risk_metrics_calculated.get("records_singoli", "N/D")
    pii_cols_list_str_rep = ", ".join(cols_with_pii_df_overall_rep["Colonna"].unique()[:5])
    if len(cols_with_pii_df_overall_rep["Colonna"].unique()) > 5: pii_cols_list_str_rep += "..."
    ldiv_entries_rep = risk_metrics_calculated.get("l_diversity", {})
    # Modificato per stringa più concisa per l-diversity
    ldiv_str_rep = ", ".join([f"{col_ldiv} (l={metrics_ldiv.get('l_min', 'N/D')})" for col_ldiv, metrics_ldiv in
                              ldiv_entries_rep.items()]) if ldiv_entries_rep else 'Nessuna calcolata o SA non specificati'

    loss_info_summary_str = "Non calcolata o non applicabile per colonne numeriche di esempio."
    if loss_of_utility_metrics:
        temp_loss_strs = []
        # Itera solo sulle chiavi che sono presumibilmente nomi di colonna (non 'n_righe_...')
        for col_util, metrics_util in loss_of_utility_metrics.items():
            if isinstance(metrics_util, dict) and "percentuale_preserved_var (%)" in metrics_util:
                perc_pres_util = metrics_util["percentuale_preserved_var (%)"]
                temp_loss_strs.append(f"Col. '{col_util}': {perc_pres_util}% varianza preservata")
        if temp_loss_strs: loss_info_summary_str = "; ".join(temp_loss_strs)

    context_summary_rep = f"""
File: '{file_name}'
Colonne totali analizzate: {num_total_cols_analyzed_rep}
Colonne con PII/sensibili stimate: {num_cols_with_pii_overall_rep} (prime: {pii_cols_list_str_rep if pii_cols_list_str_rep else 'nessuna'}) ({risk_pct_overall_rep}%)
Quasi-Identificatori (QID) per metriche: {', '.join(qid_identified_list) if qid_identified_list else 'Nessuno'}
Attributi Sensibili (SA) per metriche: {', '.join(sa_identified_list) if sa_identified_list else 'Nessuno'}

Metriche Rischio Re-identificazione (soglie indicative: k>=5, l>=2, record singoli=0):
- k-anonymity (min k): {k_min_val_overall_rep}
- k-anonymity (record singoli): {records_singoli_val_overall_rep}
- l-diversity per SA: {ldiv_str_rep}

Stima Perdita di Utilità (Varianza Preservata dopo generalizzazione, per colonne numeriche di esempio):
- {loss_info_summary_str}
"""
    prompt_to_llm_rep = f"""
Sei un consulente esperto di data privacy e protezione dati (GDPR).
Di seguito trovi un **contesto sintetico** derivante dall'analisi di un dataset CSV:
{context_summary_rep}

Tuo compito: genera un **Report di Valutazione Privacy** in Markdown. Il report deve essere chiaro, conciso e orientato all'azione, considerando l'utilità dei dati per il Machine Learning.
Struttura il report nelle seguenti sezioni principali (usa ESATTAMENTE questi titoli):
1. **Sintesi Esecutiva e Livello di Rischio Complessivo**
   (Valuta il livello di rischio generale del dataset: Basso, Medio, Alto, Molto Alto. Riassumi le principali scoperte, includendo l'impatto della perdita di utilità stimata se rilevante.)
2. **Tabella delle Metriche di Re-identificazione**
   (Presenta una tabella Markdown con le metriche calcolate: k-anonymity (minimo k e record singoli), e l-diversity per ciascun SA. Indica valore osservato, soglia consigliata (es. k>=5, l>=2), e un commento sull'esito: ✅ OK, ⚠️ Attenzione!, 🛑 Rischio Alto!, N/D.)
3. **Analisi dei Rischi di Re-identificazione e Sensibilità dei Dati**
   (Commenta i risultati delle metriche e la natura delle PII/colonne sensibili. Discuti il trade-off privacy/utilità alla luce della perdita di varianza stimata.)
4. **Principali Rischi Legali, Operativi e Reputazionali**
   (Elenca 2-4 rischi chiave (GDPR).)
5. **Azioni Correttive e Raccomandazioni Prioritarie**
   (Elenca azioni concrete: anonimizzazione (considerando i metodi applicati e la loro efficacia/impatto sull'utilità), miglioramento metriche, controlli organizzativi, DPIA.)

Non inserire commenti vuoti. Sii analitico, basandoti sul contesto.
"""
    if not model_api_id:
        logger.warning("Nessun model_api_id fornito per get_llm_overall_csv_comment. Restituisco solo il contesto.")
        return f"### Valutazione Privacy – Parziale (Analisi LLM non eseguita)\n\n**Contesto disponibile per valutazione manuale:**\n```text\n{context_summary_rep.strip()}\n```\n\nL'analisi dettagliata e le raccomandazioni richiederebbero un modello LLM."
    try:
        response = openai.ChatCompletion.create(model=model_api_id, messages=[{"role": "system",
                                                                               "content": "Sei un consulente esperto di data privacy (GDPR). Genera un report Markdown strutturato e con raccomandazioni chiare, basato sul contesto fornito. Usa i titoli di sezione richiesti."},
                                                                              {"role": "user",
                                                                               "content": prompt_to_llm_rep.strip()}],
                                                temperature=0.2, max_tokens=2500)
        return response.choices[0].message.content.strip()
    except Exception as e_overall:
        logger.error(f"Errore LLM in get_llm_overall_csv_comment: {type(e_overall).__name__} - {e_overall}",
                     exc_info=True)
        error_header_rep = f"### Valutazione Privacy Complessiva – ⚠️ Errore LLM\n\n"
        error_message_md_rep = (f"{error_header_rep}"
                                f"**1. Sintesi Esecutiva e Livello di Rischio**\n"
                                f"Impossibile generare il commento finale a causa di un errore con l'LLM: `{type(e_overall).__name__}`.\n\n"
                                f"**Contesto disponibile per analisi manuale:**\n```text\n{context_summary_rep.strip()}\n```\n\n"
                                f"Si raccomanda una revisione manuale approfondita dei dati e dei rischi basata sul contesto sopra.")
        return error_message_md_rep
