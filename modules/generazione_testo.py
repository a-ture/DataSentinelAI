# modules/generazione_testo.py
import json
import re
import ast  # <-- Aggiungi questo import all'inizio del file!
from collections import Counter
from typing import Optional, Dict, Any
import logging  # Importa il modulo logging
import socket
import subprocess
import time
from functools import lru_cache

import openai
from gliner import GLiNER  # Mantieni se usato da get_ner_pipelines
from torch import cuda  # Mantieni se usato da get_ner_pipelines
from transformers import pipeline, BertTokenizerFast, BertForTokenClassification, AutoTokenizer, \
    AutoModelForTokenClassification  # Mantieni se usato da get_ner_pipelines
from typing import Union
import json
import openai
import logging

logger = logging.getLogger(__name__)
# Configurazione del Logger per questo modulo

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Le tue impostazioni API globali (se usi LM Studio)
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"


@lru_cache(maxsize=None)
def _init_lmstudio(model_api_id: str, port: int = 1234):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_running = False
    try:
        sock.connect(("127.0.0.1", port))
        server_running = True
        print(f"LMStudio server is already running on port {port}")  # Considera logger.info
    except (ConnectionRefusedError, OSError):
        print(f"LMStudio server not found on port {port}. Attempting to start server...")  # Considera logger.warning
        try:
            subprocess.Popen(
                ["lms", "server", "start", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(15)  # Dai tempo al server di avviarsi
            sock_check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock_check.connect(("127.0.0.1", port))
                print(f"LMStudio server started successfully on port {port}.")  # Considera logger.info
                server_running = True
            except (ConnectionRefusedError, OSError):
                print(
                    f"Failed to connect to LMStudio server after attempting to start it on port {port}.")  # Considera logger.error
            finally:
                sock_check.close()
        except FileNotFoundError:
            print(
                "ERROR: 'lms' command not found. Make sure LMStudio CLI is installed and in your system's PATH.")  # Considera logger.critical
            return  # Esci se il comando non è trovato
        except Exception as e:
            print(f"An error occurred while trying to start LMStudio server: {e}")  # Considera logger.error
            return
    finally:
        sock.close()

    if not server_running:
        print(
            "Skipping model load as LMStudio server is not running or could not be started.")  # Considera logger.warning
        return

    model_to_load_cli = model_api_id  # Assumi che model_api_id sia l'identificatore per 'lms load'
    print(
        f"Attempting to load model via CLI: '{model_to_load_cli}' using 'lms load {model_to_load_cli}'. This call is "
        f"memoized.")  # Considera logger.info
    try:
        command_list = ["lms", "load", model_to_load_cli]

        process_result = subprocess.run(
            command_list, check=False, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=90
        )
        if process_result.returncode != 0:
            logger.error(f"Error during 'lms load {model_to_load_cli}': Exit code {process_result.returncode}")
            logger.error(f"STDERR: {process_result.stderr.strip() if process_result.stderr else 'N/A'}")
        else:
            logger.info(
                f"STDOUT from 'lms load {model_to_load_cli}': {process_result.stdout.strip() if process_result.stdout else 'N/A'}")

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout: 'lms load {model_to_load_cli}' took longer than 90 seconds.")
    except FileNotFoundError:  # Già gestito sopra, ma per sicurezza
        logger.critical("ERROR: 'lms' command not found for model loading.")
    except Exception as e_load:
        logger.error(f"An unexpected error occurred during 'lms load' model loading: {e_load}")
    time.sleep(5)  # Pausa dopo il tentativo di caricamento


@lru_cache(maxsize=None)
def get_ner_pipelines():
    device = 0 if cuda.is_available() else -1
    logger.info(f"Initializing NER pipelines on device: {'GPU' if device == 0 else 'CPU'}")
    pipelines_dict = {}
    try:
        tok1 = BertTokenizerFast.from_pretrained("osiria/bert-italian-cased-ner")
        mod1 = BertForTokenClassification.from_pretrained("osiria/bert-italian-cased-ner")
        pipelines_dict["bert_it"] = (
            pipeline(task="ner", model=mod1, tokenizer=tok1, aggregation_strategy="first", device=device), None)
        logger.info("Loaded NER pipeline: bert_it")
    except Exception as e:
        logger.error(f"Failed to load NER pipeline 'bert_it': {e}")
    try:
        tok2 = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        mod2 = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        pipelines_dict["wiki_multi"] = (
            pipeline(task="ner", model=mod2, tokenizer=tok2, aggregation_strategy="simple", device=device), None)
        logger.info("Loaded NER pipeline: wiki_multi")
    except Exception as e:
        logger.error(f"Failed to load NER pipeline 'wiki_multi': {e}")
    try:
        pipe3 = pipeline(task="ner", model="ZurichNLP/swissbert-ner", aggregation_strategy="simple", device=device)
        pipelines_dict["swiss_it"] = (pipe3, None)
        logger.info("Loaded NER pipeline: swiss_it")
    except Exception as e:
        logger.error(f"Failed to load NER pipeline 'swiss_it': {e}")
    try:
        gl4 = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")
        lbl4 = ["comune", "codice fiscale", "importo", "società", "indirizzo", "persona", "data", "luogo"]
        pipelines_dict["gliner_it"] = (gl4, lbl4)
        logger.info("Loaded NER pipeline: gliner_it")
    except Exception as e:
        logger.error(f"Failed to load NER pipeline 'gliner_it': {e}")
    try:
        gl5 = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        lbl5 = ["person", "organization", "location", "date", "phone number", "email address", "credit card number",
                "national id", "medical record number", "full address", "company", "postal code"]
        pipelines_dict["gliner_pii"] = (gl5, lbl5)
        logger.info("Loaded NER pipeline: gliner_pii")
    except Exception as e:
        logger.error(f"Failed to load NER pipeline 'gliner_pii': {e}")
    return pipelines_dict


def _error_report(message: str, details: Optional[str] = None, raw_output: Optional[str] = None) -> Dict[str, Any]:
    return {
        "found": False, "entities": [], "summary": message,
        "error_details": details if details else "", "raw_output_on_error": raw_output if raw_output else ""
    }


def _robust_json_parser(json_string: str, model_api_id_for_log: str, context_for_log: str = "text analysis") -> \
        Optional[Dict[str, Any]]:
    """
    Tenta di estrarre e parsare un blocco JSON, gestendo output incompleti,
    malformati o in formato dizionario Python.
    """
    s = json_string.strip()
    logger.debug(f"Avvio parsing robusto per {model_api_id_for_log}. Input: {s[:300]}...")

    # ───────────────────────────────────────────────────────────────────
    # TENTATIVO 0: se il JSON è in stile Python con apici singoli,
    # lo trasformo in JSON valido (es. per il caso {'found': false, ...})
    # Nota: ho corretto la regex rimuovendo un backslash di troppo.
    if re.match(r"^'?\{\s*'found':", s):
        # rimuovo eventuali backticks e apici esterni
        s = s.strip('`').strip("'")
        # sostituisco tutti gli apici singoli con doppi
        s = s.replace("'", '"')
        try:
            # Sostituisco anche i booleani stile Python
            s = s.replace(": true", ": true").replace(": false", ": false")
            return json.loads(s)
        except json.JSONDecodeError:
            logger.warning("Tentativo 0 fallito, continuo con fallback esistente.")
    # ───────────────────────────────────────────────────────────────────

    # 1. Estrae il contenuto da un blocco markdown ```json ... ``` se presente
    match_md = re.search(r"```json\s*([\s\S]*)\s*```", s, re.DOTALL | re.IGNORECASE)
    if match_md:
        s = match_md.group(1).strip()

    # 2. Primo tentativo: il caso ideale (JSON valido)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        logger.warning(f"Parsing JSON standard fallito per {model_api_id_for_log}. Tento fallback.")

    # 3. Secondo tentativo: il caso del "dizionario Python" (apici singoli)
    try:
        # ast.literal_eval è sicuro e gestisce apici singoli, booleani Python, etc.
        parsed_dict = ast.literal_eval(s)
        if isinstance(parsed_dict, dict):
            logger.info(f"Parsing riuscito per {model_api_id_for_log} usando ast.literal_eval.")
            return parsed_dict
    except (ValueError, SyntaxError, MemoryError, TypeError):
        logger.warning(f"ast.literal_eval fallito per {model_api_id_for_log}. Tento recupero da stringa troncata.")

    # 3 bis: se ast falla, rifaccio replace singolo -> doppio come ultima spiaggia prima del recupero
    if "'" in s:
        tmp = s.replace("'", '"')
        try:
            return json.loads(tmp)
        except json.JSONDecodeError:
            pass

    # 4. Ultimo tentativo: recupero da JSON troncato
    try:
        # Cerca l'ultima parentesi graffa chiusa, che probabilmente delimita l'ultimo oggetto completo
        last_brace_pos = s.rfind('}')
        if last_brace_pos == -1:
            raise ValueError("Nessuna parentesi graffa di chiusura trovata.")

        # Prendi la sottostringa fino a quel punto
        truncated_part = s[:last_brace_pos + 1]

        # ─────────────────────────────────────────────────────────────
        # Nuovo tentativo: estraggo tutti i singoli oggetti "entity"
        # così da non dipendere da un JSON array completo e valido.
        entity_dicts = re.findall(r'\{[^}]*"type"\s*:\s*"[^"]+"[^}]*\}', truncated_part)
        parsed_entities = []
        for ed in entity_dicts:
            try:
                parsed_entities.append(json.loads(ed))
            except json.JSONDecodeError:
                continue
        if parsed_entities:
            logger.info(
                f"Parsing fallback ENTITY-EXTRACT riuscito per {model_api_id_for_log}, {len(parsed_entities)} entities.")
            return {"found": True, "entities": parsed_entities}
        # ─────────────────────────────────────────────────────────────

        # Trova il punto di partenza dell'array 'entities' per estrarre solo gli oggetti interni
        entities_keyword = '"entities": ['
        entities_start_pos = truncated_part.find(entities_keyword)

        content_to_rebuild = ""
        if entities_start_pos != -1:
            # Estrai solo il contenuto della lista
            content_start = entities_start_pos + len(entities_keyword)
            content_to_rebuild = truncated_part[content_start:]
        else:
            # Se "entities" non è presente, forse l'intero oggetto è stato troncato
            # Prendiamo tutto a partire dalla prima parentesi graffa
            first_brace_pos = truncated_part.find('{')
            if first_brace_pos != -1:
                content_to_rebuild = truncated_part[first_brace_pos:]

        # Ricostruisci un JSON valido. Aggiungiamo le parentesi mancanti.
        # Rimuoviamo una virgola finale se presente
        if content_to_rebuild.endswith(','):
            content_to_rebuild = content_to_rebuild[:-1]

        # La struttura più comune è una lista di entità
        if entities_start_pos != -1:
            reconstructed_json_str = f'{{"found": true, "entities": [{content_to_rebuild}]}}'
        else:  # Se non abbiamo trovato la lista, proviamo a chiudere l'oggetto
            reconstructed_json_str = content_to_rebuild
            if not reconstructed_json_str.endswith('}'):
                reconstructed_json_str += '}'

        logger.info(f"Tentativo di parsing su JSON ricostruito per {model_api_id_for_log}.")
        return json.loads(reconstructed_json_str)

    except Exception as e:
        logger.error(f"Tutti i metodi di parsing sono falliti per {model_api_id_for_log}. Errore finale: {e}")
        logger.error(f"Stringa che ha causato il fallimento definitivo: {json_string[:500]}")
        return None  # Fallimento totale


def generate_report_on_full_text(text: str, model_api_id: str) -> Dict[str, Any]:
    _init_lmstudio(model_api_id)
    if not text or not text.strip():
        logger.warning("Testo vuoto fornito a generate_report_on_full_text.")
        return _error_report("Il testo fornito per l'analisi è vuoto.", "Nessun testo da analizzare.")

    final_report = {}
    raw_llm_output_content = ""

    # --- PRIMA CHIAMATA: ESTRAZIONE ENTITÀ (CON max_tokens e timeout aumentato) ---
    logger.info(f"Invio testo (lunghezza {len(text)}) a LLM `{model_api_id}` per estrazione entità (Passo 1/2).")

    system_message_entities = (
        "You are an AI assistant. Your task is to analyze the given text for sensitive information "
        "and return your findings strictly as a single, valid JSON object. Do not include any explanations or text "
        "outside of this JSON."
        "If no sensitive information is found, the JSON should indicate this with 'found': false and 'entities': []. "
        "The JSON schema MUST be exactly: "
        "{'found': boolean, 'entities':[{'type': 'string', 'text': 'string', 'context': 'string', 'reasoning': "
        "'string', 'source_chunk_info': 'string'}]}."
        "Do NOT add a 'summary' field in this response."
    )

    user_message_entities = (
        "Analyze the following text and return ONLY a single valid JSON object with the exact schema: "
        "{'found': boolean, 'entities':[{'type': 'string', 'text': 'string', 'context': 'string', 'reasoning': "
        "'string', 'source_chunk_info': 'string'}]}."
        "For 'source_chunk_info', use 'full document'. Do NOT generate a summary.\n\n"
        f"Text:\n\"\"\"\n{text}\n\"\"\""
    )

    try:
        response_entities = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system", "content": system_message_entities},
                {"role": "user", "content": user_message_entities}
            ],
            temperature=0.0,
            max_tokens=8192,
            request_timeout=300
        )

        if response_entities.choices and response_entities.choices[0].message and response_entities.choices[
            0].message.content:
            raw_llm_output_content = response_entities.choices[0].message.content.strip()
        else:
            logger.error(f"Risposta LLM (entità) malformata o vuota da {model_api_id}: {str(response_entities)[:500]}")
            return _error_report(f"Risposta LLM ({model_api_id}) malformata o vuota per le entità.",
                                 "Struttura API inattesa.", str(response_entities)[:1000])

        parsed_json = _robust_json_parser(raw_llm_output_content, model_api_id, "entity extraction")

        if parsed_json is None:
            # Errore se il parser robusto fallisce definitivamente
            raise json.JSONDecodeError("Impossibile parsare l'output: " + raw_llm_output_content,
                                       raw_llm_output_content, 0)

        final_report = parsed_json
        final_report.setdefault("found",
                                isinstance(final_report.get("entities"), list) and bool(final_report.get("entities")))
        final_report.setdefault("entities", [])

    except Exception as e:
        logger.error(f"Errore grave durante l'estrazione delle entità con {model_api_id}: {type(e).__name__} - {e}",
                     exc_info=True)
        return _error_report(f"Errore grave durante l'estrazione delle entità ({type(e).__name__}).", str(e),
                             raw_llm_output_content)

    # --- SECONDA CHIAMATA: GENERAZIONE RIASSUNTO ---
    logger.info(f"Invio testo a LLM `{model_api_id}` per il riassunto (Passo 2/2).")

    system_message_summary = ("You are a helpful AI assistant. Your task is to provide a brief, concise summary of the "
                              "provided text. Output only the summary text, without any introductory phrases.")
    user_message_summary = f"Provide a brief summary of the following text:\n\nText:\n\"\"\"\n{text}\n\"\"\""

    summary_text = "Riassunto non generato a causa di un errore."
    try:
        response_summary = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": user_message_summary}
            ],
            temperature=0.1,
            max_tokens=512,
            request_timeout=300  # Timeout aumentato anche qui per coerenza e robustezza
        )
        if response_summary.choices and response_summary.choices[0].message and response_summary.choices[
            0].message.content:
            summary_text = response_summary.choices[0].message.content.strip()
        else:
            logger.warning(f"Risposta LLM per il riassunto malformata o vuota da {model_api_id}.")

    except Exception as e_summary:
        logger.error(f"Errore durante la generazione del riassunto con {model_api_id}: {e_summary}", exc_info=True)
        summary_text = f"Errore durante la generazione del riassunto: {type(e_summary).__name__}"

    # --- UNIONE DEI RISULTATI ---
    final_report["summary"] = summary_text

    logger.info("Report finale combinato (entità + riassunto) creato con successo.")
    return final_report








def edit_document(text: str,
                  report: Union[str, dict],
                  model_api_id: str) -> str:
    """
    Modifica un testo per anonimizzare le PII basandosi su un report JSON.
    Accetta il report sia come dict (che serializza internamente) sia già come JSON-string.
    Fa fino a 2 retry in caso di AttributeError interno al client OpenAI,
    e se entrambi falliscono utilizza un fallback regex-only basato sul report.
    """
    _init_lmstudio(model_api_id)

    # Serializzo il report se arriva come dict
    if isinstance(report, dict):
        try:
            report_str = json.dumps(report, ensure_ascii=False)
        except Exception as e_dump:
            logger.warning(f"edit_document: impossibile serializzare report dict: {e_dump}")
            report_str = str(report)
    else:
        report_str = report

    system_prompt = (
        "You are an AI assistant. Your task is to remove sensitive information from the provided text, "
        "based on the given JSON report. Replace sensitive items with meaningful placeholders "
        "(e.g., [PERSON_NAME], [ADDRESS]). Do not summarize or correct grammar. "
        f"The JSON report detailing sensitive information is: {report_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Modify the following document based on the report:\n\nText:\n\"\"\"\n{text}\n\"\"\""}
    ]

    # Retry loop per AttributeError interno
    for attempt in range(2):
        try:
            resp = openai.ChatCompletion.create(
                model=model_api_id,
                messages=messages,
                temperature=0.0,
                max_tokens=len(text.split()) + 500
            )
            return resp.choices[0].message.content.strip()

        except AttributeError as e_attr:
            logger.warning(f"edit_document: AttributeError ChatCompletion (tentativo {attempt + 1}/2): {e_attr}")
            if attempt == 1:
                logger.error("edit_document: retry falliti, passo al fallback locale.")
        except openai.error.APIError as e_api:
            logger.error(f"OpenAI APIError during editing (model: {model_api_id}): {e_api}")
            return f"Error during document editing (API Error): {e_api}\nOriginal text was:\n{text}"
        except Exception as e:
            logger.error(
                f"Unexpected error during document editing (model: {model_api_id}): {e}",
                exc_info=True
            )
            return f"Unexpected error during document editing: {e}\nOriginal text was:\n{text}"

    # --- Fallback locale: sostituzione diretta delle entità dal report ---
    try:
        report_dict = report if isinstance(report, dict) else json.loads(report)
    except Exception:
        return f"Unable to parse report JSON for fallback.\nOriginal text was:\n{text}"

    fallback = text
    entities = report_dict.get("entities", [])
    for ent in entities:
        if isinstance(ent, dict) and ent.get("text"):
            term = ent["text"]
            typ = ent.get("type", "").upper() or "REDACTED"
            placeholder = f"[{typ}]"
            fallback = fallback.replace(term, placeholder)

    return fallback


def sensitive_informations(report_str: str, model_api_id: str) -> str:
    _init_lmstudio(model_api_id)
    system_prompt = (
        "You are an AI assistant. Based on the provided JSON report of sensitive information (which includes a "
        "'reasoning' field for each entity),"
        "list the contexts for each piece of sensitive information and briefly incorporate or refer to the reasoning "
        "provided."
        "Format your answer clearly, for example: "
        "'Sensitive Information: [Value of 'text' field from an entity]\n"
        "Type: [Value of 'type' field from an entity]\n"
        "Context: [Value of 'context' field from an entity]\n"
        "Reasoning from report: [Value of 'reasoning' field from an entity]\n\n"
        "---Next Entity---'\n"
        "If the report indicates no sensitive information was found, state that clearly."
        f"The JSON report is: {report_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Provide the sensitive contexts and reasoning based on the report."}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model_api_id, messages=messages, temperature=0.0, max_tokens=2048
        )
        contexts_text = response.choices[0].message.content.strip()
        return contexts_text
    except openai.error.APIError as e:
        logger.error(f"OpenAI APIError for sensitive contexts (model: {model_api_id}): {e}")
        return f"Error generating sensitive contexts (API Error): {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred while generating sensitive contexts (model: {model_api_id}): {e}",
                     exc_info=True)
        return f"Unexpected error generating sensitive contexts: {e}"


def extract_entities(text: str) -> list:
    pipelines_dict = get_ner_pipelines()
    all_entities = []
    if not text.strip():
        logger.warning("Input text for NER is empty. Skipping entity extraction.")
        return all_entities

    for name, (ner_pipe, labels) in pipelines_dict.items():
        logger.info(f"Extracting entities with NER model: {name}")
        try:
            if labels is None:  # Transformers pipeline
                results = ner_pipe(text)
                current_entities = []
                # Gestisci output che potrebbe essere lista di liste (per chunking interno di pipeline) o lista di dict
                if results and isinstance(results, list) and results[0] and isinstance(results[0], list):
                    for chunk_results in results:
                        for ent in chunk_results:
                            current_entities.append({
                                "model_name": name,
                                "type": ent.get("entity_group", ent.get("entity", "N/A")),  # entity_group o entity
                                "text": ent.get("word", "N/A"),
                                "score": round(float(ent.get("score", 0.0)), 4)
                            })
                else:  # Lista di dict
                    for ent in results:
                        current_entities.append({
                            "model_name": name,
                            "type": ent.get("entity_group", ent.get("entity", "N/A")),
                            "text": ent.get("word", "N/A"),
                            "score": round(float(ent.get("score", 0.0)), 4)
                        })
                all_entities.extend(current_entities)
            else:
                if not all(isinstance(label, str) for label in labels):
                    logger.warning(f"Labels for GLiNER model {name} are not all strings: {labels}. Skipping.")
                    continue
                if hasattr(ner_pipe, 'predict_entities'):
                    gliner_results = ner_pipe.predict_entities(text, labels, threshold=0.5)
                    current_entities_gliner = [{
                        "model_name": name, "type": ent.get("label", "N/A"),
                        "text": ent.get("text", "N/A"), "score": round(float(ent.get("score", 0.0)), 4)
                    } for ent in gliner_results]
                    all_entities.extend(current_entities_gliner)
                else:
                    logger.error(f"GLiNER model {name} does not have 'predict_entities' method.")
        except Exception as e:
            logger.error(f"Error during NER with {name}: {e}", exc_info=True)
            continue
    return all_entities
