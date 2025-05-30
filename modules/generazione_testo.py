import json
import socket
import subprocess
import time
import re
import ast
from functools import lru_cache

import openai
from gliner import GLiNER
from torch import cuda
from transformers import pipeline, BertTokenizerFast, BertForTokenClassification, AutoTokenizer, \
    AutoModelForTokenClassification

openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"


@lru_cache(maxsize=None)
def _init_lmstudio(model_api_id: str, port: int = 1234):
    # ... (implementazione di _init_lmstudio come nella versione precedente del Canvas)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_running = False
    try:
        sock.connect(("127.0.0.1", port))
        server_running = True
        print(f"LMStudio server is already running on port {port}")
    except (ConnectionRefusedError, OSError):
        print(f"LMStudio server not found on port {port}. Attempting to start server...")
        try:
            subprocess.Popen(
                ["lms", "server", "start", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(15)
            sock_check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock_check.connect(("127.0.0.1", port))
                print(f"LMStudio server started successfully on port {port}.")
                server_running = True
            except (ConnectionRefusedError, OSError):
                print(f"Failed to connect to LMStudio server after attempting to start it on port {port}.")
            finally:
                sock_check.close()
        except FileNotFoundError:
            print("ERROR: 'lms' command not found. Make sure LMStudio CLI is installed and in your system's PATH.")
            return
        except Exception as e:
            print(f"An error occurred while trying to start LMStudio server: {e}")
            return
    finally:
        sock.close()

    if not server_running:
        print("Skipping model load as LMStudio server is not running or could not be started.")
        return

    model_to_load_cli = model_api_id
    print(
        f"Attempting to load model via CLI: '{model_to_load_cli}' using 'lms load {model_to_load_cli}'. This call is memoized.")
    try:
        command_list = ["lms", "load", model_to_load_cli]
        print(f"Executing command: {' '.join(command_list)} with a 90-second timeout.")
        process_result = subprocess.run(
            command_list,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=90
        )
        output_stdout = process_result.stdout.strip() if process_result.stdout else ""
        output_stderr = process_result.stderr.strip() if process_result.stderr else ""
        print(f"STDOUT from 'lms load {model_to_load_cli}':\n{output_stdout}")
        if output_stderr:
            print(f"STDERR from 'lms load {model_to_load_cli}':\n{output_stderr}")
        if process_result.returncode == 0:
            combined_output = output_stdout.lower() + " " + output_stderr.lower()
            if "already loaded" in combined_output or \
                    "model loaded successfully" in combined_output or \
                    model_api_id.lower() in combined_output:
                print(f"Model '{model_api_id}' appears to be loaded or loading initiated successfully for API usage.")
            else:
                print(
                    f"Warning: 'lms load {model_to_load_cli}' command succeeded (exit code 0), but success message or API identifier '{model_api_id}' not clearly confirmed in combined output. Check LMStudio UI or `lms ls`.")
        else:
            print(
                f"Error during 'lms load {model_to_load_cli}': Command failed with exit code {process_result.returncode}.")
            print(f"Please ensure the model identifier '{model_to_load_cli}' is correct for 'lms load'.")
            print("This should be the identifier LMStudio uses (e.g., 'repository/model_name' or 'model_name.gguf').")
            print(
                f"The API will expect '{model_api_id}'. If 'lms load' fails, the model might not be usable by the API.")
            print(f"Try running `lms load \"{model_to_load_cli}\"` manually in your terminal to diagnose.")
        time.sleep(5)
    except subprocess.TimeoutExpired:
        print(f"Timeout: 'lms load {model_to_load_cli}' took longer than 90 seconds to respond.")
        print("The model might be very large, LMStudio might be stuck, or there could be other issues.")
        print(
            "Check the LMStudio UI for model status. The script will proceed, but API calls to this model might fail if it's not actually loaded.")
    except FileNotFoundError:
        print("ERROR: 'lms' command not found. Make sure LMStudio CLI is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during 'lms load' model loading: {e}")


@lru_cache(maxsize=None)
def get_ner_pipelines():
    device = 0 if cuda.is_available() else -1
    print(f"Initializing NER pipelines on device: {'GPU' if device == 0 else 'CPU'}")
    pipelines_dict = {}
    try:
        tok1 = BertTokenizerFast.from_pretrained("osiria/bert-italian-cased-ner")
        mod1 = BertForTokenClassification.from_pretrained("osiria/bert-italian-cased-ner")
        pipelines_dict["bert_it"] = (
            pipeline(task="ner", model=mod1, tokenizer=tok1, aggregation_strategy="first", device=device), None)
        print("Loaded NER pipeline: bert_it")
    except Exception as e:
        print(f"Failed to load NER pipeline 'bert_it': {e}")
    try:
        tok2 = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        mod2 = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        pipelines_dict["wiki_multi"] = (
            pipeline(task="ner", model=mod2, tokenizer=tok2, aggregation_strategy="simple", device=device), None)
        print("Loaded NER pipeline: wiki_multi")
    except Exception as e:
        print(f"Failed to load NER pipeline 'wiki_multi': {e}")
    try:
        pipe3 = pipeline(task="ner", model="ZurichNLP/swissbert-ner", aggregation_strategy="simple", device=device)
        pipelines_dict["swiss_it"] = (pipe3, None)
        print("Loaded NER pipeline: swiss_it")
    except Exception as e:
        print(f"Failed to load NER pipeline 'swiss_it': {e}")
    try:
        gl4 = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")
        lbl4 = ["comune", "codice fiscale", "importo", "società", "indirizzo", "persona", "data", "luogo"]
        pipelines_dict["gliner_it"] = (gl4, lbl4)
        print("Loaded NER pipeline: gliner_it")
    except Exception as e:
        print(f"Failed to load NER pipeline 'gliner_it': {e}")
    try:
        gl5 = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        lbl5 = ["person", "organization", "location", "date", "phone number", "email address", "credit card number",
                "national id", "medical record number", "full address", "company", "postal code"]
        pipelines_dict["gliner_pii"] = (gl5, lbl5)
        print("Loaded NER pipeline: gliner_pii")
    except Exception as e:
        print(f"Failed to load NER pipeline 'gliner_pii': {e}")
    return pipelines_dict


def generate_report(text: str, model_api_id: str) -> dict:
    _init_lmstudio(model_api_id)

    # Il troncamento è stato rimosso da qui. La gestione dei chunk avverrà in streamlit_app.py

    system_message_content = (
        "You are an AI assistant. Your sole task is to analyze the given text for sensitive information "
        "and return your findings strictly as a JSON object. Do not include any explanations, apologies, "
        "or any text outside of the JSON object. The user has consented to data processing. "
        "If no sensitive information is found, the JSON should indicate this appropriately "
        "(e.g., 'found': false, 'entities': []). The JSON schema must be: "
        "{'found': boolean, 'entities':[{'type': string, 'text': string, 'context': string, 'reasoning': string}, "
        "...], 'summary': string}."
        "The 'text' field within each entity object MUST be a single string, not an array or list. "
        "The 'reasoning' field should briefly explain why the identified 'text' is considered sensitive in the given "
        "'context' or why it matches the 'type'."
        "All keys and string values in the JSON output MUST use double quotes."
    )
    system = {"role": "system", "content": system_message_content}

    user_message_content = (
        "Analyze the following text and return a single JSON object with the schema: "
        "{'found': true|false, 'entities':[{'type': string, 'text': string, 'context': string, 'reasoning': string}, "
        "...], 'summary': string}."
        "Ensure the output is only this JSON object, with all keys and string values using double quotes. "
        "The 'text' field for each entity must be a single string. "
        "For each entity, provide a brief 'reasoning' explaining its sensitivity or classification. "
        "Provide a brief summary of the text in the 'summary' field.\n\n"
        f"Text:\n\"\"\"\n{text}\n\"\"\""
    )
    user = {"role": "user", "content": user_message_content}

    resp = None
    try:
        print(f"DEBUG (generate_report): Sto per chiamare openai.ChatCompletion.create per il modello {model_api_id}")
        print(f"DEBUG (generate_report): Lunghezza del testo EFFETTIVAMENTE inviato: {len(text)} caratteri")
        # Non stampare l'intero testo se è un chunk, ma solo una parte per conferma
        print(f"DEBUG (generate_report): Inizio del testo EFFETTIVAMENTE inviato (primi 200 caratteri): {text[:200]}")

        try:
            resp_candidate = openai.ChatCompletion.create(
                model=model_api_id,
                messages=[system, user],
                temperature=0.0,
                max_tokens=3072,  # Questo potrebbe dover essere aggiustato in base alla lunghezza del chunk
            )
            print(f"DEBUG (generate_report): Chiamata openai.ChatCompletion.create COMPLETATA per {model_api_id}.")
            resp = resp_candidate
        except AttributeError as ae_during_create:
            print(
                f"CRITICAL DEBUG (generate_report): AttributeError ('{ae_during_create}') si è verificato *DURANTE* "
                f"la chiamata openai.ChatCompletion.create per {model_api_id}")
            if hasattr(ae_during_create, '__cause__') and ae_during_create.__cause__:
                print(
                    f"CRITICAL DEBUG (generate_report): Causa dell'AttributeError: {type(ae_during_create.__cause__).__name__} - {ae_during_create.__cause__}")
                if hasattr(ae_during_create.__cause__, 'http_body'):  # Improbabile per AttributeError ma check
                    print(
                        f"CRITICAL DEBUG (generate_report): Causa dell'AttributeError (http_body): {ae_during_create.__cause__.http_body}")
            elif hasattr(ae_during_create, '__context__') and ae_during_create.__context__:
                print(
                    f"CRITICAL DEBUG (generate_report): Contesto dell'AttributeError: {type(ae_during_create.__context__).__name__} - {ae_during_create.__context__}")
            raise
        except openai.error.APIError as api_err_during_create:
            print(
                f"CRITICAL DEBUG (generate_report): OpenAI APIError ({type(api_err_during_create).__name__}: {api_err_during_create}) si è verificata *DURANTE* la chiamata openai.ChatCompletion.create per {model_api_id}")
            if hasattr(api_err_during_create, 'http_body') and api_err_during_create.http_body:
                try:
                    body_content = api_err_during_create.http_body
                    if isinstance(body_content, bytes):
                        body_content = body_content.decode('utf-8', errors='replace')
                    print(f"CRITICAL DEBUG (generate_report): OpenAI APIError HTTP body: {body_content}")
                except Exception as e_decode:
                    print(f"CRITICAL DEBUG (generate_report): Errore nel decodificare http_body: {e_decode}")
            if hasattr(api_err_during_create, 'json_body') and api_err_during_create.json_body:
                print(f"CRITICAL DEBUG (generate_report): OpenAI APIError JSON body: {api_err_during_create.json_body}")
            raise
        except Exception as e_during_create:
            print(
                f"CRITICAL DEBUG (generate_report): Un'eccezione generica ({type(e_during_create).__name__}: {e_during_create}) si è verificata *DURANTE* la chiamata openai.ChatCompletion.create per {model_api_id}")
            raise

        print(f"DEBUG (generate_report): Tipo dell'oggetto resp per {model_api_id}: {type(resp)}")
        print(f"DEBUG (generate_report): Oggetto resp completo per {model_api_id} (primi 2000 caratteri):")
        print(str(resp)[:2000])

        if hasattr(resp, 'choices') and resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            # ... (altri controlli di debug sulla struttura di resp come prima) ...
        else:
            print(f"DEBUG (generate_report): ERRORE - resp NON HA 'choices' o è vuoto/malformato per {model_api_id}")

        raw_llm_output = ""
        if hasattr(resp, 'choices') and resp.choices and \
                hasattr(resp.choices[0], 'message') and \
                hasattr(resp.choices[0].message, 'content') and \
                isinstance(resp.choices[0].message.content, str):
            raw_llm_output = resp.choices[0].message.content.strip()
        else:
            print(
                f"Error (generate_report): Impossibile estrarre raw_llm_output a causa di una struttura di 'resp' ({type(resp)}) inattesa o tipo di contenuto non stringa per il modello {model_api_id}.")

        if not raw_llm_output:
            print(
                f"Error (generate_report): raw_llm_output è vuoto per il modello {model_api_id}. L'oggetto resp era: {str(resp)[:1000]}")
            return {"found": False, "entities": [],
                    "summary": f"Error: Risposta vuota o contenuto non estraibile dall'LLM per {model_api_id}.",
                    "raw_output": str(resp)}

        print(
            f"Raw LLM response for report generation (model: {model_api_id}):\n{raw_llm_output}\n--------------------")

        json_str_to_parse = None
        match_md_json = re.search(r"```json\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL)
        if match_md_json:
            json_str_to_parse = match_md_json.group(1)
            print("Extracted JSON from ```json ... ``` block.")
        else:
            start_index = raw_llm_output.find('{')
            end_index = raw_llm_output.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str_to_parse = raw_llm_output[start_index: end_index + 1]
                print("Extracted JSON using find('{') and rfind('}').")
            else:
                print(
                    f"No JSON object could be reliably extracted from LLM response. Raw output:\n{raw_llm_output}\n--------------------")
                return {"found": False, "entities": [], "summary": "Error: No JSON object in LLM response.",
                        "raw_output": raw_llm_output}
        try:
            return json.loads(json_str_to_parse)
        except json.JSONDecodeError as e_direct:
            print(f"Direct json.loads failed: {e_direct}. Attempting to fix common issues...")
            try:
                evaluated_data = ast.literal_eval(json_str_to_parse)
                print("Successfully parsed with ast.literal_eval. Re-dumping to JSON standard.")
                return json.loads(json.dumps(evaluated_data))
            except (ValueError, SyntaxError, TypeError) as e_ast:
                print(f"ast.literal_eval also failed: {e_ast}. Attempting regex cleaning for JSON...")
                try:
                    json_str_fixed_quotes = re.sub(r"(?<!\\)'", "\"", json_str_to_parse)
                    json_str_fixed_commas = re.sub(r",\s*(\}|\])", r"\1", json_str_fixed_quotes)
                    json_str_cleaned_final = re.sub(r"//.*?\n", "\n", json_str_fixed_commas)
                    print(
                        f"Attempting json.loads on heuristically cleaned string:\n{json_str_cleaned_final}\n--------------------")
                    return json.loads(json_str_cleaned_final)
                except json.JSONDecodeError as e_final_clean:
                    print(f"Failed to parse even after heuristic cleaning: {e_final_clean}")
                    print(f"Problematic JSON string (original) was:\n{json_str_to_parse}\n--------------------")
                    return {"found": False, "entities": [],
                            "summary": f"Error: Could not parse LLM JSON response. Details: {e_final_clean}",
                            "raw_output": raw_llm_output}
            except Exception as e_unknown_ast:
                print(f"An unexpected error occurred during ast.literal_eval or subsequent json.dumps: {e_unknown_ast}")
                return {"found": False, "entities": [],
                        "summary": f"Error: Processing after ast.literal_eval failed. {e_unknown_ast}",
                        "raw_output": raw_llm_output}

    except openai.error.APIError as api_e:
        print(f"OpenAI APIError ({type(api_e).__name__}) during report generation (model: {model_api_id}): {api_e}")
        http_body_info = "N/A"
        if hasattr(api_e, 'http_body') and api_e.http_body:
            try:
                body_content = api_e.http_body
                if isinstance(body_content, bytes):
                    body_content = body_content.decode('utf-8', errors='replace')
                http_body_info = body_content
            except Exception as e_decode_outer:
                http_body_info = f"[Error decoding http_body: {e_decode_outer}]"
            print(f"OpenAI APIError HTTP body: {http_body_info}")
        json_body_info = "N/A"
        if hasattr(api_e, 'json_body') and api_e.json_body:
            json_body_info = str(api_e.json_body)
            print(f"OpenAI APIError JSON body: {json_body_info}")
        return {"found": False, "entities": [],
                "summary": f"Error: OpenAI API Error ({type(api_e).__name__}) for model '{model_api_id}'. HTTP Body: {http_body_info[:200]}",
                "raw_output": f"APIError: {api_e}, HTTP Body: {http_body_info}, JSON Body: {json_body_info}"}
    except AttributeError as ae:
        print(f"Outer AttributeError during report generation (model: {model_api_id}): {ae}")
        print(
            f"This likely means the structure of 'resp' from openai.ChatCompletion.create was not as expected, or 'resp' was not assigned.")
        resp_val_str = str(resp)[:1000] if resp is not None else "resp was None or not assigned"
        print(f"Value of 'resp' when AttributeError occurred (or if it was assigned): {resp_val_str}")
        return {"found": False, "entities": [],
                "summary": f"Error: AttributeError - {ae}. Unexpected API response structure.",
                "raw_output": resp_val_str}
    except Exception as e:
        print(
            f"An unexpected error occurred during report generation (model: {model_api_id}): {type(e).__name__} - {e}")
        resp_val_str_general = str(resp)[:1000] if resp is not None else "resp was None or not assigned"
        return {"found": False, "entities": [],
                "summary": f"Error: An unexpected error occurred ({type(e).__name__}). {e}",
                "raw_output": resp_val_str_general}


def edit_document(text: str, report_str: str, model_api_id: str) -> str:
    _init_lmstudio(model_api_id)
    # ... (implementazione come prima)
    system_prompt = (
        "You are an AI assistant. Your task is to remove sensitive information from the provided text, "
        "based on the given JSON report. Replace sensitive items with meaningful placeholders (e.g., [PERSON_NAME], [ADDRESS]). "
        "Do not summarize or correct grammar. Only output the modified text. "
        f"The JSON report detailing sensitive information is: {report_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Modify the following document based on the report:\n\nText:\n\"\"\"\n{text}\n\"\"\""},
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model_api_id,
            messages=messages,
            temperature=0.0,
            max_tokens=len(text.split()) + 500  # Sufficient for modified text
        )
        edited_text = response.choices[0].message.content.strip()
        print(f"Raw LLM response for document editing (model: {model_api_id}):\n{edited_text}\n--------------------")
        return edited_text
    except openai.error.APIError as e:  # Catch generic API errors first
        print(f"OpenAI APIError during editing (model: {model_api_id}): {e}")
        if hasattr(e, 'http_body'): print(f"HTTP body: {e.http_body}")
        return f"Error during document editing (API Error): {e}\nOriginal text was:\n{text}"
    except Exception as e:
        print(f"An unexpected error occurred during document editing (model: {model_api_id}): {e}")
        return f"Unexpected error during document editing: {e}\nOriginal text was:\n{text}"


def sensitive_informations(report_str: str, model_api_id: str) -> str:
    _init_lmstudio(model_api_id)
    # ... (implementazione come prima)
    system_prompt = (
        "You are an AI assistant. Based on the provided JSON report of sensitive information (which includes a 'reasoning' field for each entity), "
        "list the contexts for each piece of sensitive information and briefly incorporate or refer to the reasoning provided. "
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
            model=model_api_id,
            messages=messages,
            temperature=0.0,
            max_tokens=2048
        )
        contexts_text = response.choices[0].message.content.strip()
        print(
            f"Raw LLM response for sensitive contexts (model: {model_api_id}):\n{contexts_text}\n--------------------")
        return contexts_text
    except openai.error.APIError as e:  # Catch generic API errors first
        print(f"OpenAI APIError for sensitive contexts (model: {model_api_id}): {e}")
        if hasattr(e, 'http_body'): print(f"HTTP body: {e.http_body}")
        return f"Error generating sensitive contexts (API Error): {e}"
    except Exception as e:
        print(f"An unexpected error occurred while generating sensitive contexts (model: {model_api_id}): {e}")
        return f"Unexpected error generating sensitive contexts: {e}"


def extract_entities(text: str) -> list:
    # ... (implementazione di extract_entities come nella versione precedente del Canvas)
    pipelines_dict = get_ner_pipelines()
    all_entities = []
    if not text.strip():
        print("Input text for NER is empty. Skipping entity extraction.")
        return all_entities

    for name, (ner_pipe, labels) in pipelines_dict.items():
        print(f"Extracting entities with NER model: {name}")
        try:
            if labels is None:
                results = ner_pipe(text)
                current_entities = []
                if results and isinstance(results, list) and results[0] and isinstance(results[0], list):
                    for chunk_results in results:
                        for ent in chunk_results:
                            current_entities.append({
                                "model_name": name,
                                "type": ent.get("entity_group", ent.get("entity", "N/A")),
                                "text": ent.get("word", "N/A"),
                                "score": round(float(ent.get("score", 0.0)), 4)
                            })
                else:
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
                    print(
                        f"Warning: Labels for GLiNER model {name} are not all strings: {labels}. Skipping this model.")
                    continue
                if hasattr(ner_pipe, 'predict_entities'):
                    gliner_results = ner_pipe.predict_entities(text, labels, threshold=0.5)
                    current_entities_gliner = []
                    for ent in gliner_results:
                        current_entities_gliner.append({
                            "model_name": name,
                            "type": ent.get("label", "N/A"),
                            "text": ent.get("text", "N/A"),
                            "score": round(float(ent.get("score", 0.0)), 4)
                        })
                    all_entities.extend(current_entities_gliner)
                else:
                    print(f"Error: GLiNER model {name} does not have 'predict_entities' method.")
        except Exception as e:
            print(f"Error during NER with {name}: {e}")
            continue
    return all_entities
