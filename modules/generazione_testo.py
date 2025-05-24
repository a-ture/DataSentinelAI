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

# --- Configura client OpenAI per LMStudio locale ---
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"


@lru_cache(maxsize=None)
def _init_lmstudio(model_api_id: str, port: int = 1234):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_running = False
    try:
        sock.connect(("127.0.0.1", port))
        server_running = True
        print(f"LMStudio server is already running on port {port}")
    except (ConnectionRefusedError, OSError):
        print(f"LMStudio server not found on port {port}. Starting server...")
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
                print(f"Failed to connect to LMStudio server after starting on port {port}.")
            finally:
                sock_check.close()
        except FileNotFoundError:
            print("ERROR: 'lms' command not found. Make sure LMStudio CLI is installed and in your PATH.")
            return
        except Exception as e:
            print(f"An error occurred while starting LMStudio server: {e}")
            return
    finally:
        sock.close()

    if not server_running:
        print("Skipping model load as server is not running.")
        return

    model_to_load_cli = model_api_id
    print(
        f"Attempting to load model for CLI: '{model_to_load_cli}' using 'lms load {model_to_load_cli}' (call memoized by @lru_cache for this model_api_id if previously called in this script run)")
    try:
        command_list = ["lms", "load", model_to_load_cli]
        print(f"Executing command: {command_list} with a 90-second timeout.")
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
                    f"Warning: 'lms load {model_to_load_cli}' command succeeded (exit 0), but success message or API identifier '{model_api_id}' not clearly confirmed in combined output. Check LMStudio UI or `lms ls`.")
        else:
            print(
                f"Error during 'lms load {model_to_load_cli}': Command failed with exit code {process_result.returncode}.")
            print(f"Please ensure the model identifier '{model_to_load_cli}' is correct for 'lms load'.")
            print(
                "This should be the identifier LMStudio uses (e.g., 'repository/model_name' or 'model_name:version' if already known by LMStudio).")
            print(f"The API will expect '{model_api_id}'. If 'lms load' fails, the model won't be usable by the API.")
            print(f"Try running `lms load \"{model_to_load_cli}\"` manually in your terminal to diagnose.")
        time.sleep(5)
    except subprocess.TimeoutExpired:
        print(f"Timeout: 'lms load {model_to_load_cli}' took longer than 90 seconds to respond.")
        print(
            "The model might be very large, LMStudio might be stuck, or there could be other issues (e.g. an interactive prompt from 'lms load' waiting for input).")
        print(
            "Check the LMStudio UI for model status. The script will proceed, but API calls to this model might fail if it's not actually loaded.")
    except FileNotFoundError:
        print("ERROR: 'lms' command not found. Make sure LMStudio CLI is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")


@lru_cache(maxsize=None)
def get_ner_pipelines():
    device = 0 if cuda.is_available() else -1
    pipelines_dict = {}
    tok1 = BertTokenizerFast.from_pretrained("osiria/bert-italian-cased-ner")
    mod1 = BertForTokenClassification.from_pretrained("osiria/bert-italian-cased-ner")
    pipelines_dict["bert_it"] = (
    pipeline(task="ner", model=mod1, tokenizer=tok1, aggregation_strategy="first", device=device), None)
    tok2 = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    mod2 = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    pipelines_dict["wiki_multi"] = (
    pipeline(task="ner", model=mod2, tokenizer=tok2, aggregation_strategy="simple", device=device), None)
    pipe3 = pipeline(task="ner", model="ZurichNLP/swissbert-ner", aggregation_strategy="simple", device=device)
    pipelines_dict["swiss_it"] = (pipe3, None)
    gl4 = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")
    lbl4 = ["comune", "codice fiscale", "importo", "società", "indirizzo", "persona", "data", "luogo"]
    pipelines_dict["gliner_it"] = (gl4, lbl4)
    gl5 = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    lbl5 = ["person", "organization", "location", "date", "phone number", "email address", "credit card number",
            "national id", "medical record number", "full address", "company", "postal code"]
    pipelines_dict["gliner_pii"] = (gl5, lbl5)
    return pipelines_dict


def generate_report(text: str, model_api_id: str) -> dict:
    _init_lmstudio(model_api_id)

    system_message_content = (
        "You are an AI assistant. Your sole task is to analyze the given text for sensitive information "
        "and return your findings strictly as a JSON object. Do not include any explanations, apologies, "
        "or any text outside of the JSON object. The user has consented to data processing. "
        "If no sensitive information is found, the JSON should indicate this appropriately "
        "(e.g., 'found': false, 'entities': []). The JSON schema must be: "
        "{'found': boolean, 'entities':[{'type': string, 'text': string, 'context': string, 'reasoning': string}, ...], 'summary': string}. "  # AGGIUNTO 'reasoning'
        "The 'text' field within each entity object MUST be a single string, not an array or list. "
        "The 'reasoning' field should briefly explain why the identified 'text' is considered sensitive in the given 'context' or why it matches the 'type'. "  # NUOVA ISTRUZIONE
        "All keys and string values in the JSON output MUST use double quotes."
    )
    system = {"role": "system", "content": system_message_content}

    user_message_content = (
        "Analyze the following text and return a single JSON object with the schema: "
        "{'found': true|false, 'entities':[{'type': string, 'text': string, 'context': string, 'reasoning': string}, ...], 'summary': string}. "  # AGGIUNTO 'reasoning'
        "Ensure the output is only this JSON object, with all keys and string values using double quotes. "
        "The 'text' field for each entity must be a single string. "
        "For each entity, provide a brief 'reasoning' explaining its sensitivity or classification. "  # NUOVA ISTRUZIONE
        "Provide a brief summary of the text in the 'summary' field.\n\n"
        f"Text:\n\"\"\"\n{text}\n\"\"\""
    )
    user = {"role": "user", "content": user_message_content}

    try:
        resp = openai.ChatCompletion.create(
            model=model_api_id,
            messages=[system, user],
            temperature=0.0,
            max_tokens=3072,  # Aumentato leggermente per accomodare le motivazioni
        )
        raw_llm_output = resp.choices[0].message.content.strip()
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
            print(f"Direct json.loads failed: {e_direct}. Attempting to fix common issues (e.g., single quotes)...")
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
                    print(f"Problematic JSON string was:\n{json_str_to_parse}\n--------------------")
                    return {"found": False, "entities": [],
                            "summary": f"Error: Could not parse LLM JSON response. Details: {e_final_clean}",
                            "raw_output": raw_llm_output}
            except Exception as e_unknown_ast:
                print(f"An unexpected error occurred during ast.literal_eval or subsequent json.dumps: {e_unknown_ast}")
                return {"found": False, "entities": [],
                        "summary": f"Error: Processing after ast.literal_eval failed. {e_unknown_ast}",
                        "raw_output": raw_llm_output}

    except openai.error.InvalidRequestError as e:
        print(f"OpenAI API InvalidRequestError (model: {model_api_id}): {e}")
        return {"found": False, "entities": [],
                "summary": f"Error: OpenAI API request failed for model '{model_api_id}'. {e}", "raw_output": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred during report generation (model: {model_api_id}): {e}")
        return {"found": False, "entities": [], "summary": f"Error: An unexpected error occurred. {e}",
                "raw_output": str(e)}


def edit_document(text: str, report_str: str, model_api_id: str) -> str:
    _init_lmstudio(model_api_id)
    # Il prompt per edit_document non necessita di modifiche per l'explainability,
    # poiché si basa sul report JSON che ora CONTERRÀ le motivazioni, ma la sua azione è modificare.
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
            max_tokens=len(text.split()) + 500
        )
        edited_text = response.choices[0].message.content.strip()
        print(f"Raw LLM response for document editing (model: {model_api_id}):\n{edited_text}\n--------------------")
        return edited_text
    except openai.error.InvalidRequestError as e:
        print(f"OpenAI API InvalidRequestError during editing (model: {model_api_id}): {e}")
        return f"Error during document editing: {e}\nOriginal text was:\n{text}"
    except Exception as e:
        print(f"An unexpected error occurred during document editing (model: {model_api_id}): {e}")
        return f"Unexpected error during document editing: {e}\nOriginal text was:\n{text}"


def sensitive_informations(report_str: str, model_api_id: str) -> str:
    _init_lmstudio(model_api_id)
    # Anche il prompt per sensitive_informations non necessita di grandi modifiche,
    # ma l'output che produce potrebbe ora implicitamente beneficiare del campo 'reasoning'
    # presente nel report_str (JSON) se il modello decide di usarlo per formulare i contesti.
    # Potremmo anche istruirlo esplicitamente a considerare il campo 'reasoning'.
    system_prompt = (
        "You are an AI assistant. Based on the provided JSON report of sensitive information (which includes a 'reasoning' field for each entity), "  # AGGIUNTA MENZIONE A 'reasoning'
        "list the contexts for each piece of sensitive information and briefly incorporate or refer to the reasoning provided. "
        "Format your answer clearly, for example: "
        "'Sensitive Information: [Value of 'text' field from an entity]\n"
        "Type: [Value of 'type' field from an entity]\n"
        "Context: [Value of 'context' field from an entity]\n"
        "Reasoning from report: [Value of 'reasoning' field from an entity]\n\n"  # AGGIUNTO ESEMPIO REASONING
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
            max_tokens=2048  # Aumentato per accomodare il reasoning
        )
        contexts_text = response.choices[0].message.content.strip()
        print(
            f"Raw LLM response for sensitive contexts (model: {model_api_id}):\n{contexts_text}\n--------------------")
        return contexts_text
    except openai.error.InvalidRequestError as e:
        print(f"OpenAI API InvalidRequestError for sensitive contexts (model: {model_api_id}): {e}")
        return f"Error generating sensitive contexts: {e}"
    except Exception as e:
        print(f"An unexpected error occurred while generating sensitive contexts (model: {model_api_id}): {e}")
        return f"Unexpected error generating sensitive contexts: {e}"


def extract_entities(text: str) -> list:
    pipelines_dict = get_ner_pipelines()
    all_entities = []
    for name, (ner_pipe, labels) in pipelines_dict.items():
        print(f"Extracting entities with: {name}")
        try:
            if not text.strip():
                print(f"Skipping NER model {name} due to empty input text.")
                continue

            if labels is None:
                results = ner_pipe(text)
                current_entities = []
                if results and isinstance(results[0], list):
                    for chunk_results in results:
                        for ent in chunk_results:
                            current_entities.append({
                                "model_name": name,
                                "type": ent.get("entity_group", ent.get("entity", "N/A")),
                                "text": ent.get("word", "N/A"),
                                "score": ent.get("score", 0.0)
                            })
                else:
                    for ent in results:
                        current_entities.append({
                            "model_name": name,
                            "type": ent.get("entity_group", ent.get("entity", "N/A")),
                            "text": ent.get("word", "N/A"),
                            "score": ent.get("score", 0.0)
                        })
                all_entities.extend(current_entities)
            else:
                if not all(isinstance(label, str) for label in labels):
                    print(f"Warning: Labels for GLiNER model {name} are not all strings: {labels}. Skipping.")
                    continue

                gliner_results = ner_pipe.predict_entities(text, labels, threshold=0.5)
                current_entities_gliner = []
                for ent in gliner_results:
                    current_entities_gliner.append({
                        "model_name": name,
                        "type": ent.get("label", "N/A"),
                        "text": ent.get("text", "N/A"),
                        "score": ent.get("score", 0.0)
                    })
                all_entities.extend(current_entities_gliner)
        except Exception as e:
            print(f"Error during NER with {name}: {e}")
            continue
    return all_entities
