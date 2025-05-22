import os
from functools import lru_cache
import openai
from torch import cuda
from transformers import (
    pipeline,
    BertTokenizerFast,
    BertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from gliner import GLiNER

# Configura OpenAI per LMStudio
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"

@lru_cache(maxsize=None)
def _init_lmstudio() -> None:
    """
    Avvia LMStudio server e carica il modello mini (solo la prima volta).
    """
    os.environ["LMS_NO_KV_CACHE"] = "1"
    os.system("lms server start")
    os.system(
        "lms load lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF/"
        "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"
    )

@lru_cache(maxsize=None)
def get_ner_pipelines():
    """
    Inizializza e ritorna tutte le pipeline NER.

    Returns:
        dict: nome -> (pipeline, labels)
    """
    device = 0 if cuda.is_available() else -1
    pipelines = {}

    # 1) BERT italiano
    tok1 = BertTokenizerFast.from_pretrained("osiria/bert-italian-cased-ner")
    mod1 = BertForTokenClassification.from_pretrained("osiria/bert-italian-cased-ner")
    pipelines['bert_it'] = (
        pipeline(
            "ner",
            model=mod1,
            tokenizer=tok1,
            aggregation_strategy="first",
            device=device
        ),
        None
    )

    # 2) WikiNeural multilingue
    tok2 = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    mod2 = AutoModelForTokenClassification.from_pretrained(
        "Babelscape/wikineural-multilingual-ner"
    )
    pipelines['wiki_multi'] = (
        pipeline(
            "ner",
            model=mod2,
            tokenizer=tok2,
            grouped_entities=True,
            device=device
        ),
        None
    )

    # 3) SwissBERT
    pipe3 = pipeline(
        "ner",
        model="ZurichNLP/swissbert-ner",
        aggregation_strategy="simple",
        device=device
    )
    try:
        pipe3.model.set_default_language("it_CH")
    except Exception:
        pass
    pipelines['swiss_it'] = (pipe3, None)

    # 4) GLiNER italiano generico
    gl4 = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")
    lbl4 = ["comune", "codice fiscale", "importo", "società", "indirizzo"]
    pipelines['gliner_it'] = (gl4, lbl4)

    # 5) GLiNER PII multilingue
    gl5 = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    lbl5 = [
        'person',
        'full address',
        'company',
        'phone number',
        'postal code'
    ]
    pipelines['gliner_pii'] = (gl5, lbl5)

    return pipelines


def generate_report(text: str, model: str) -> str:
    """
    Genera un report sulle informazioni sensibili del testo.
    """
    _init_lmstudio()
    system_prompt = (
        "Detect sensitive information. For each found, output '[WhatToEdit]\n\n"
        "[Reason]\n'. If none, output 'Il documento inserito non contiene informazioni sensibili'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I consent to data processing."},
        {"role": "assistant", "content": f"Document: '{text}'"},
        {"role": "user", "content": "Provide the report in English."}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def edit_document(text: str, report: str, model: str) -> str:
    """
    Modifica il testo originale in base al report.
    """
    _init_lmstudio()
    system_prompt = (
        "Remove sensitive information using meaningful placeholders."
        f" Report: {report}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"Modify document: {text}"},
        {"role": "user", "content": "Do not summarize or correct grammar."}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def sensitive_informations(report: str, model: str) -> str:
    """
    Elenca i contesti in cui le informazioni sensibili sono rilevanti.
    """
    _init_lmstudio()
    system_prompt = (
        f"Given the report '{report}', list contexts for each sensitive information: "
        "'[SensitiveInformation]\n\n[1]. [Context]'"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Provide the sensitive contexts."}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def extract_entities(text: str) -> str:
    """
    Esegue l'analisi NER con tutte le pipeline e aggrega i risultati.
    """
    pipelines = get_ner_pipelines()
    lines = []
    for name, (ner_pipe, labels) in pipelines.items():
        lines.append(f"### {name}\n")
        if labels is None:
            for ent in ner_pipe(text):
                label = ent.get("entity_group", "")
                word = ent.get("word") or ent.get("text", "")
                lines.append(f"- {label}: {word}\n")
        else:
            for ent in ner_pipe.predict_entities(text, labels):
                lines.append(f"- {ent['label']}: {ent['text']}\n")
        lines.append("\n")
    return "".join(lines)
