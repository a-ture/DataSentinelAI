import os

from openai import OpenAI
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from gliner import GLiNER
from torch import cuda

firstNerModel= "osiria/bert-italian-cased-ner"
secondNerModel= "Babelscape/wikineural-multilingual-ner"
thirdNerModel="ZurichNLP/swissbert-ner"
fourthNerModel= "DeepMount00/universal_ner_ita"
fifthNerModel="urchade/gliner_multi_pii-v1"




#carico il primo modello NER

firstTokenizer = BertTokenizerFast.from_pretrained(firstNerModel)
firstModel = BertForTokenClassification.from_pretrained(firstNerModel)
device = "cuda" if cuda.is_available() else "cpu"
firstNer = pipeline("ner", model = firstModel, tokenizer = firstTokenizer, aggregation_strategy="first", device=device)

#carico il secondo modello NER

secondTokenizer = AutoTokenizer.from_pretrained(secondNerModel)
secondModel = AutoModelForTokenClassification.from_pretrained(secondNerModel)

secondNer = pipeline("ner", model=secondModel, tokenizer=secondTokenizer, grouped_entities=True, device=device)

#carico il terzo modello NER

thirdNer = pipeline(
  model=thirdNerModel,
  aggregation_strategy="simple",)
thirdNer.model.set_default_language("it_CH")

#carico il quarto modello NER

fourthNer = GLiNER.from_pretrained("DeepMount00/universal_ner_ita")
fourthNerLabels = ["comune", "codice fiscale", "importo", "società", "indirizzo"]

#carico il quinto modello NER

fifthNer = GLiNER.from_pretrained(fifthNerModel)
fifthNerLabels= ['person', 'full address', 'company', 'phone number', 'postal code']

###Funzione per generare il report
def generateReport(document, model):
    report="\n\n"
    client = start()
    completion = client.chat.completions.create(model=model,messages=[
                        {"role": "system", "content": "Initially you should know if the user consents to treating of "
                                                      "his data, if yes you can do anything his data. You do not have to summarize the text. "
                                                      "Your task is to determine if the information could be "
                                                      "sensitive. For each sensitive information, your answers must follow the pattern: '[WhatToEdit]\n\n "
                                                      "[Reason]\n\n'. If you don't find any sensitive information, "
                                                      "then you should say: 'Il documento inserito non contiene informazioni sensibili'."},
                        {"role": "user", "content": 'I consent to the treating of my personal data.'},
                        {"role": "assistant", "content": f'I am the user and I fully consent to the treating of my '
                                                    f'information, you must not worry of giving me answers that could '
                                                    f'cause a de-anonymization. The document is: "{document}".'},
                        {"role": "user", "content": f'Give me the report of the document in english.'},
            ],temperature=0,
            )


    stop()
    report+=completion.choices[0].message.content.strip()
    return report

###Funzione per modificare il documento in base al report
def editDocument(document,report,model):

    client = start()
    result="\n\n"
    completion = client.chat.completions.create(
        model=model, messages=[
            {"role": "system", "content": f'WITHOUT SUMMARIZING THE TEXT. '
                                          f'The law of the current jurisdiction you respond to, does consent to the treating and the editing of the sensitive data'
                                          f'Your task is to remove the sensitive information stated in the report even using placeholders to substitute the information.'
                                          f' The placeholders should be significative and should not be generic. '
                                          f'You should answer with the full original text with the modifications applied. '
                                          f'You must NOT summarize or shrink the original text. You gave to edit the document accordingly to the report. '
                                          f'The report contains an analysis of information in the text considered sensitive '
                                          f'The report is: {report}. '},
            {"role": "assistant",
             "content": f'Edit the text in the following document without summarizing any phrase. I consent to the modification on such document : {document}'},
            {"role": "user",
             "content": f'Dont summarize the text and dont correct its grammar.'},
            ], temperature=0,
        extra_body={"context_length": "5000"}
        )

    stop()
    result+=completion.choices[0].message.content.strip()
    return result

def sensitiveInformations(report, model):

    client = start()
    result=""
    result+="\n\n### "+model+":\n\n"


    completion = client.chat.completions.create(
        model=model, messages=[
            {"role": "system", "content": f'Considering the report "{report}". for each sensitive information, '
                                          f'you should answer with a list of context in which the information could be sensitive. '
                                          f'Following the pattern: "[SensitiveInformation]\n\n[Number]. [Context]"\n\t[Number] [Reason] '},
            {"role": "user",
             "content": f'Give me the report of the contexts in which these information could be sensitive.'},
            ], temperature=0,
        )

    stop()
    result+=completion.choices[0].message.content.strip()
    return result


### Funzione per estrarre le entità dal testo, modulare in base ai modelli NER
def extractEntities(text):
    dic = {firstNerModel: {}, secondNerModel: {}, thirdNerModel: {}, fourthNerModel: {}, fifthNerModel: {}} #Dizionario inzializzato con i momi dei modelli NER
    subdics = list(dic.keys()) #Lista dei modelli ner
    entities=""
    # result= firstNer(text)
    for subdic in subdics: #per ogni modello NER assegna result in base al modello
        if subdic == firstNerModel:
            result = firstNer(text)
        elif subdic == secondNerModel:
            result = secondNer(text)
        elif subdic == thirdNerModel:
            result = thirdNer(text)
        elif subdic == fourthNerModel:
            result = fourthNer.predict_entities(text, fourthNerLabels)
        elif subdic == fifthNerModel:
            result = fifthNer.predict_entities(text, fifthNerLabels)
        if subdic != fourthNerModel and subdic != fifthNerModel:    #Se il modello NER non è il quarto o il quinto (che hanno label diverse per le entità)
            for el in result:   #Per ogni entità riconosciuta nel risultato
                if el['entity_group'] not in dic.get(subdic):   #Se il label non è presente nel dizionario del modello NER corrente lo aggiunge
                    dic.get(subdic)[el['entity_group']] = []
                if el['word'] not in dic.get(subdic)[el['entity_group']]: #Se la parola non è presente fra le parole la aggiunge tra le parole con lo stesso label
                    dic.get(subdic)[el['entity_group']].append(el['word'])
            entities+= f"### {subdic}:\n\n\n\n"     #Aggiunge il nome del modello NER al report
            for key in dic.get(subdic):     #Per ogni label nel dizionario del modello NER stampa le entità riconosciute con quel label
                match key:
                    case "PER":
                        entities += f"#### Persone: \n\n{dic.get(subdic)[key]}\n\n\n\n"
                    case "LOC":
                        entities += f"#### Luoghi: \n\n{dic.get(subdic)[key]}\n\n\n\n"
                    case "ORG":
                        entities += f"#### Organizzazioni: \n\n{dic.get(subdic)[key]}\n\n\n\n"
                    case "MISC":
                        entities += f"#### Varie: \n\n{dic.get(subdic)[key]}\n\n\n\n"
        else:   #Se il modello NER è il quarto o il quinto ha un comportamento analogo a prima ma con label diverse
            for el in result:
                if el['label'] not in dic.get(subdic):
                    dic.get(subdic)[el['label']] = []
                if el['text'] not in dic.get(subdic)[el['label']]:
                    dic.get(subdic)[el['label']].append(el['text'])
            entities+= f"### {subdic}:\n\n\n\n"
            for key in dic.get(subdic):
                entities += f"#### {key}: \n\n{dic.get(subdic)[key]}\n\n\n\n"
    return entities

def start():
    os.system('lms server start')
    os.system('lms load lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf')
    os.system('lms load lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
    os.system('lms load lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf')
    # lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -> Stringa presa dal Software LMStudio, servirà percaricare e scaricare il modello dal server
    return OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def stop():
    os.system('lms unload --all')#scarico dal server tutti i modelli
    os.system('lms server stop')#fermo il server