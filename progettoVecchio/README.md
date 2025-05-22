# DataSentinelAI
___
_DataSentinelAI_ è un tool per l'identificazione e rimozione di dati sensibili da documenti di testo.\
Esso è stato sviluppato per la tesi di laurea triennale in Informatica presso l'Università degli Studi di Salerno. 
Il tool è stato sviluppato in Python e utilizza 5 modelli per il task NER:
1. _osiria/bert-italian-cased-ner_, 
2. _Babelscape/wikineural-multilingual-ner_, 
2. _ZurichNLP/swissbert-ner_, 
3. _DeepMount00/universal_ner_ita_, 
4. _urchade/gliner_multi_pii-v1_. 

Utilizza inoltre 3 LLM per analizzare il testo, comporne un report e rimuovere i dati sensibili:
1. _lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF/Phi-3.1-mini-4k-instruct-Q4_K_M_ (**Microsoft Phi 3.1 mini**)
2. _lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M_ (**Meta Llama 3.1**)
3. _lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0_ (**Google Gemma 2B**)
---
Il tool utilizza il software LM Studio per l'hosting e l'interazione con gli LLM.\
Per utilizzare il tool è innanzitutto necessario installare LM Studio e scaricare i modelli sopracitati,
successivamente è necessario installare le librerie Python richieste dal tool:\
`pip install -r requirements.txt` \
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` 
 
Per eseguire il tool è necessario avviare LM Studio e solo dopo utilizzare la libreria _streamlit_ e il modulo run: `streamlit run streamlitPage.py`.\
È poi possibile caricare un file dei tipi: PDF, DOCX, TXT. Una volta premuto il bottone **Genera Report**, il _DataSentinelAI_ restituirà un report per modello con i dati sensibili identificati, con una spiegazione e un consiglio su cosa modificare.\
Premendo il bottone **Modifica** verrà generato un nuovo file con i dati sensibili rimossi e contemporaneamente verrà presentato il risultato dei 5 modelli NER con le entità riconosciute organizzate in base al modello utilizzato, viene inoltre mostrato all'utente il contenuto del file originale.\
Contemporaneamente, il tool genera sul lato una serie di contesti nei quali le informazioni identificate possono risultare sensibili fornendo una serie di esempi.\
È possibile scaricare i seguenti file: documento modificato, entità riconosciute, entrambi i precedenti nello stesso file.
___