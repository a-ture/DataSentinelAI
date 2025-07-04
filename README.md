# 🛡️ DataSentinelAI

**DataSentinelAI** è uno strumento avanzato di analisi e anonimizzazione dei dati, progettato per identificare e proteggere le Informazioni Personali Identificabili (PII) all'interno di documenti e dataset. La sua architettura garantisce che tutta l'elaborazione avvenga **100% in locale**, offrendo massima sicurezza e sovranità dei dati.

L'applicazione utilizza una pipeline ibrida che combina la precisione dei modelli di **Named Entity Recognition (NER)** con la potenza di ragionamento contestuale dei **Large Language Models (LLM)** per fornire un'analisi dettagliata e spiegabile dei rischi di privacy.

## ✨ Funzionalità Principali

- **Esecuzione 100% Locale**: Nessun dato viene mai inviato a servizi cloud. Tutta l'analisi, inclusa l'inferenza LLM, avviene sulla macchina locale tramite [LM Studio](https://lmstudio.ai/).
- **Pipeline di Analisi Ibrida**: Combina modelli NER specializzati per un riconoscimento atomico ad alta precisione e LLM per la validazione contestuale, la classificazione del rischio e la spiegabilità.
- **Supporto Multi-Formato**: Analizza una vasta gamma di file, tra cui:
  - Documenti di testo (`.pdf`, `.docx`, `.txt`)
  - Dati strutturati (`.csv`)
- **Estrazione Robusta con OCR**: Per i file PDF, il sistema estrae il testo nativo e applica automaticamente l'OCR (Optical Character Recognition) su pagine basate su immagini o documenti scansionati.
- **Analisi Avanzata per CSV**:
  - **Analisi per Colonna**: Applica strategie differenziate per colonne testuali, numeriche e temporali.
  - **Prompt Engineering Avanzato**: Utilizza prompt ingegnerizzati per forzare l'LLM a restituire output JSON strutturati e conformi per le colonne testuali.
  - **Generalizzazione Basata su Regole**: Anonimizza colonne numeriche e temporali (identificate come Quasi-Identificatori) con tecniche che preservano l'utilità statistica dei dati (es. generalizzazione in quantili).
- **Interazione Resiliente con LLM**:
  - **Parsing Robusto**: Implementa una catena di fallback (`json.loads` -> `ast.literal_eval` -> `regex`) per gestire output JSON imperfetti o malformati.
  - **Gestione Automatica del Server**: Verifica se il server di LM Studio è attivo e tenta di avviarlo programmaticamente.
  - **Esecuzione Asincrona**: Ottimizza le chiamate all'LLM per l'analisi dei CSV tramite `asyncio` e un `Semaphore` per prevenire il sovraccarico del server.
- **Metriche di Privacy Quantitative**: Calcola metriche standard come **k-anonimato** e **l-diversity** per fornire una valutazione oggettiva del rischio di re-identificazione e inferenza.
- **Anonimizzazione Configurabile**: Offre molteplici tecniche di anonimizzazione (redazione, masking, pseudonimizzazione, generalizzazione) e genera versioni anonimizzate dei documenti e dei dataset.

---

## 🔧 Prerequisiti e Installazione

Per eseguire DataSentinelAI, sono necessari i seguenti componenti:

### Prerequisiti Software
1.  **Python 3.9+**
2.  **LM Studio**: Scarica e installa l'applicazione desktop da [lmstudio.ai](https://lmstudio.ai/).
3.  **Tesseract OCR**: Necessario per l'analisi di PDF scannerizzati.
    - **Windows**: Segui le istruzioni di installazione e assicurati di aggiungere l'eseguibile di Tesseract al `PATH` di sistema.
    - **macOS**: `brew install tesseract`
    - **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

### Guida all'Installazione

1.  **Clona il Repository**
    ```bash
    git clone [https://github.com/a-ture/ProgettoDL](https://github.com/a-ture/ProgettoDL)
    cd DataSentinelAI
    ```

2.  **Installa le Dipendenze Python**
    Si consiglia di creare un ambiente virtuale:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```
    Installa le librerie necessarie:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configura LM Studio e i Modelli**
    a. Avvia LM Studio.
    b. Cerca e scarica i seguenti modelli dalla sezione "Discover":
       - `gemma-2-2b-it` (o una sua variante compatibile)
       - `meta-llama-3.1-8b-instruct` (o una sua variante compatibile)
    c. Vai alla sezione del server locale (icona `<->`) e carica uno dei modelli.
    d. **IMPORTANTE**: Verifica gli identificatori dei modelli caricati in LM Studio. Apri il file `modules/config.py` e assicurati che gli ID corrispondano esattamente a quelli mostrati da LM Studio.
       ```python
       # modules/config.py
       LLM_MODELS = {
           "Google Gemma 2B": "gemma-2-2b-it", # Verifica questo ID in LM Studio
           "Meta Llama 3.1": "meta-llama-3.1-8b-instruct", # Verifica questo ID in LM Studio
       }
       ```

---

## 🚀 Utilizzo
1.  **Avvia l'Applicazione Streamlit**
    Assicurati che il tuo ambiente virtuale sia attivo, quindi esegui:
    ```bash
    streamlit run streamlit_app.py
    ```
    L'applicazione si aprirà automaticamente nel tuo browser.

3.  **Workflow di Analisi**
    a. **Carica un File**: Utilizza l'uploader per selezionare un file `.pdf`, `.txt`, `.docx` o `.csv`.
    b. **Configura l'Analisi (per CSV)**: Se hai caricato un file CSV, puoi configurare opzioni avanzate come il modello LLM da usare per le colonne testuali e i parametri di generalizzazione.
    c. **Avvia l'Analisi**: Clicca sul pulsante "Analizza". L'applicazione mostrerà una barra di avanzamento durante l'elaborazione.
    d. **Visualizza i Risultati**: Al termine, verranno mostrati i report dettagliati, le metriche di privacy e le entità PII identificate.
    e. **Esporta gli Artefatti**: Puoi scaricare il report di analisi in formato JSON e una versione anonimizzata del documento o del dataset.

---

## 📂 Struttura del Progetto

```
DataSentinelAI/
│
├── modules/
│   ├── __init__.py
│   ├── analysis_csv.py       # Logica di analisi specifica per file CSV
│   ├── config.py             # Configurazione dei modelli LLM
│   ├── generazione_testo.py  # Interazione con LM Studio, prompt engineering, parsing robusto
│   ├── privacy_metrics.py    # Calcolo di k-anonimato e l-diversity
│   ├── text_extractor.py     # Estrazione del testo da PDF, DOCX, TXT con fallback OCR
│   └── utils.py              # Funzioni di utilità
│
├── dati_sintetici/           # Cartella per i dati di test generati
│   └── ...
│
├── streamlit_app.py          # File principale dell'interfaccia utente Streamlit
├── script_generazione_dati.py # Script per generare dati di test fittizi
├── requirements.txt          # Elenco delle dipendenze Python
└── README.md                 # Questo file
```
