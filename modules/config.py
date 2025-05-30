
"""
Configurazione dei modelli LLM per DataSentinelAI.
ATTENZIONE: Usa gli IDENTIFICATORI ESATTI che il server API di LMStudio
si aspetta per ogni modello. Questi identificatori vengono mostrati
da LMStudio quando un modello viene caricato con successo tramite la UI,
o tramite il comando `lms load <nome_modello_base_o_path>`, oppure
visualizzati tramite `lms ls` (o `lms list loaded`).

Esempio dai tuoi log `lms ls`:
- Per Phi-3.1 Mini: `phi-3.1-mini-4k-instruct ... ✓ LOADED (2)` -> API ID: "phi-3.1-mini-4k-instruct:2"
- Per Gemma 2B: `gemma-2-2b-it ... ✓ LOADED` -> API ID: "gemma-2-2b-it"
- Per Meta Llama 3.1: `meta-llama-3.1-8b-instruct` (NON CARICATO NEI LOG) -> API ID: "meta-llama-3.1-8b-instruct" (o con suffisso :N una volta caricato)
"""

# modules/config.py (Esempio di come dovrebbe essere)
LLM_MODELS = {
    #"Microsoft Phi 3.1 Mini": "phi-3.1-mini-4k-instruct:2",
    "Google Gemma 2B": "gemma-2-2b-it",
    "Meta Llama 3.1": "meta-llama-3.1-8b-instruct",
}

# Verifica che LLM_MODELS non sia vuoto, altrimenti l'app potrebbe non funzionare come previsto.
if not LLM_MODELS:
    print("ATTENZIONE: Il dizionario LLM_MODELS in config.py è vuoto.")
    print("L'applicazione necessita di almeno un modello configurato per funzionare.")

