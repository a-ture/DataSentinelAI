"""
Configurazione dei modelli LLM per DataSentinelAI.
ATTENZIONE: Usa gli IDENTIFICATORI ESATTI che il server API di LMStudio
si aspetta per ogni modello. Questi identificatori vengono mostrati
da LMStudio quando un modello viene caricato con successo tramite la UI,
o tramite il comando `lms load <nome_modello_base_o_path>`, oppure
visualizzati tramite `lms ls` (o `lms list loaded`).
"""

LLM_MODELS = {
    # "Microsoft Phi 3.1 Mini": "phi-3.1-mini-4k-instruct:2",
    "Google Gemma 2B": "gemma-2-2b-it",
    "Meta Llama 3.1": "meta-llama-3.1-8b-instruct",
}
