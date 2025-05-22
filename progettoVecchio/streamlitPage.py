import streamlit as st

from GenerazioneTesto import sensitiveInformations
from textExtractor import extract, writeFile
from textExtractor import findExtension
import GenerazioneTesto

firstLLMModel="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
secondLLMModel="lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF/Phi-3.1-mini-4k-instruct-Q4_K_M.gguf"
thirdLLMModel="lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf"

containerHeight=500

st.set_page_config(page_title="Analizza il tuo documento",layout="wide")
first,center,last = st.columns([0.15,0.7,0.15])
_,ccenter,_ = center.columns(3)
ccenter.title('Analyze your data')
file = center.file_uploader("Upload your file to get you report", type=['pdf', 'docx', 'txt'], accept_multiple_files=False)
testo=""
flag=False
if file is not None:
    st.session_state['fileExtension'] = findExtension(file)
    testo = extract(file)
    if center.button("Genera report", type="primary", use_container_width=True):
        flag=True
        with center.chat_message("assistant"):
            with center.status('**Analizzando...**', expanded=True) as status:
                firstreport = GenerazioneTesto.generateReport(testo,firstLLMModel)
                secondreport= GenerazioneTesto.generateReport(testo,secondLLMModel)
                thirdreport= GenerazioneTesto.generateReport(testo,thirdLLMModel)
                status.update(
                    label="**Analisi completa!**", state="complete", expanded=True)
                st.markdown(f"# Ecco il tuo report:\n\n {"#### "+firstLLMModel+":\n\n"+firstreport+"\n\n #### "+secondLLMModel+":\n\n"+secondreport+"\n\n #### "+thirdLLMModel+":\n\n"+thirdreport}")
                st.session_state['firstreport'] = firstreport
                st.session_state['secondreport'] = firstreport
                st.session_state['thirdreport'] = firstreport

if flag:
    center.write("Modificare il file secondo le indicazioni?")
    button = center.button("Modifica", type="secondary", disabled=False, use_container_width=True)
else:
    button = center.button("Modifica", type="secondary", disabled=True, use_container_width=True)


if button:
    firstreport = st.session_state['firstreport']
    secondreport = st.session_state['secondreport']
    thirdreport = st.session_state['thirdreport']
    with center.status('**Modificando...**') as status:
        left, centerr, right = center.columns(3)
        firstedited = GenerazioneTesto.editDocument(testo, firstreport,firstLLMModel)
        secondedited = GenerazioneTesto.editDocument(testo, secondreport,secondLLMModel)
        thirdedited = GenerazioneTesto.editDocument(testo, thirdreport,thirdLLMModel)
        nerEdited = GenerazioneTesto.extractEntities(testo)
        status.update(
            label="**Modifica completata!**", state="complete", expanded=True)
        leftContainer = left.container(height=containerHeight)
        centerContainer = centerr.container(height=containerHeight)
        rightContainer= right.container(height=containerHeight)

        editedResult = firstLLMModel+":\n\n"+firstedited+"\n\n"+secondLLMModel+":\n\n"+secondedited+"\n\n"+thirdLLMModel+":\n\n"+thirdedited

        leftContainer.markdown(f"## File modificato:\n\n{editedResult}")
        centerContainer.markdown(f"## Entità riconosciute:\n\n{nerEdited}")
        rightContainer.markdown(f"## File originale:\n\n{testo}")
    extension = st.session_state['fileExtension']
    left.download_button(label="**Scarica file modificato**", data=writeFile(editedResult, extension),
                       file_name='edited' + extension)
    centerr.download_button(label="**Scarica entità riconosciute**", data=writeFile(nerEdited, ".txt"),
                         file_name='recognized' + ".txt")
    center.download_button(label="**Scarica file modificato\n ed entità riconosciute**", data=writeFile(testo+"\n\n"+'-'*30+"Entità riconosciute:"+'-'*30+"\n\n"+nerEdited, extension), file_name="Bundle"+extension)
    firstContainer = first.container()
    firstContainerr = firstContainer.container(height=containerHeight%2)
    firstContainerr.markdown(f"## Contesti in cui le entità possono essere sensibili:\n\n{sensitiveInformations(firstreport, firstLLMModel)}")
    firstContainerr.markdown(f"## Contesti in cui le entità possono essere sensibili:\n\n{sensitiveInformations(secondreport, secondLLMModel)}")
    firstContainerr.markdown(f"## Contesti in cui le entità possono essere sensibili:\n\n{sensitiveInformations(thirdreport, thirdLLMModel)}")
