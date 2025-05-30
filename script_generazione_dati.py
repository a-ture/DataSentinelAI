# script_generazione_dati.py
import csv
import random
from faker import Faker
import os  # Importato per la gestione delle cartelle

# Inizializza Faker per la lingua italiana
fake = Faker('it_IT')

# Nome della cartella dove salvare i file generati
NOME_CARTELLA_OUTPUT = "dati_sintetici"

# Liste di esempio per dati medici fittizi (puoi espanderle)
DIAGNOSI_FITTIZIE = [
    "Sindrome da affaticamento digitale", "Lieve ipertensione da stress lavorativo",
    "Disturbo ansioso situazionale", "Cefalea tensiva occasionale", "Rinite allergica stagionale",
    "Gastrite da alimentazione irregolare", "Lombalgia posturale", "Insonnia lieve",
    "Congiuntivite irritativa", "Sindrome del tunnel carpale lieve"
]
TRATTAMENTI_FITTIZI = [
    "Pausa digitale e esercizi oculari", "Tecniche di rilassamento e dieta equilibrata",
    "Supporto psicologico breve", "Analgesici al bisogno e stretching", "Antistaminico al bisogno",
    "Regolarizzazione pasti e antiacidi", "Correzione posturale ed esercizi specifici",
    "Igiene del sonno e tisane rilassanti",
    "Collirio lubrificante", "Tutore notturno e fisioterapia"
]
FARMACI_FITTIZI = [
    "Riposol Forte Compresse", "Calmante Naturale Plus Gocce", "AnalgesicoX 500mg",
    "Benessere Quotidiano Integratore", "AllerStop Spray Nasale", "Gastoprotettore 20mg",
    "SchienaSana Cerotti", "SogniSereni Capsule", "OculVis Gocce", "NeuroLen Crema"
]


def genera_codice_fiscale_fittizio():
    return fake.bothify(text='??????##?##?###?', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ').upper()


def genera_numero_cartella_clinica_fittizio():
    return f"NCC{fake.numerify(text='#######')}"


def genera_dati_medici_fittizi():
    return {
        "numero_cartella_clinica": genera_numero_cartella_clinica_fittizio(),
        "diagnosi": random.choice(DIAGNOSI_FITTIZIE),
        "trattamento": random.choice(TRATTAMENTI_FITTIZI),
        "farmaco_prescritto": random.choice(FARMACI_FITTIZI),
        "medico_curante": f"Dott. {fake.last_name()}"
    }


def genera_dati_persona_completi():
    nome_completo = fake.name()
    indirizzo = fake.address().replace('\n', ', ')
    telefono = fake.phone_number()
    email = fake.email()
    data_nascita = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
    codice_fiscale = genera_codice_fiscale_fittizio()
    azienda = fake.company()
    partita_iva = fake.numerify(text='###########')
    dati_medici = genera_dati_medici_fittizi()

    return {
        "nome_completo": nome_completo,
        "indirizzo": indirizzo,
        "telefono": telefono,
        "email": email,
        "data_nascita": data_nascita,
        "codice_fiscale_fittizio": codice_fiscale,
        "azienda_associata": azienda,
        "partita_iva_fittizia": partita_iva,
        "numero_cartella_clinica_fittizio": dati_medici["numero_cartella_clinica"],
        "diagnosi_recente_fittizia": dati_medici["diagnosi"],
        "trattamento_suggerito_fittizio": dati_medici["trattamento"],
        "farmaco_esempio_fittizio": dati_medici["farmaco_prescritto"],
        "medico_riferimento_fittizio": dati_medici["medico_curante"]
    }


def crea_documento_testo(nome_file_completo="dati_sintetici.txt", num_paragrafi=10, num_pii_per_paragrafo=2,
                         includi_medici=True):
    # Assicura che la cartella di output esista
    os.makedirs(os.path.dirname(nome_file_completo), exist_ok=True)
    with open(nome_file_completo, 'w', encoding='utf-8') as f:
        for _ in range(num_paragrafi):
            paragrafo_base = [fake.sentence() for _ in range(random.randint(5, 8))]

            pii_da_inserire_in_paragrafo = []
            for _ in range(num_pii_per_paragrafo):
                dati_persona = genera_dati_persona_completi()
                scelte_pii_possibili = [
                    dati_persona["nome_completo"], dati_persona["indirizzo"],
                    dati_persona["telefono"], dati_persona["email"],
                    dati_persona["codice_fiscale_fittizio"], dati_persona["partita_iva_fittizia"],
                    dati_persona["azienda_associata"], dati_persona["data_nascita"]
                ]
                if includi_medici:
                    scelte_pii_possibili.extend([
                        dati_persona["numero_cartella_clinica_fittizio"],
                        dati_persona['diagnosi_recente_fittizia'],
                        dati_persona['trattamento_suggerito_fittizio'],
                        dati_persona['farmaco_esempio_fittizio'],
                        dati_persona["medico_riferimento_fittizio"]
                    ])
                pii_da_inserire_in_paragrafo.append(str(random.choice(scelte_pii_possibili)))

            for pii in pii_da_inserire_in_paragrafo:
                if not paragrafo_base: continue
                indice_frase = random.randrange(len(paragrafo_base))
                parole_frase = paragrafo_base[indice_frase].split(' ')
                if len(parole_frase) > 1:
                    indice_parola = random.randint(0, len(parole_frase) - 1)
                    if random.random() < 0.5 and indice_parola > 0:
                        parole_frase[indice_parola - 1] += f" {pii}"
                    else:
                        parole_frase.insert(indice_parola, pii)
                    paragrafo_base[indice_frase] = ' '.join(parole_frase)
                else:
                    paragrafo_base[indice_frase] += f" {pii}"

            f.write(" ".join(paragrafo_base) + "\n\n")
    print(f"File di testo '{nome_file_completo}' generato con successo.")


def crea_documento_csv(nome_file_completo="dati_sintetici.csv", num_righe=50, includi_medici=True):
    # Assicura che la cartella di output esista
    os.makedirs(os.path.dirname(nome_file_completo), exist_ok=True)
    intestazioni = [
        "IDUnivoco", "NomeCompleto", "IndirizzoCompleto", "RecapitoTelefonico", "IndirizzoEmail",
        "DataDiNascita", "CodiceFiscaleSimulato", "NomeAzienda", "PartitaIVASimulata"
    ]
    if includi_medici:
        intestazioni.extend([
            "NumeroCartella", "UltimaDiagnosi",
            "TrattamentoCorrente", "FarmacoPrincipale", "MedicoReferente"
        ])
    intestazioni.append("DescrizioneCasoOUlterioriNote")

    with open(nome_file_completo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=intestazioni)
        writer.writeheader()
        for i in range(num_righe):
            dati_persona = genera_dati_persona_completi()

            riga_csv = {
                "IDUnivoco": f"ID_{i + 1:04d}",
                "NomeCompleto": dati_persona["nome_completo"],
                "IndirizzoCompleto": dati_persona["indirizzo"],
                "RecapitoTelefonico": dati_persona["telefono"],
                "IndirizzoEmail": dati_persona["email"],
                "DataDiNascita": dati_persona["data_nascita"],
                "CodiceFiscaleSimulato": dati_persona["codice_fiscale_fittizio"],
                "NomeAzienda": dati_persona["azienda_associata"],
                "PartitaIVASimulata": dati_persona["partita_iva_fittizia"],
            }

            note_testuali = []
            note_testuali.append(fake.sentence(nb_words=random.randint(8, 15)))
            note_testuali.append(f"Il soggetto {dati_persona['nome_completo']} è stato registrato.")
            note_testuali.append(fake.sentence(nb_words=random.randint(10, 18)))
            if random.random() < 0.7:
                note_testuali.append(f"Per contatti urgenti, si prega di utilizzare {dati_persona['email']}.")
            if random.random() < 0.6:
                note_testuali.append(f"L'ultimo aggiornamento dell'indirizzo è {dati_persona['indirizzo']}.")
            note_testuali.append(fake.sentence(nb_words=random.randint(5, 12)))
            if random.random() < 0.5:
                note_testuali.append(f"Documento di riferimento fiscale {dati_persona['codice_fiscale_fittizio']}.")

            if includi_medici:
                riga_csv.update({
                    "NumeroCartella": dati_persona["numero_cartella_clinica_fittizio"],
                    "UltimaDiagnosi": dati_persona["diagnosi_recente_fittizia"],
                    "TrattamentoCorrente": dati_persona["trattamento_suggerito_fittizio"],
                    "FarmacoPrincipale": dati_persona["farmaco_esempio_fittizio"],
                    "MedicoReferente": dati_persona["medico_riferimento_fittizio"]
                })
                note_testuali.append(f"Il {dati_persona['medico_riferimento_fittizio']} ha seguito il caso.")
                if random.random() < 0.8:
                    note_testuali.append(
                        f"La cartella {dati_persona['numero_cartella_clinica_fittizio']} riporta una diagnosi di {dati_persona['diagnosi_recente_fittizia']}.")
                if random.random() < 0.7:
                    note_testuali.append(
                        f"Attualmente in cura con {dati_persona['farmaco_esempio_fittizio']} come da piano {dati_persona['trattamento_suggerito_fittizio']}.")
                note_testuali.append(fake.sentence(nb_words=random.randint(8, 15)))

            random.shuffle(note_testuali)
            riga_csv["DescrizioneCasoOUlterioriNote"] = " ".join(note_testuali)

            writer.writerow(riga_csv)
    print(f"File CSV '{nome_file_completo}' generato con successo.")


if __name__ == "__main__":
    print("Inizio generazione dati sintetici...")

    # Crea la cartella di output se non esiste
    if not os.path.exists(NOME_CARTELLA_OUTPUT):
        os.makedirs(NOME_CARTELLA_OUTPUT)
        print(f"Cartella '{NOME_CARTELLA_OUTPUT}' creata.")

    # File con mix di dati generali e medici, PII più realistiche
    crea_documento_testo(nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "documento_testo_realistico_misto.txt"),
                         num_paragrafi=30, num_pii_per_paragrafo=3, includi_medici=True)
    crea_documento_csv(nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "tabella_dati_realistici_misti.csv"),
                       num_righe=200, includi_medici=True)

    # File con focus maggiore su dati medici, PII più realistiche
    crea_documento_testo(nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "documento_testo_realistico_medico.txt"),
                         num_paragrafi=20, num_pii_per_paragrafo=4, includi_medici=True)
    crea_documento_csv(nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "tabella_dati_realistici_medici.csv"),
                       num_righe=100, includi_medici=True)

    # File con dati solo generali, PII più realistiche
    crea_documento_testo(
        nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "documento_testo_realistico_generale.txt"),
        num_paragrafi=20, num_pii_per_paragrafo=3, includi_medici=False)
    crea_documento_csv(nome_file_completo=os.path.join(NOME_CARTELLA_OUTPUT, "tabella_dati_realistici_generali.csv"),
                       num_righe=100, includi_medici=False)

    print("Generazione dati sintetici completata.")
    print(f"Controlla i file generati (con 'realistico' nel nome) nella cartella '{NOME_CARTELLA_OUTPUT}'.")

