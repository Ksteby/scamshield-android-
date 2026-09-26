# ScamShield

## 1. Descrizione del progetto

ScamShield è un'applicazione Android sviluppata in Kotlin con
l'obiettivo di aiutare l'utente a riconoscere messaggi potenzialmente
fraudolenti.

L'applicazione analizza il contenuto di un messaggio e restituisce un
**punteggio di rischio da 0 a 100**, accompagnato da un livello di
rischio e da indicazioni utili per l'utente.

Il progetto integra un sistema di analisi ibrido basato su Machine
Learning e analisi a regole.

Le principali caratteristiche sono: - analisi offline sul dispositivo; -
analisi online tramite REST API; - classificazione di messaggi in
italiano e inglese; - ricezione di testo tramite il menu Condividi di
Android.

## 2. Funzionalità principali

### Analisi offline

La modalità offline utilizza il modello V3 integrato negli assets
dell'applicazione. Il messaggio viene analizzato direttamente sul
dispositivo e non è necessaria una connessione Internet.

### Analisi online

La modalità online invia il messaggio a un server FastAPI tramite
Retrofit. Il server utilizza lo stesso modello V3 della modalità
offline.

### Condivisione di messaggi

ScamShield può ricevere un messaggio tramite il normale menu
**Condividi** di Android.

### Risultato

L'app mostra: - punteggio di rischio da 0 a 100; - livello di rischio; -
classificazione del messaggio; - spiegazione generale; - segnali
sospetti rilevati; - consigli sul comportamento da adottare.

## 3. Architettura

``` text
ScannerScreen
      ↓
ScannerViewModel
      ↓
AnalysisRepository
      ↓
 ┌───────────────┬────────────────┐
 │               │                │
Offline        Online             │
 │               ↓                │
 ↓            Retrofit            │
HybridAnalyzer  ↓                 │
 │            FastAPI             │
 │               ↓                │
 └───────────────┴────────────────┘
                 ↓
          AnalysisResult
                 ↓
           ResultScreen
```

La modalità offline e la modalità online utilizzano lo stesso modello
V3, permettendo di ottenere risultati coerenti.

## 4. Machine Learning

### Dataset

Il dataset combinato contiene **9.572 messaggi**: - 6.825 messaggi
ham; - 2.747 messaggi spam.

Sono stati utilizzati messaggi in lingua inglese e italiana.

Il dataset è stato suddiviso in: - 80% sviluppo; - 20% test finale
indipendente.

### Modello V3

La versione finale utilizza due rappresentazioni complementari del
testo:

1.  **Word TF-IDF**
    -   unigrammi e bigrammi;
    -   `ngram_range=(1,2)`.
2.  **Character TF-IDF**
    -   n-grammi di caratteri da 3 a 5;
    -   `sublinear_tf=True`.

Le due rappresentazioni vengono concatenate e utilizzate da una
**Logistic Regression**.

In parallelo viene utilizzato un **Rule Analyzer** basato su 14 feature.
Tra i segnali considerati: - urgenza; - richieste d'azione; -
informazioni monetarie; - informazioni sensibili; - URL; - contatti; -
numeri di telefono; - interazioni tra segnali.

Il punteggio finale è ottenuto combinando il modello ML e il Rule
Analyzer.

### Risultati sul test finale

  Metrica       Modello ML   Sistema ibrido V3
  ----------- ------------ -------------------
  Accuracy          96,61%          **97,34%**
  Precision         98,21%          **96,99%**
  Recall            89,82%          **93,64%**
  F1-score          93,83%          **95,28%**

La soglia utilizzata dal sistema è circa **38,92/100**.

## 5. Tecnologie utilizzate

### Android

-   Kotlin
-   Android Studio
-   Jetpack Compose
-   Material 3
-   ViewModel
-   Retrofit
-   Gson
-   Android Intent / Share
-   Assets locali

### Machine Learning e server

-   Python
-   FastAPI
-   Uvicorn
-   NumPy
-   SciPy
-   Scikit-learn
-   Pandas
-   Regular Expressions

## 6. Struttura principale

``` text
ScamShield/
├── app/
│   └── src/main/
│       ├── java/com/kemo/scamshield/
│       │   ├── data/
│       │   ├── domain/
│       │   ├── navigation/
│       │   ├── userInterface/
│       │   │   ├── scanner/
│       │   │   ├── result/
│       │   │   └── challenges/
│       │   └── utils/
│       └── assets/
│           └── model/
│               ├── v3_tfidf_word_vocabulary.json
│               ├── v3_tfidf_word_idf.json
│               ├── v3_tfidf_char_vocabulary.json
│               ├── v3_tfidf_char_idf.json
│               ├── v3_ml_model.json
│               ├── v3_rule_model.json
│               └── v3_model_config.json
└── ml/
    ├── server.py
    ├── models_v3/
    └── ...
```

## 7. Installazione

### Requisiti

Sono necessari: - Android Studio; - JDK compatibile con il progetto; -
Python 3; - un dispositivo Android oppure un emulatore; - per la
modalità online, il server FastAPI.

### Progetto Android

1.  Aprire la cartella `ScamShield` con Android Studio.
2.  Attendere la sincronizzazione Gradle.
3.  Eseguire `Build → Make Project`.
4.  Avviare l'app su un dispositivo o emulatore.

## 8. Avvio del server FastAPI

Entrare nella cartella:

``` text
ScamShield/ml
```

Attivare l'ambiente virtuale:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Avviare il server:

``` powershell
uvicorn server:app --host 127.0.0.1 --port 8000
```

Verificare il funzionamento aprendo:

``` text
http://127.0.0.1:8000
```

La risposta prevista è:

``` json
{
  "application": "ScamShield API",
  "status": "online"
}
```

## 9. Modalità online con l'emulatore

Con l'emulatore Android avviato, eseguire:

``` powershell
adb reverse tcp:8000 tcp:8000 oppure
 & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
```

L'applicazione utilizza:

``` text
http://10.0.2.2:8000/
```

come `BASE_URL` di Retrofit.

Procedura: 1. avviare FastAPI; 2. eseguire
`adb reverse tcp:8000 tcp:8000`; 3. avviare ScamShield; 4. selezionare
**Online**; 5. inserire o condividere un messaggio; 6. premere
**Analizza messaggio**.

## 10. Modalità offline

1.  Aprire l'applicazione.
2.  Selezionare **Offline**.
3.  Inserire un messaggio.
4.  Premere **Analizza messaggio**.

Il server non è necessario.

## 11. Test effettuati

Durante lo sviluppo sono state verificate: - analisi di messaggi spam in
italiano e inglese; - analisi di messaggi legittimi; - rilevamento di
URL; - rilevamento di numeri di telefono; - rilevamento di segnali di
urgenza; - rilevamento di richieste d'azione; - analisi offline; -
analisi online; - coerenza tra modalità offline e online; - ricezione
tramite Android Share; - navigazione tra le schermate; - gestione del
campo di testo; - funzionamento del pulsante di analisi.

## 12. Limitazioni

Il modello è stato addestrato su dataset di messaggi SMS e non può
garantire il riconoscimento corretto di ogni possibile truffa.

Il punteggio visualizzato rappresenta un **punteggio di rischio del
sistema**, non una probabilità calibrata che il messaggio sia realmente
fraudolento.

La modalità online richiede che il server FastAPI sia attivo e
raggiungibile.

## 13. Possibili sviluppi futuri

-   utilizzo di dataset più grandi e diversificati;
-   aggiornamento periodico del modello;
-   supporto a ulteriori lingue;
-   miglioramento dell'analisi contestuale;
-   deployment del server su un'infrastruttura remota;
-   supporto a ulteriori tipologie di messaggi.

## 14. Autore

**ScamShield**\
Progetto per l'esame di Mobile Development\
Università degli Studi di Parma\
KEMO TOUOHOU STEBY
