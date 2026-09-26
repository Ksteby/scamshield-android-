import json
import re
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_v3"


# ------------------------------------------------------------
# Création de l'API
# ------------------------------------------------------------

app = FastAPI(
    title="ScamShield API",
    description="API d'analyse de messages frauduleux",
    version="1.0"
)


# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------

def load_json(filename):
    """Carica un file JSON dalla cartella dei modelli V3."""

    with open(MODEL_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


# ------------------------------------------------------------
# Chargement du modèle V3
# ------------------------------------------------------------

word_vocabulary = load_json(
    "v3_tfidf_word_vocabulary.json"
)

word_idf = np.array(
    load_json("v3_tfidf_word_idf.json"),
    dtype=float
)

character_vocabulary = load_json(
    "v3_tfidf_char_vocabulary.json"
)

character_idf = np.array(
    load_json("v3_tfidf_char_idf.json"),
    dtype=float
)

ml_model_data = load_json(
    "v3_ml_model.json"
)

rule_model_data = load_json(
    "v3_rule_model.json"
)

model_config = load_json(
    "v3_model_config.json"
)


# ------------------------------------------------------------
# Creazione dei vectorizer TF-IDF
# ------------------------------------------------------------

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    vocabulary=word_vocabulary
)

character_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    vocabulary=character_vocabulary,
    sublinear_tf=True
)

# Inizializziamo i vectorizer prima di utilizzare i valori IDF.
word_vectorizer.fit(["test"])

character_vectorizer.fit(["test"])

# Utilizziamo esattamente i valori IDF esportati dal modello V3.
word_vectorizer._tfidf.idf_ = word_idf

character_vectorizer._tfidf.idf_ = character_idf


# ------------------------------------------------------------
# Fonctions de vectorisation
# ------------------------------------------------------------

def transform_word(text):
    """
    Trasforma il testo con il vectorizer Word TF-IDF V3.
    """

    matrix = word_vectorizer.transform([text])

    return matrix.toarray()[0]


def transform_character(text):
    """
    Trasforma il testo con il vectorizer Character TF-IDF V3.
    """

    matrix = character_vectorizer.transform([text])

    return matrix.toarray()[0]

# ------------------------------------------------------------
# Rule Analyzer
# ------------------------------------------------------------

def has_urgency(message):
    pattern = (
        r"\burgente\b|\burgenti\b|\burgent\b|"
        r"\bsubito\b|\bimmediatamente\b|\bscadenza\b|"
        r"\boggi\b|\bora\b|\bnow\b|\bimmediately\b|"
        r"\btoday\b|\bentro\b"
    )

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_action(message):
    pattern = (
        r"\bclicca\b|\bclick\b|\brispondi\b|\breply\b|"
        r"\bchiama\b|\bcall\b|\bconferma\b|\bconfirm\b|"
        r"\bverifica\b|\bverify\b|\bscarica\b|\bdownload\b|"
        r"\bapri\b|\bopen\b"
    )

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_money(message):
    pattern = r"€|\$|£|\beuro\b|\beuros\b"

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_sensitive(message):
    pattern = (
        r"\bbanca\b|\bbank\b|\bconto\b|\baccount\b|"
        r"\bpassword\b|\bcodice\b|\bcode\b|\bpin\b|"
        r"\bcredenziali\b|\bcredentials\b|\bcarta\b|"
        r"\bcard\b|\bpagamento\b|\bpayment\b"
    )

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_url(message):
    pattern = r"https?://|www\."

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_contact(message):
    pattern = (
        r"\bcontatta\b|\bcontattare\b|\bcontatto\b|"
        r"\bcontatti\b|\bchiamaci\b|\bcall\b|\bcontact\b|"
        r"\bcontatta\s+la\s+propria\s+sede\b"
    )

    return bool(re.search(pattern, message, re.IGNORECASE))


def has_phone(message):
    pattern = r"(?<!\d)\+?\d[\d\s().-]{6,}\d(?!\d)"

    return bool(re.search(pattern, message))


def create_rule_features(message):
    """
    Crea le 14 feature nello stesso ordine utilizzato
    durante l'addestramento V3.
    """

    urgency = has_urgency(message)
    action = has_action(message)
    money = has_money(message)
    sensitive = has_sensitive(message)
    url = has_url(message)
    contact = has_contact(message)
    phone = has_phone(message)

    return np.array([
        float(urgency),
        float(action),
        float(money),
        float(sensitive),
        float(url),
        float(contact),
        float(phone),
        float(urgency and sensitive),
        float(action and money),
        float(action and sensitive),
        float(urgency and money),
        float(phone and contact),
        float(phone and urgency),
        float(contact and urgency)
    ])


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

def logistic_probability(features, coefficients, intercept):
    """
    Calcola la probabilità della classe SPAM
    con la formula della Logistic Regression.
    """

    score = np.dot(features, coefficients) + intercept

    # Calcoliamo la funzione sigmoide senza utilizzare SciPy.
    probability = 1 / (1 + math.exp(-score))

    return float(probability)


# ------------------------------------------------------------
# Analyse complète
# ------------------------------------------------------------

def analyze_message(message):
    """
    Esegue l'analisi completa del messaggio.
    """

    # Word TF-IDF.
    word_vector = transform_word(message)

    # Character TF-IDF.
    character_vector = transform_character(message)

    # V3 utilizza prima le feature Word e poi quelle Character.
    combined_vector = np.concatenate([
        word_vector,
        character_vector
    ])

    # Coefficienti del modello ML.
    ml_coefficients = np.array(
        ml_model_data["coefficients"],
        dtype=float
    )

    ml_intercept = float(
        ml_model_data["intercept"]
    )

    # Probabilità SPAM del modello ML.
    ml_probability = logistic_probability(
        combined_vector,
        ml_coefficients,
        ml_intercept
    )

    # Features del Rule Analyzer.
    rule_features = create_rule_features(message)

    rule_coefficients = np.array(
        rule_model_data["coefficients"],
        dtype=float
    )

    rule_intercept = float(
        rule_model_data["intercept"]
    )

    # Probabilité SPAM del Rule Analyzer.
    rule_probability = logistic_probability(
        rule_features,
        rule_coefficients,
        rule_intercept
    )

    # Recupero della configurazione V3.
    alpha = float(model_config["alpha"])
    rule_weight = float(model_config["rule_weight"])
    threshold = float(model_config["threshold"])

    # Calcul du score hybride.
    hybrid_score = (
        alpha * ml_probability
        + rule_weight * rule_probability
    )

    # Décision finale.
    is_spam = hybrid_score >= threshold

    return {
        "ml_probability": ml_probability,
        "rule_probability": rule_probability,
        "score": hybrid_score * 100,
        "is_spam": is_spam,
        "threshold": threshold * 100
    }


# ------------------------------------------------------------
# Modèle de requête
# ------------------------------------------------------------

class MessageRequest(BaseModel):
    message: str


# ------------------------------------------------------------
# Route principale
# ------------------------------------------------------------

@app.post("/analyze")
def analyze(request: MessageRequest):

    # Vérifie que le message n'est pas vide.
    if not request.message.strip():
        return {
            "error": "Il messaggio non può essere vuoto."
        }

    return analyze_message(request.message)


# ------------------------------------------------------------
# Route de vérification
# ------------------------------------------------------------

@app.get("/")
def root():

    return {
        "application": "ScamShield API",
        "status": "online"
    }