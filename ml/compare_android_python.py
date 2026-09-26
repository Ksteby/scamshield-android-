import json
import re
import numpy as np


# ============================================================
# MESSAGE DI TEST
# ============================================================

message = (
    "Servizio Sanitario: Si prega di contattare la propria sede ASL di riferimento al numero 89347782 per comunicazioni urgenti che la riguardano."
)


# ============================================================
# CHARICAMENTO DEI PARAMETRI
# ============================================================

with open(
    "models/tfidf_vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    vocabulary = json.load(file)


with open(
    "models/tfidf_idf.json",
    "r",
    encoding="utf-8"
) as file:
    idf = np.array(json.load(file))


with open(
    "models/ml_model.json",
    "r",
    encoding="utf-8"
) as file:
    ml_model = json.load(file)


with open(
    "models/rule_model.json",
    "r",
    encoding="utf-8"
) as file:
    rule_model = json.load(file)


with open(
    "models/model_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)


# ============================================================
# 1. RIPRODUZIONE  DEL TF-IDF
# ============================================================

text = message.lower()


tokens = re.findall(
    r"[\w]{2,}",
    text,
    flags=re.UNICODE
)


ngrams = list(tokens)

for i in range(len(tokens) - 1):
    ngrams.append(
        f"{tokens[i]} {tokens[i + 1]}"
    )


# Vettore TF.
tf = np.zeros(len(idf), dtype=float)

for ngram in ngrams:

    if ngram in vocabulary:

        index = vocabulary[ngram]

        tf[index] += 1.0


# TF × IDF
tfidf = tf * idf


# Normalizzazione L2.
norm = np.linalg.norm(tfidf)

if norm > 0:
    tfidf = tfidf / norm


# ============================================================
# 2. MODELLO ML
# ============================================================

ml_coefficients = np.array(
    ml_model["coefficients"]
)

ml_intercept = ml_model["intercept"]


# Score linéaire.
z_ml = (
    ml_intercept
    + np.dot(
        ml_coefficients,
        tfidf
    )
)


# Fonction sigmoïde.
ml_probability = (
    1.0
    / (1.0 + np.exp(-z_ml))
)


# ============================================================
# 3. FEATURES DEL RULE ANALYZER V2
# ============================================================

def has_urgency(text):
    pattern = (
        r"\burgente\b|\burgenti\b|\burgent\b|\bsubito\b|\bimmediatamente\b|"
        r"\bscadenza\b|\boggi\b|\bora\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b"
    )
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_action(text):
    pattern = (
        r"\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|"
        r"\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|"
        r"\bscarica\b|\bdownload\b|\bapri\b|\bopen\b"
    )
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_money(text):
    pattern = r"€|\$|£|\beuro\b|\beuros\b"
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_sensitive(text):
    pattern = (
        r"\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|"
        r"\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|"
        r"\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b"
    )
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_url(text):
    pattern = r"https?://|www\."
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_contact(text):
    pattern = (
        r"\bcontatta\b|\bcontattare\b|\bcontattaci\b|\bcontattatemi\b|"
        r"\bchiamaci\b|\bchiamare\b|\bnumero\b|\bsede\b|\bufficio\b"
    )
    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_phone(text):

    pattern = r"\b\d{6,}\b"
    return bool(
        re.search(
            pattern,
            text
        )
    )




urgency = int(has_urgency(message))
action = int(has_action(message))
money = int(has_money(message))
sensitive = int(has_sensitive(message))
url = int(has_url(message))
contact = int(has_contact(message))
phone = int(has_phone(message))


# Calcolo delle interaction features utilizzate dal modello V2.

action_sensitive = action * sensitive
urgency_sensitive = urgency * sensitive
contact_urgency = contact * urgency
action_money = action * money
phone_contact = phone * contact
phone_urgency = phone * urgency
urgency_money = urgency * money




rule_features = np.array([
    urgency,
    action,
    money,
    sensitive,
    url,
    contact,
    action_sensitive,
    urgency_sensitive,
    contact_urgency,
    action_money,
    phone,
    phone_contact,
    phone_urgency,
    urgency_money
], dtype=float)


# ============================================================
# 4. RULE MODEL
# ============================================================

rule_coefficients = np.array(
    rule_model["coefficients"]
)

rule_intercept = rule_model["intercept"]


z_rule = (
    rule_intercept
    + np.dot(
        rule_coefficients,
        rule_features
    )
)


rule_probability = (
    1.0
    / (1.0 + np.exp(-z_rule))
)


# ============================================================
# 5. SISTEMA IBRIDE
# ============================================================

alpha = config["alpha"]
rule_weight = config["rule_weight"]

hybrid_score = (
    alpha * ml_probability
    + rule_weight * rule_probability
)


is_spam = (
    hybrid_score >= config["threshold"]
)


# ============================================================
# 6. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("COMPARAISON PYTHON")
print("=" * 60)

print(f"\nMessage :\n{message}")

print("\nFeatures Rule :")
print(rule_features)

print("\n--- SCORES ---")

print(
    f"ML probability     : {ml_probability:.12f}"
)

print(
    f"Rule probability   : {rule_probability:.12f}"
)

print(
    f"Hybrid score       : {hybrid_score:.12f}"
)

print(
    f"Hybrid score /100  : {hybrid_score * 100:.4f}"
)

print(
    f"Threshold          : {config['threshold']:.12f}"
)

print(
    f"Decision           : {'SPAM' if is_spam else 'HAM'}"
)

print("\n" + "=" * 60)