import pandas as pd
import re


# ============================================================
# 1. CARICAMENTO DEI RISULTATI ML
# ============================================================

results = pd.read_csv("data/test_results.csv")


# ============================================================
# 2. RILEVAMENTO DEI SEGNALI
# ============================================================

def has_urgency(text):
    pattern = (
        r"\burgente\b|\burgent\b|\bsubito\b|\bimmediatamente\b|"
        r"\bscadenza\b|\boggi\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_action(text):
    pattern = (
        r"\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|"
        r"\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|"
        r"\bscarica\b|\bdownload\b|\bapri\b|\bopen\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_money(text):
    pattern = r"€|\$|£|\beuro\b|\beuros\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_sensitive(text):
    pattern = (
        r"\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|"
        r"\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|"
        r"\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def has_url(text):
    pattern = r"https?://|www\."
    return bool(re.search(pattern, text, re.IGNORECASE))


# Calcola i segnali per ogni messaggio
results["urgency"] = results["message"].apply(has_urgency)
results["action"] = results["message"].apply(has_action)
results["money"] = results["message"].apply(has_money)
results["sensitive"] = results["message"].apply(has_sensitive)
results["url"] = results["message"].apply(has_url)


# ============================================================
# 3. CALCOLO DEL RULE SCORE
# ============================================================

def calculate_rule_score(row):
    """
    Calcola un punteggio semplice compreso tra 0 e 1.

    I segnali più affidabili secondo la nostra analisi
    ricevono un peso maggiore.

    IMPORTANTE:
    Questi pesi costituiscono una configurazione sperimentale
    utilizzata per confrontare il sistema ibrido. Non rappresentano
    i coefficienti della regressione logistica.
    """

    score = 0.0

    # Segnale monetario: molto preciso nella nostra analisi.
    if row["money"]:
        score += 0.30

    # Presenza di un URL: molto precisa ma poco frequente.
    if row["url"]:
        score += 0.25

    # Richiesta di un'azione: informativa ma meno precisa.
    if row["action"]:
        score += 0.15

    # Presenza di dati sensibili.
    if row["sensitive"]:
        score += 0.15

    # Urgenza: segnale utile ma con molti falsi positivi.
    if row["urgency"]:
        score += 0.10

    # Bonus quando più segnali compaiono contemporaneamente.
    if row["urgency"] and row["sensitive"]:
        score += 0.05

    # Limita il punteggio a 1.
    return min(score, 1.0)


results["rule_score"] = results.apply(
    calculate_rule_score,
    axis=1
)


# ============================================================
# 4. FUNZIONE DI VALUTAZIONE
# ============================================================

def evaluate(y_true, y_pred):

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    accuracy = (tp + tn) / len(y_true)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return tp, fp, fn, tn, accuracy, precision, recall, f1


# ============================================================
# 5. SOLO MODELLO ML
# ============================================================

y_true = (
    results["label"] == "spam"
).astype(int)

ml_probability = results["spam_probability"]


# Classificazione ML con soglia del 50%.
ml_prediction = (
    ml_probability >= 0.50
).astype(int)


ml_metrics = evaluate(
    y_true,
    ml_prediction
)


print("\n" + "=" * 80)
print("RIFERIMENTO : SOLO ML")
print("=" * 80)

print(f"TP        : {ml_metrics[0]}")
print(f"FP        : {ml_metrics[1]}")
print(f"FN        : {ml_metrics[2]}")
print(f"TN        : {ml_metrics[3]}")
print(f"Accuracy  : {ml_metrics[4]:.4f}")
print(f"Precision : {ml_metrics[5]:.4f}")
print(f"Recall    : {ml_metrics[6]:.4f}")
print(f"F1-score  : {ml_metrics[7]:.4f}")


# ============================================================
# 6. TEST DELLE DIVERSE CONFIGURAZIONI IBRIDE
# ============================================================

print("\n" + "=" * 80)
print("SISTEMA IBRIDO ML + RULE ANALYZER")
print("=" * 80)


# Variamo l'influenza del Rule Analyzer.
#
# Esempio:
#
# ML = 0.90
# Rules = 0.10
#
# significa che il modello ML rappresenta il 90 % del punteggio
# finale e le regole il 10 %.

configurations = [
    (0.95, 0.05),
    (0.90, 0.10),
    (0.85, 0.15),
    (0.80, 0.20),
]


for ml_weight, rule_weight in configurations:

    # --------------------------------------------------------
    # PUNTEGGIO IBRIDO
    # --------------------------------------------------------

    hybrid_score = (
        ml_weight * ml_probability
        + rule_weight * results["rule_score"]
    )

    # Classificazione finale.
    hybrid_prediction = (
        hybrid_score >= 0.50
    ).astype(int)

    metrics = evaluate(
        y_true,
        hybrid_prediction
    )

    print("\n" + "-" * 80)

    print(
        f"ML = {ml_weight:.0%} | "
        f"Rules = {rule_weight:.0%}"
    )

    print("-" * 80)

    print(f"TP        : {metrics[0]}")
    print(f"FP        : {metrics[1]}")
    print(f"FN        : {metrics[2]}")
    print(f"TN        : {metrics[3]}")
    print(f"Accuracy  : {metrics[4]:.4f}")
    print(f"Precision : {metrics[5]:.4f}")
    print(f"Recall    : {metrics[6]:.4f}")
    print(f"F1-score  : {metrics[7]:.4f}")


print("\n" + "=" * 80)
print("FINE DELL'ANALISI")
print("=" * 80)