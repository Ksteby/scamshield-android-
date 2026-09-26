import pandas as pd
import re

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CARICAMENTO DEI DATI
# ============================================================

results = pd.read_csv("data/test_results.csv")


# ============================================================
# 2. DEFINIZIONE DEI SEGNALI
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


# ============================================================
# 3. CREAZIONE DI VARIABILI
# ============================================================

results["urgency"] = results["message"].apply(has_urgency)
results["action"] = results["message"].apply(has_action)
results["money"] = results["message"].apply(has_money)
results["sensitive"] = results["message"].apply(has_sensitive)
results["url"] = results["message"].apply(has_url)


# ============================================================
# 4. INTERACTZIONI TRA SEGNALI
# ============================================================

# consentono al modello di apprendere che una
#combinazione di due segnali può essere più interessante
#rispetto a ciascun segnale considerato separatamente.


results["urgency_money"] = (
    results["urgency"] & results["money"]
).astype(int)

results["urgency_sensitive"] = (
    results["urgency"] & results["sensitive"]
).astype(int)

results["action_sensitive"] = (
    results["action"] & results["sensitive"]
).astype(int)

results["action_money"] = (
    results["action"] & results["money"]
).astype(int)


# ============================================================
# 5. VARIABILE D'INGRESSO
# ============================================================

features = [
    "urgency",
    "action",
    "money",
    "sensitive",
    "url",
    "urgency_money",
    "urgency_sensitive",
    "action_sensitive",
    "action_money",
]

X = results[features].astype(int)

# spam = 1
# ham  = 0
y = (results["label"] == "spam").astype(int)


# ============================================================
# 6. ADDESTTRAMENTO DEI PESI
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X, y)


# ============================================================
# 7. VISUALIZZAZIONE DEI COEF.
# ============================================================

print("\n" + "=" * 70)
print("PESI IMPARATI DAL MODELLO")
print("=" * 70)

coefficients = model.coef_[0]

for feature, coefficient in sorted(
    zip(features, coefficients),
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(f"{feature:<20} : {coefficient:+.4f}")


# ============================================================
# 8. PROBABILITA DEL MODELLO DI REGOLE
# ============================================================

rule_probability = model.predict_proba(X)[:, 1]

results["rule_probability"] = rule_probability


# ============================================================
# 9. EVALUAZIONE
# ============================================================

predictions = (rule_probability >= 0.50).astype(int)

tp = ((predictions == 1) & (y == 1)).sum()
fp = ((predictions == 1) & (y == 0)).sum()
fn = ((predictions == 0) & (y == 1)).sum()
tn = ((predictions == 0) & (y == 0)).sum()

accuracy = (tp + tn) / len(y)

precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)


print("\n" + "=" * 70)
print("PERFORMANZE DEL MODELLO DI REGOLE")
print("=" * 70)

print(f"TP        : {tp}")
print(f"FP        : {fp}")
print(f"FN        : {fn}")
print(f"TN        : {tn}")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


print("\n" + "=" * 70)
print("FINE")
print("=" * 70)