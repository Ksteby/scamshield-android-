import pandas as pd
import re


# ============================================================
# 1. CARICAMENTO DEI RISULTATI DEL TEST
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
# 3. CALCOLO DEI SEGNALI
# ============================================================

results["urgency"] = results["message"].apply(has_urgency)
results["action"] = results["message"].apply(has_action)
results["money"] = results["message"].apply(has_money)
results["sensitive"] = results["message"].apply(has_sensitive)
results["url"] = results["message"].apply(has_url)


# ============================================================
# 4. CREAZIONE DELLE REGOLE
# ============================================================

# Chaque règle est représentée par une fonction.
#
# Une fonction retourne True lorsque le message déclenche
# la règle et False dans le cas contraire.

rules = {

    # --------------------------------------------------------
    # Signali individuali
    # --------------------------------------------------------

    "urgency": lambda df: df["urgency"],

    "action": lambda df: df["action"],

    "money": lambda df: df["money"],

    "sensitive": lambda df: df["sensitive"],

    "url": lambda df: df["url"],


    # --------------------------------------------------------
    # Combinazioni AND
    # --------------------------------------------------------

    "urgency AND action": lambda df:
        df["urgency"] & df["action"],

    "action AND sensitive": lambda df:
        df["action"] & df["sensitive"],

    "action AND money": lambda df:
        df["action"] & df["money"],

    "urgency AND money": lambda df:
        df["urgency"] & df["money"],

    "urgency AND sensitive": lambda df:
        df["urgency"] & df["sensitive"],


    # --------------------------------------------------------
    # Combinazioni OR
    # --------------------------------------------------------

    "urgency OR action": lambda df:
        df["urgency"] | df["action"],

    "urgency OR money": lambda df:
        df["urgency"] | df["money"],

    "money OR sensitive": lambda df:
        df["money"] | df["sensitive"],

    "action OR money": lambda df:
        df["action"] | df["money"],

    "action OR sensitive": lambda df:
        df["action"] | df["sensitive"],


    # --------------------------------------------------------
    # combinazioni ibridi
    # --------------------------------------------------------

    # Argent OU combinaison urgence + données sensibles
    "money OR (urgency AND sensitive)": lambda df:
        df["money"] | (df["urgency"] & df["sensitive"]),

    # Argent OU action + données sensibles
    "money OR (action AND sensitive)": lambda df:
        df["money"] | (df["action"] & df["sensitive"]),

    # Action + argent OU urgence + données sensibles
    "(action AND money) OR (urgency AND sensitive)": lambda df:
        (df["action"] & df["money"]) |
        (df["urgency"] & df["sensitive"]),

    # URL + action OU urgence + argent
    "(url AND action) OR (urgency AND money)": lambda df:
        (df["url"] & df["action"]) |
        (df["urgency"] & df["money"]),
}


# ============================================================
# 5. EVALUAZIONE DELLE REGOLE
# ============================================================

print("\n" + "=" * 80)
print("Evaluazione delle regole come clasificatore")
print("=" * 80)


# Les valeurs réelles
# True  = spam
# False = ham

y_true = results["label"] == "spam"


for rule_name, rule_function in rules.items():

    # Risultato della regola :
    # True  = la regola considera il msg come spam
    # False = la regola considera il msg come ham
    y_pred = rule_function(results)

    # --------------------------------------------------------
    # CONFUSION MATRICE
    # --------------------------------------------------------

    tp = (y_pred & y_true).sum()
    fp = (y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()
    tn = (~y_pred & ~y_true).sum()

    # --------------------------------------------------------
    # METRICHE
    # --------------------------------------------------------

    accuracy = (tp + tn) / len(results)

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

    # --------------------------------------------------------
    #VISUALIZZAZIONE
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print(f"REGOLE : {rule_name}")
    print("-" * 80)

    print(f"TP : {tp}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"TN : {tn}")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")


print("\n" + "=" * 80)
print("FINE")
print("=" * 80)