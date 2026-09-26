import pandas as pd
import re


# ============================================================
# 1. CARICAMENTO DEI RISULTATI DEL MODELLO ML
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
# 3. CREAZIONE DEI SEGNALI
# ============================================================

results["urgency"] = results["message"].apply(has_urgency)
results["action"] = results["message"].apply(has_action)
results["money"] = results["message"].apply(has_money)
results["sensitive"] = results["message"].apply(has_sensitive)
results["url"] = results["message"].apply(has_url)


# ============================================================
# 4. DEFINIZIONI DELLE REGOLE DA VERIFICARE
# ============================================================

rules = {

    
    "urgency": ["urgency"],

    "action": ["action"],

    "money": ["money"],

    "sensitive": ["sensitive"],

    "url": ["url"],

   
    "urgency + action": ["urgency", "action"],

    "action + sensitive": ["action", "sensitive"],

    "action + money": ["action", "money"],

    "urgency + money": ["urgency", "money"],

    "urgency + sensitive": ["urgency", "sensitive"],

    "urgency + action + sensitive": [
        "urgency",
        "action",
        "sensitive"
    ],

    "urgency + action + money": [
        "urgency",
        "action",
        "money"
    ],

    "action + sensitive + money": [
        "action",
        "sensitive",
        "money"
    ],
}


# ============================================================
# 5. EVALUAZIONE DELLE REGOLE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSE DES REGLES")
print("=" * 70)

print(f"\nNombre total de messages : {len(results)}")

# Nmero di fn prodotti dal modello ML
total_fn = (
    (results["label"] == "spam") &
    (results["prediction"] == "ham")
).sum()


total_ham = (results["label"] == "ham").sum()

print(f"Faux négatifs ML : {total_fn}")
print(f"Messages ham      : {total_ham}")


for rule_name, signals in rules.items():

    # Une règle est déclenchée uniquement si TOUS les signaux
    # qui la composent sont présents.
    rule_triggered = results[signals].all(axis=1)

    # --------------------------------------------------------
    #Quanti fn sono recuperati ?
    # --------------------------------------------------------

    fn_caught = (
        rule_triggered &
        (results["label"] == "spam") &
        (results["prediction"] == "ham")
    ).sum()

    # --------------------------------------------------------
    # Quanti falsi positivi in più genererebbe
    # questa regola?
    #
    # Qui si considerano i messaggi effettivamente spam che
    # attivano la regola.
    # --------------------------------------------------------

    fp_created = (
        rule_triggered &
        (results["label"] == "ham")
    ).sum()

    # --------------------------------------------------------
    # Tasso di recuperazione di fn
    # --------------------------------------------------------

    fn_recovery = (
        fn_caught / total_fn * 100
        if total_fn > 0
        else 0
    )

    # --------------------------------------------------------
    # Tausso di fp tra gli ham
    # --------------------------------------------------------

    false_positive_rate = (
        fp_created / total_ham * 100
        if total_ham > 0
        else 0
    )

    # --------------------------------------------------------
    # Numero totale dei messaggi attivando la regola
    # --------------------------------------------------------

    triggered = rule_triggered.sum()

    print("\n--------------------------------------------")
    print(f"Regola : {rule_name}")
    print("--------------------------------------------")

    print(f"Messages déclenchant la règle : {triggered}")
    print(f"FN récupérés                 : {fn_caught}/{total_fn}")
    print(f"Récupération des FN          : {fn_recovery:.2f}%")
    print(f"FP créés                     : {fp_created}")
    print(f"Tasso di FP tra gli ham     : {false_positive_rate:.2f}%")


print("\n" + "=" * 70)
print("FINE")
print("=" * 70)