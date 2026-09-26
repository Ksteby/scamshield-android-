import pandas as pd
import re


# ============================================================
# 1. CHARGEMENT DES RESULTATS
# ============================================================

results_path = "data/test_results_hybrid.csv"

df = pd.read_csv(
    results_path,
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("ANALYSE DES ERREURS DU SYSTEME HYBRIDE")
print("=" * 80)

print(f"Nombre de messages testés : {len(df)}")


# ============================================================
# 2. IDENTIFICATION DES ERREURS
# ============================================================

# Faux négatif :
# véritable spam mais prédit comme ham.

false_negatives = df[
    (df["label"] == "spam")
    & (df["hybrid_prediction"] == "ham")
].copy()


# Faux positif :
# véritable ham mais prédit comme spam.

false_positives = df[
    (df["label"] == "ham")
    & (df["hybrid_prediction"] == "spam")
].copy()


print("\n" + "-" * 80)
print("ERREURS DU SYSTEME HYBRIDE")
print("-" * 80)

print(
    f"Faux négatifs : {len(false_negatives)}"
)

print(
    f"Faux positifs : {len(false_positives)}"
)


# ============================================================
# 3. COMPARAISON AVEC LE MODELE ML SEUL
# ============================================================

# Faux négatifs du ML seul.
ml_false_negatives = df[
    (df["label"] == "spam")
    & (df["ml_prediction"] == "ham")
].copy()


# Faux positifs du ML seul.
ml_false_positives = df[
    (df["label"] == "ham")
    & (df["ml_prediction"] == "spam")
].copy()


# ------------------------------------------------------------
# Scams que l'hybride réussit à récupérer alors que le ML
# seul les ratait.
# ------------------------------------------------------------

rescued_spams = df[
    (df["label"] == "spam")
    & (df["ml_prediction"] == "ham")
    & (df["hybrid_prediction"] == "spam")
].copy()


# ------------------------------------------------------------
# Scams que le ML réussit mais que l'hybride rate.
# ------------------------------------------------------------

new_false_negatives = df[
    (df["label"] == "spam")
    & (df["ml_prediction"] == "spam")
    & (df["hybrid_prediction"] == "ham")
].copy()


# ------------------------------------------------------------
# Messages ham correctement classés par le ML mais
# transformés en faux positifs par l'hybride.
# ------------------------------------------------------------

new_false_positives = df[
    (df["label"] == "ham")
    & (df["ml_prediction"] == "ham")
    & (df["hybrid_prediction"] == "spam")
].copy()


print("\n" + "-" * 80)
print("EFFET DE L'HYBRIDE PAR RAPPORT AU ML SEUL")
print("-" * 80)

print(
    f"Scams récupérés par l'hybride : "
    f"{len(rescued_spams)}"
)

print(
    f"Scams nouvellement manqués : "
    f"{len(new_false_negatives)}"
)

print(
    f"Nouveaux faux positifs : "
    f"{len(new_false_positives)}"
)


# ============================================================
# 4. REPARTITION PAR LANGUE
# ============================================================

print("\n" + "-" * 80)
print("FAUX NEGATIFS PAR LANGUE")
print("-" * 80)

print(
    false_negatives["language"].value_counts()
)


print("\n" + "-" * 80)
print("FAUX POSITIFS PAR LANGUE")
print("-" * 80)

print(
    false_positives["language"].value_counts()
)


print("\n" + "-" * 80)
print("SCAMS RECUPERES PAR L'HYBRIDE")
print("-" * 80)

print(
    rescued_spams["language"].value_counts()
)


# ============================================================
# 5. DEFINITION DES SIGNAUX
# ============================================================

def has_urgency(text):

    pattern = (
        r"\burgente\b|\burgent\b|\bsubito\b|\bimmediatamente\b|"
        r"\bscadenza\b|\boggi\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b"
    )

    return bool(
        re.search(
            pattern,
            str(text),
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
            str(text),
            re.IGNORECASE
        )
    )


def has_money(text):

    pattern = (
        r"€|\$|£|\beuro\b|\beuros\b"
    )

    return bool(
        re.search(
            pattern,
            str(text),
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
            str(text),
            re.IGNORECASE
        )
    )


def has_url(text):

    pattern = r"https?://|www\."

    return bool(
        re.search(
            pattern,
            str(text),
            re.IGNORECASE
        )
    )


def add_signals(dataframe):

    result = dataframe.copy()

    result["urgency"] = result["message"].apply(
        has_urgency
    )

    result["action"] = result["message"].apply(
        has_action
    )

    result["money"] = result["message"].apply(
        has_money
    )

    result["sensitive"] = result["message"].apply(
        has_sensitive
    )

    result["url"] = result["message"].apply(
        has_url
    )

    return result


# ============================================================
# 6. ANALYSE DES FAUX NEGATIFS
# ============================================================

fn_with_signals = add_signals(
    false_negatives
)


print("\n" + "=" * 80)
print("ANALYSE DES FAUX NEGATIFS")
print("=" * 80)


signal_columns = [
    "url",
    "money",
    "urgency",
    "action",
    "sensitive"
]


for signal in signal_columns:

    count = fn_with_signals[signal].sum()

    percentage = (
        count / len(fn_with_signals) * 100
        if len(fn_with_signals) > 0
        else 0
    )

    print(
        f"{signal:<12} : "
        f"{count:>3} / {len(fn_with_signals)} "
        f"({percentage:>5.1f} %)"
    )


# ============================================================
# 7. ANALYSE DES FAUX POSITIFS
# ============================================================

fp_with_signals = add_signals(
    false_positives
)


print("\n" + "=" * 80)
print("ANALYSE DES FAUX POSITIFS")
print("=" * 80)


for signal in signal_columns:

    count = fp_with_signals[signal].sum()

    percentage = (
        count / len(fp_with_signals) * 100
        if len(fp_with_signals) > 0
        else 0
    )

    print(
        f"{signal:<12} : "
        f"{count:>3} / {len(fp_with_signals)} "
        f"({percentage:>5.1f} %)"
    )


# ============================================================
# 8. ANALYSE DES SCAMS RECUPERES
# ============================================================

rescued_with_signals = add_signals(
    rescued_spams
)


print("\n" + "=" * 80)
print("SCAMS RECUPERES PAR L'HYBRIDE")
print("=" * 80)


for signal in signal_columns:

    count = rescued_with_signals[signal].sum()

    percentage = (
        count / len(rescued_with_signals) * 100
        if len(rescued_with_signals) > 0
        else 0
    )

    print(
        f"{signal:<12} : "
        f"{count:>3} / {len(rescued_with_signals)} "
        f"({percentage:>5.1f} %)"
    )


# ============================================================
# 9. PROBABILITES DES FAUX NEGATIFS
# ============================================================

print("\n" + "=" * 80)
print("PROBABILITES DES FAUX NEGATIFS")
print("=" * 80)

if len(false_negatives) > 0:

    print(
        false_negatives[
            [
                "language",
                "ml_probability",
                "rule_probability",
                "hybrid_probability"
            ]
        ].describe()
    )


# ============================================================
# 10. AFFICHAGE DES FAUX NEGATIFS
# ============================================================

print("\n" + "=" * 80)
print("FAUX NEGATIFS - MESSAGES")
print("=" * 80)

for index, row in false_negatives.iterrows():

    print("\n" + "-" * 80)

    print(
        f"Index : {index}"
    )

    print(
        f"Langue : {row['language']}"
    )

    print(
        f"ML probability    : "
        f"{row['ml_probability']:.4f}"
    )

    print(
        f"Rule probability  : "
        f"{row['rule_probability']:.4f}"
    )

    print(
        f"Hybrid probability: "
        f"{row['hybrid_probability']:.4f}"
    )

    print(
        f"Message : {row['message']}"
    )


# ============================================================
# 11. AFFICHAGE DES NOUVEAUX FAUX POSITIFS
# ============================================================

print("\n" + "=" * 80)
print("NOUVEAUX FAUX POSITIFS INTRODUITS PAR L'HYBRIDE")
print("=" * 80)

for index, row in new_false_positives.iterrows():

    print("\n" + "-" * 80)

    print(
        f"Index : {index}"
    )

    print(
        f"Langue : {row['language']}"
    )

    print(
        f"ML probability    : "
        f"{row['ml_probability']:.4f}"
    )

    print(
        f"Rule probability  : "
        f"{row['rule_probability']:.4f}"
    )

    print(
        f"Hybrid probability: "
        f"{row['hybrid_probability']:.4f}"
    )

    print(
        f"Message : {row['message']}"
    )


# ============================================================
# 12. RESUME FINAL
# ============================================================

print("\n" + "=" * 80)
print("RESUME")
print("=" * 80)

print(
    f"ML seul - faux négatifs : "
    f"{len(ml_false_negatives)}"
)

print(
    f"Hybride - faux négatifs : "
    f"{len(false_negatives)}"
)

print(
    f"Scams récupérés         : "
    f"{len(rescued_spams)}"
)

print(
    f"Nouveaux faux positifs  : "
    f"{len(new_false_positives)}"
)

print(
    f"Hybride - faux positifs : "
    f"{len(false_positives)}"
)

print("\nAnalyse terminée.")