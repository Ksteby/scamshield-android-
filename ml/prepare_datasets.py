import pandas as pd

# ============================================================
# 1. CARICAMENTO DEL DATASET INGLESE
# ============================================================

english_path = "data/sms+spam+collection/SMSSpamCollection"

english_df = pd.read_csv(
    english_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

english_df = english_df[["message", "label"]]


# ============================================================
# 2. CARICAMENTO DEL DATASET ITALIANO
# ============================================================

italian_path = "data/italian_sms.csv"

italian_df = pd.read_csv(italian_path)

# Rinomina la colonna "text" in "message".
italian_df = italian_df.rename(
    columns={"text": "message"}
)

# Converte i labels numerici :
# 0 = ham
# 1 = spam
italian_df["label"] = italian_df["labels"].map({
    0: "ham",
    1: "spam"
})

italian_df = italian_df[["message", "label"]]


# ============================================================
# 3. CAMPIONAMENTO DEL DATASET ITALIANO
# ============================================================

# Selezioniamo 2 000 messagi ham
# e 2 000 messagi spam.
italian_ham = italian_df[
    italian_df["label"] == "ham"
].sample(
    n=2000,
    random_state=42 # consente di rendere il campionamento riproduttibile
)

italian_spam = italian_df[
    italian_df["label"] == "spam"
].sample(
    n=2000,
    random_state=42
)

# Fusione dei due campioni.
italian_sample = pd.concat(
    [italian_ham, italian_spam],
    ignore_index=True
)

# Mescola le righe in modo che ham e spam
# in siano ragruppati per blocco.
italian_sample = italian_sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 4. FUSIONE CON IL DATASET INGLESE
# ============================================================

combined_df = pd.concat(
    [english_df, italian_sample],
    ignore_index=True
)

# Missaggio finale del dataset.
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 5. Visualizzazione degli informazioni
# ============================================================

print("===== DATASET INGLESE =====")
print("Numero :", len(english_df))
print(english_df["label"].value_counts())


print("\n===== DATASET ITALIANO ORIGINALE=====")
print("Numero :", len(italian_df))
print(italian_df["label"].value_counts())


print("\n===== CAMPIONE ITALIANO =====")
print("Numero :", len(italian_sample))
print(italian_sample["label"].value_counts())


print("\n===== DATASET COMBINATO =====")
print("Numero :", len(combined_df))
print(combined_df["label"].value_counts())


print("\nPrimi righe :")
print(combined_df.head())