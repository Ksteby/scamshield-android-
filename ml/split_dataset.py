import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CARICAMENTO DEL DATASET COMBINATO
# ============================================================

english_path = "data/sms+spam+collection/SMSSpamCollection"
italian_path = "data/italian_sms.csv"


# Dataset inglese
english_df = pd.read_csv(
    english_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

english_df = english_df[["message", "label"]]


# Dataset italiano
italian_df = pd.read_csv(italian_path)

italian_df = italian_df.rename(
    columns={"text": "message"}
)

italian_df["label"] = italian_df["labels"].map({
    0: "ham",
    1: "spam"
})

italian_df = italian_df[["message", "label"]]


# ============================================================
# 2. SELEZIONE DEI 4000 MESSAGGI ITALIANI
# ============================================================

italian_ham = italian_df[
    italian_df["label"] == "ham"
].sample(
    n=2000,
    random_state=42
)

italian_spam = italian_df[
    italian_df["label"] == "spam"
].sample(
    n=2000,
    random_state=42
)

italian_sample = pd.concat(
    [italian_ham, italian_spam],
    ignore_index=True
)


# ============================================================
# 3. FUSIONE DEI DATASETS
# ============================================================

combined_df = pd.concat(
    [english_df, italian_sample],
    ignore_index=True
)

# Miscelazione casuale ma riproductibile
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 4. FIRST SPLIT : TRAIN + TEST
# ============================================================

train_df, test_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"] # Per conservare circa la stessa proporzione tra ham e spam
)


# ============================================================
# 5. SECOND SPLIT : TRAIN + VALIDATION
# ============================================================

train_df, validation_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df["label"]
)


# ============================================================
# 6. VISUALIZZAZIONE DEI RISULTATI
# ============================================================

print("===== DATASET COMPLETO =====")
print("Numero :", len(combined_df))


print("\n===== TRAIN =====")
print("Numero :", len(train_df))
print(train_df["label"].value_counts())


print("\n===== VALIDATION =====")
print("Numero :", len(validation_df))
print(validation_df["label"].value_counts())


print("\n===== TEST =====")
print("Numero :", len(test_df))
print(test_df["label"].value_counts())