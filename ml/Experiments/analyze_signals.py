import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. CARICAMENTO DEI DATASETS
# ============================================================

english_path = "data/sms+spam+collection/SMSSpamCollection"
italian_path = "data/italian_sms.csv"


english_df = pd.read_csv(
    english_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

english_df["language"] = "english"


italian_df = pd.read_csv(
    italian_path
)

italian_df = italian_df.rename(
    columns={"text": "message"}
)

italian_df["label"] = italian_df["labels"].map({
    0: "ham",
    1: "spam"
})

italian_df["language"] = "italian"


# ============================================================
# 2. CAMPIONI ITALIANI
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
# 3. FUSION
# ============================================================

combined_df = pd.concat(
    [english_df, italian_sample],
    ignore_index=True
)

combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 4. TRAIN / TEST
# ============================================================

train_df, test_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"]
)


# ============================================================
# 5. TF-IDF + MODELLO
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(
    train_df["message"]
)

X_test = vectorizer.transform(
    test_df["message"]
)


model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    train_df["label"]
)


# ============================================================
# 6. IDENTIFICAZIONE DEI FN
# ============================================================

predictions = model.predict(X_test)

results = test_df.reset_index(drop=True).copy()

results["prediction"] = predictions

false_negatives = results[
    (results["label"] == "spam") &
    (results["prediction"] == "ham")
].copy()


# ============================================================
# 7. Segnali cercati
# ============================================================

signals = [
    "urgente",
    "urgent",
    "subito",
    "clicca",
    "link",
    "banca",
    "conto",
    "bloccato",
    "bloccata",
    "password",
    "codice",
    "credenziali",
    "premio",
    "vincita",
    "gratis",
    "offerta",
    "conferma",
    "verifica",
    "scarica",
    "chiama",
    "entro",
    "24h",
    "12h",
    "6h"
]


# ============================================================
# 8. Contaggio dei segnali nei fn
# ============================================================

print("\n==============================================")
print("SIGNALI DEI FN")
print("==============================================")

for signal in signals:

    count = false_negatives[
        false_negatives["message"]
        .str.contains(
            signal,
            case=False,
            na=False
        )
    ].shape[0]

    print(
        f"{signal:15} : {count}"
    )