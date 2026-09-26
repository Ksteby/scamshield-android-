import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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

# On ajoute la langue.
english_df["language"] = "english"

english_df = english_df[
    ["message", "label", "language"]
]


# ============================================================
# 2. CARICAMENTO DEI DATASET
# ============================================================

italian_path = "data/italian_sms.csv"

italian_df = pd.read_csv(italian_path)


italian_df = italian_df.rename(
    columns={"text": "message"}
)

# Conversion des labels :
# 0 -> ham
# 1 -> spam
italian_df["label"] = italian_df["labels"].map({
    0: "ham",
    1: "spam"
})


italian_df["language"] = "italian"

italian_df = italian_df[
    ["message", "label", "language"]
]


# ============================================================
# 3. SÉLEZIONE DEI MSG ITALIANI
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



italian_sample = italian_sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 4. FUSION INGLESE + ITALIANO
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
# 5. TRAIN / TEST
# ============================================================

train_df, test_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"]
)


# ============================================================
# 6. TRAIN / VALIDATION
# ============================================================

train_df, validation_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df["label"]
)


# ============================================================
# 7. TESTI E LABELS
# ============================================================

X_train = train_df["message"]
y_train = train_df["label"]

X_test = test_df["message"]
y_test = test_df["label"]


# ============================================================
# 8. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)


X_train_tfidf = vectorizer.fit_transform(X_train)

# I test utilizzano lo stesso vocabolario.
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 9. ADDESTRAMENTO DEL MODELLO
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 10. PRÉDICTIONS SUL TEST
# ============================================================

test_predictions = model.predict(
    X_test_tfidf
)


# ============================================================
# 11. FONCTIONE D'ÉVALUAZIONE
# ============================================================

def evaluate_dataset(name, real_labels, predictions):

    accuracy = accuracy_score(
        real_labels,
        predictions
    )

    precision = precision_score(
        real_labels,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        real_labels,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        real_labels,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    print(f"\n===== {name.upper()} =====")

    print(f"Nombre de messages : {len(real_labels)}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")


# ============================================================
# 12. ÉVALUAZIONE GLOBALE
# ============================================================

evaluate_dataset(
    "Test global",
    y_test,
    test_predictions
)


# ============================================================
# 13. ÉVALUAZIONE INGLESE
# ============================================================

english_mask = test_df["language"] == "english"

evaluate_dataset(
    "Test anglais",
    test_df.loc[english_mask, "label"],
    test_predictions[english_mask]
)


# ============================================================
# 14. ÉVALUAZIONE ITALIANA
# ============================================================

italian_mask = test_df["language"] == "italian"

evaluate_dataset(
    "Test italien",
    test_df.loc[italian_mask, "label"],
    test_predictions[italian_mask]
)