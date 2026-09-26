import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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
# 6. SEPARAZIONE TEXT / LABEL
# ============================================================

x_train = train_df["message"]
y_train = train_df["label"]

x_validation = validation_df["message"]
y_validation = validation_df["label"]

x_test = test_df["message"]
y_test = test_df["label"]

# ============================================================
# 7. CREAZIONE DEL VETTORE TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2, #per ignorare i termini che compaiono in meno di 2 documenti
    max_df=0.95 # se un termine compare in oltre 95% dei documenti, è probabile troppo frequente per essere davvero discriminante.
)

#Addestramento del vocabolario solo sul TRAIN
x_train_tfidf = vectorizer.fit_transform(x_train)

#Trasformazione della validazione e del test con lo stesso vocabolario
x_validation_tfidf = vectorizer.transform(x_validation)
x_test_tfidf = vectorizer.transform(x_test)

print("=== TF-IDF ===")
print("Numero di documenti:", x_train_tfidf.shape[0])
print("Numero di caratteri:", x_train_tfidf.shape[1])

# ============================================================
# 8.CREAZIONE DEL MODELLO
# ============================================================

model = LogisticRegression(
    max_iter=100
)


# ============================================================
#9.ADESTRAMENTO
# ============================================================

model.fit(
    x_train_tfidf,
    y_train
)

print("\n=== MODELLO ADESTRATO ===")
print("Adestramento completato con successo")

# ============================================================
# 10. PRIME PREVISIONI
# ============================================================

validation_prediction = model.predict(
    x_validation_tfidf
)

print("Primi previsioni :")
print(validation_prediction[:10])

# ============================================================
# 11. ÉVALUATION SUR LE JEU DE TEST
# ============================================================

# Le modèle prédit les classes des messages de test.
test_predictions = model.predict(
    x_test_tfidf
)


# ============================================================
# 12. CALCUL DES MÉTRIQUES
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    pos_label="spam"
)

recall = recall_score(
    y_test,
    test_predictions,
    pos_label="spam"
)

f1 = f1_score(
    y_test,
    test_predictions,
    pos_label="spam"
)


print("\n===== PERFORMANCE SUR LE TEST =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ============================================================
# 13. RAPPORT COMPLET
# ============================================================

print("\n===== RAPPORT DE CLASSIFICATION =====")

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ============================================================
# 14. MATRICE DE CONFUSION
# ============================================================

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=["ham", "spam"]
)

print("\n===== MATRICE DE CONFUSION =====")

print(matrix)