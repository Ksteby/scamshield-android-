import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. CHARGEMENT DU DATASET ANGLAIS
# ============================================================

english_path = "data/sms+spam+collection/SMSSpamCollection"

english_df = pd.read_csv(
    english_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

english_df["language"] = "english"

english_df = english_df[
    ["message", "label", "language"]
]


# ============================================================
# 2. CHARGEMENT DU DATASET ITALIEN
# ============================================================

italian_path = "data/italian_sms.csv"

italian_df = pd.read_csv(italian_path)

italian_df = italian_df.rename(
    columns={"text": "message"}
)

italian_df["label"] = italian_df["labels"].map({
    0: "ham",
    1: "spam"
})

italian_df["language"] = "italian"

italian_df = italian_df[
    ["message", "label", "language"]
]


# ============================================================
# 3. ÉCHANTILLON ITALIEN
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
# 4. FUSION DES DATASETS
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
# 7. TF-IDF
# ============================================================

X_train = train_df["message"]
y_train = train_df["label"]

X_test = test_df["message"]
y_test = test_df["label"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Le vocabulaire est appris uniquement sur TRAIN.
X_train_tfidf = vectorizer.fit_transform(X_train)

# Le TEST utilise le vocabulaire déjà appris.
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 8. ENTRAÎNEMENT DU MODÈLE
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 9. PRÉDICTIONS ET PROBABILITÉS
# ============================================================

predictions = model.predict(X_test_tfidf)

probabilities = model.predict_proba(X_test_tfidf)

# On récupère l'index correspondant à la classe "spam".
spam_index = list(model.classes_).index("spam")

# Probabilité que chaque message soit un spam.
spam_probability = probabilities[:, spam_index]


# ============================================================
# 10. CRÉATION D'UN TABLEAU AVEC LES RÉSULTATS
# ============================================================

results = test_df.reset_index(drop=True).copy()

results["prediction"] = predictions

results["spam_probability"] = spam_probability


# ============================================================
# 11. FAUX NÉGATIFS
# ============================================================

# Vrai spam mais prédit comme ham.
false_negatives = results[
    (results["label"] == "spam") &
    (results["prediction"] == "ham")
].copy()


# ============================================================
# 12. FAUX POSITIFS
# ============================================================

# Vrai ham mais prédit comme spam.
false_positives = results[
    (results["label"] == "ham") &
    (results["prediction"] == "spam")
].copy()


# ============================================================
# 13. AFFICHAGE DES FAUX NÉGATIFS
# ============================================================

print("\n")
print("==============================================")
print("       FAUX NÉGATIFS - SPAMS NON DÉTECTÉS")
print("==============================================")

print(
    "Nombre de faux négatifs :",
    len(false_negatives)
)

# On affiche les erreurs les plus intéressantes :
# celles où le modèle était le plus proche de dire "spam".
false_negatives = false_negatives.sort_values(
    "spam_probability",
    ascending=False
)

for index, row in false_negatives.head(10).iterrows():

    print("\n--- Message", index, "---")

    print("Langue :", row["language"])

    print("Vrai label :", row["label"])

    print("Prédiction :", row["prediction"])

    print(
        f"Probabilité spam : "
        f"{row['spam_probability']:.2%}"
    )

    print("Message :", row["message"])


# ============================================================
# 14. AFFICHAGE DES FAUX POSITIFS
# ============================================================

print("\n")
print("==============================================")
print("       FAUX POSITIFS - HAM CLASSÉS SPAM")
print("==============================================")

print(
    "Nombre de faux positifs :",
    len(false_positives)
)

# On affiche les erreurs les plus proches de la frontière.
false_positives = false_positives.sort_values(
    "spam_probability"
)

for index, row in false_positives.head(10).iterrows():

    print("\n--- Message", index, "---")

    print("Langue :", row["language"])

    print("Vrai label :", row["label"])

    print("Prédiction :", row["prediction"])

    print(
        f"Probabilité spam : "
        f"{row['spam_probability']:.2%}"
    )

    print("Message :", row["message"])


# ============================================================
# 15. RÉPARTITION DES ERREURS PAR LANGUE
# ============================================================

print("\n")
print("==============================================")
print("          ERREURS PAR LANGUE")
print("==============================================")


print("\nFaux négatifs :")

print(
    false_negatives["language"].value_counts()
)


print("\nFaux positifs :")

print(
    false_positives["language"].value_counts()
)

# ============================================================
# 16. SAUVEGARDE DES RÉSULTATS
# ============================================================

# Sauvegarde tous les résultats du test dans un fichier CSV.
results.to_csv(
    "data/test_results.csv",
    index=False,
    encoding="utf-8"
)

print("\nLes résultats ont été sauvegardés dans :")
print("data/test_results.csv")