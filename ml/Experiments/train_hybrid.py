import pandas as pd
import numpy as np
import re
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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

italian_df["label"] = italian_df["labels"].map(
    {0: "ham", 1: "spam"}
)

italian_df["language"] = "italian"

italian_df = italian_df[
    ["message", "label", "language"]
]


# ============================================================
# 3. ECHANTILLONNAGE ITALIEN
# ============================================================

# Nous conservons exactement la stratégie utilisée
# précédemment :
# 2000 messages ham + 2000 messages spam.

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
# 4. DATASET FINAL
# ============================================================

combined_df = pd.concat(
    [english_df, italian_sample],
    ignore_index=True
)

combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(f"Nombre total de messages : {len(combined_df)}")

print("\nRépartition des classes :")
print(combined_df["label"].value_counts())


# ============================================================
# 5. SEPARATION DEVELOPPEMENT / TEST
# ============================================================

# 80 % des données servent au développement du modèle.
#
# 20 % sont conservées comme TEST FINAL.
#
# Le test ne sera utilisé à aucun moment pour :
# - entraîner un modèle ;
# - choisir alpha ;
# - choisir le seuil.

development_df, test_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"]
)


print("\n" + "=" * 80)
print("SEPARATION DEVELOPPEMENT / TEST")
print("=" * 80)

print(f"Développement : {len(development_df)}")
print(f"Test final    : {len(test_df)}")


# ============================================================
# 6. DEFINITION DES SIGNAUX DU RULE ANALYZER
# ============================================================

def has_urgency(text):
    pattern = (
        r"\burgente\b|\burgent\b|\bsubito\b|\bimmediatamente\b|"
        r"\bscadenza\b|\boggi\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b"
    )

    return bool(
        re.search(pattern, text, re.IGNORECASE)
    )


def has_action(text):
    pattern = (
        r"\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|"
        r"\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|"
        r"\bscarica\b|\bdownload\b|\bapri\b|\bopen\b"
    )

    return bool(
        re.search(pattern, text, re.IGNORECASE)
    )


def has_money(text):
    pattern = r"€|\$|£|\beuro\b|\beuros\b"

    return bool(
        re.search(pattern, text, re.IGNORECASE)
    )


def has_sensitive(text):
    pattern = (
        r"\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|"
        r"\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|"
        r"\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b"
    )

    return bool(
        re.search(pattern, text, re.IGNORECASE)
    )


def has_url(text):
    pattern = r"https?://|www\."

    return bool(
        re.search(pattern, text, re.IGNORECASE)
    )


# ============================================================
# 7. CREATION DES VARIABLES DE REGLES
# ============================================================

def create_rule_features(df):

    features = pd.DataFrame(index=df.index)

    features["urgency"] = df["message"].apply(
        has_urgency
    ).astype(int)

    features["action"] = df["message"].apply(
        has_action
    ).astype(int)

    features["money"] = df["message"].apply(
        has_money
    ).astype(int)

    features["sensitive"] = df["message"].apply(
        has_sensitive
    ).astype(int)

    features["url"] = df["message"].apply(
        has_url
    ).astype(int)

    # --------------------------------------------------------
    # Interactions entre les signaux.
    #
    # Elles permettent au modèle d'apprendre qu'une
    # combinaison de signaux peut être plus informative
    # qu'un signal isolé.
    # --------------------------------------------------------

    features["urgency_sensitive"] = (
        features["urgency"] &
        features["sensitive"]
    ).astype(int)

    features["action_money"] = (
        features["action"] &
        features["money"]
    ).astype(int)

    features["action_sensitive"] = (
        features["action"] &
        features["sensitive"]
    ).astype(int)

    features["urgency_money"] = (
        features["urgency"] &
        features["money"]
    ).astype(int)

    return features


# ============================================================
# 8. FONCTION D'EVALUATION
# ============================================================

def evaluate(y_true, predictions):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "confusion": confusion_matrix(
            y_true,
            predictions
        )
    }


# ============================================================
# 9. LABELS
# ============================================================

y_development = (
    development_df["label"] == "spam"
).astype(int).to_numpy()

y_test = (
    test_df["label"] == "spam"
).astype(int).to_numpy()


# ============================================================
# 10. PREPARATION DE LA CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION")
print("=" * 80)

# Nous allons produire une prédiction pour chaque message
# du développement, mais cette prédiction sera toujours
# produite par un modèle qui N'A PAS été entraîné sur ce
# message.
#
# Ces prédictions sont appelées "out-of-fold predictions".

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Ces tableaux contiendront les probabilités produites
# par les modèles sur les données de validation de chaque fold.

oof_ml_probability = np.zeros(
    len(development_df)
)

oof_rule_probability = np.zeros(
    len(development_df)
)


# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

for fold_number, (train_indices, validation_indices) in enumerate(
    skf.split(
        development_df["message"],
        y_development
    ),
    start=1
):

    print("\n" + "-" * 80)
    print(f"FOLD {fold_number}/5")
    print("-" * 80)

    # --------------------------------------------------------
    # Données du fold
    # --------------------------------------------------------

    fold_train = development_df.iloc[
        train_indices
    ]

    fold_validation = development_df.iloc[
        validation_indices
    ]

    y_fold_train = (
        fold_train["label"] == "spam"
    ).astype(int)

    # ========================================================
    # 11.1 TF-IDF DU FOLD
    # ========================================================

    # IMPORTANT :
    # Le vectorizer est créé et entraîné À L'INTÉRIEUR
    # du fold.
    #
    # Il ne voit donc jamais les messages du fold de
    # validation pendant son apprentissage.

    fold_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    fold_X_train = fold_vectorizer.fit_transform(
        fold_train["message"]
    )

    fold_X_validation = fold_vectorizer.transform(
        fold_validation["message"]
    )


    # ========================================================
    # 11.2 MODELE ML DU FOLD
    # ========================================================

    fold_ml_model = LogisticRegression(
        max_iter=1000
    )

    fold_ml_model.fit(
        fold_X_train,
        y_fold_train
    )

    fold_ml_probability = fold_ml_model.predict_proba(
        fold_X_validation
    )[:, 1]


    # ========================================================
    # 11.3 MODELE DES REGLES DU FOLD
    # ========================================================

    fold_train_rules = create_rule_features(
        fold_train
    )

    fold_validation_rules = create_rule_features(
        fold_validation
    )

    fold_rule_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    fold_rule_model.fit(
        fold_train_rules,
        y_fold_train
    )

    fold_rule_probability = fold_rule_model.predict_proba(
        fold_validation_rules
    )[:, 1]


    # ========================================================
    # 11.4 STOCKAGE DES PREDICTIONS OUT-OF-FOLD
    # ========================================================

    oof_ml_probability[validation_indices] = (
        fold_ml_probability
    )

    oof_rule_probability[validation_indices] = (
        fold_rule_probability
    )


    print(
        f"Messages entraînement : {len(train_indices)}"
    )

    print(
        f"Messages validation   : {len(validation_indices)}"
    )


print("\nCross-validation terminée.")


# ============================================================
# 12. VERIFICATION DES PREDICTIONS OOF
# ============================================================

# Chaque message du développement doit maintenant avoir
# exactement une prédiction ML et une prédiction Rule.
#
# Cela constitue notre ensemble de validation "virtuel"
# pour sélectionner alpha et le seuil.

if np.any(oof_ml_probability == 0):
    print(
        "\nAVERTISSEMENT : certaines probabilités ML valent 0."
    )

if np.any(oof_rule_probability == 0):
    print(
        "\nAVERTISSEMENT : certaines probabilités Rule valent 0."
    )


# ============================================================
# 13. RECHERCHE DE ALPHA ET DU SEUIL
# ============================================================

print("\n" + "=" * 80)
print("RECHERCHE DE ALPHA ET DU SEUIL")
print("=" * 80)


def find_best_threshold(y_true, scores):
    """
    Recherche efficacement le meilleur seuil pour une série
    de probabilités.

    Au lieu de tester chaque seuil avec plusieurs fonctions
    sklearn, nous trions les scores puis calculons les
    métriques avec des sommes cumulées NumPy.

    Cela donne exactement le même principe de recherche :
    chaque seuil pertinent correspond à un score réellement
    produit par le modèle.
    """

    # Tri des scores du plus grand au plus petit.
    order = np.argsort(-scores)

    sorted_scores = scores[order]
    sorted_y = y_true[order]

    # Après avoir classé les messages par score décroissant :
    #
    # à la position i, tous les messages de 0 à i sont
    # considérés comme spam.

    true_positives = np.cumsum(
        sorted_y
    )

    false_positives = np.cumsum(
        1 - sorted_y
    )

    total_positives = np.sum(
        y_true
    )

    false_negatives = (
        total_positives - true_positives
    )

    # --------------------------------------------------------
    # On ne teste qu'une seule fois chaque seuil distinct.
    #
    # Si plusieurs messages ont exactement le même score,
    # le seuil correspondant doit inclure tous ces messages.
    # --------------------------------------------------------

    unique_score_indices = np.r_[
        np.where(
            np.diff(sorted_scores) != 0
        )[0],
        len(sorted_scores) - 1
    ]

    tp = true_positives[
        unique_score_indices
    ]

    fp = false_positives[
        unique_score_indices
    ]

    fn = false_negatives[
        unique_score_indices
    ]

    # --------------------------------------------------------
    # Calcul vectorisé de précision, rappel et F1.
    # --------------------------------------------------------

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fp) != 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fn) != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            precision,
            dtype=float
        ),
        where=(precision + recall) != 0
    )

    # --------------------------------------------------------
    # Sélection :
    #
    # 1. F1 maximal
    # 2. précision maximale en cas d'égalité
    # 3. rappel maximal en cas de nouvelle égalité
    # --------------------------------------------------------

    best_f1 = np.max(f1)

    best_indices = np.where(
        np.isclose(
            f1,
            best_f1
        )
    )[0]

    # Parmi les configurations ayant le meilleur F1,
    # nous conservons celle ayant la meilleure précision.

    best_precision = np.max(
        precision[best_indices]
    )

    best_indices = best_indices[
        np.isclose(
            precision[best_indices],
            best_precision
        )
    ]

    # En cas d'égalité supplémentaire, on choisit
    # le meilleur rappel.

    best_recall = np.max(
        recall[best_indices]
    )

    best_indices = best_indices[
        np.isclose(
            recall[best_indices],
            best_recall
        )
    ]

    best_index = best_indices[0]

    best_threshold = sorted_scores[
        unique_score_indices[best_index]
    ]

    return (
        best_threshold,
        best_f1,
        best_precision,
        best_recall
    )


# ============================================================
# Recherche de alpha
# ============================================================

# alpha représente le poids du modèle ML :
#
# Score hybride =
#     alpha * score_ML
#     +
#     (1 - alpha) * score_RULES
#
# Nous explorons tout l'intervalle [0, 1].
#
# 1001 valeurs correspondent à un pas de 0.001.
# Le but est simplement d'obtenir une recherche suffisamment
# fine ; aucune valeur particulière n'est supposée meilleure
# à l'avance.

alphas = np.linspace(
    0.0,
    1.0,
    1001
)


best_alpha = None
best_threshold = None
best_f1 = -1.0
best_precision = -1.0
best_recall = -1.0


for alpha in alphas:

    rule_weight = 1.0 - alpha

    # Score hybride calculé uniquement sur les prédictions
    # out-of-fold du développement.

    hybrid_oof_score = (
        alpha * oof_ml_probability
        +
        rule_weight * oof_rule_probability
    )

    (
        threshold,
        current_f1,
        current_precision,
        current_recall
    ) = find_best_threshold(
        y_development,
        hybrid_oof_score
    )

    # --------------------------------------------------------
    # Comparaison avec la meilleure configuration trouvée.
    # --------------------------------------------------------

    is_better = False

    if current_f1 > best_f1:

        is_better = True

    elif np.isclose(
        current_f1,
        best_f1
    ):

        if current_precision > best_precision:

            is_better = True

        elif np.isclose(
            current_precision,
            best_precision
        ):

            if current_recall > best_recall:

                is_better = True

    if is_better:

        best_f1 = current_f1
        best_precision = current_precision
        best_recall = current_recall

        best_alpha = alpha
        best_threshold = threshold


# ============================================================
# 14. CONFIGURATION RETENUE
# ============================================================

best_rule_weight = (
    1.0 - best_alpha
)


print("\n" + "-" * 80)
print("CONFIGURATION OPTIMALE")
print("-" * 80)

print(
    f"Poids ML       : {best_alpha:.3f}"
)

print(
    f"Poids règles   : {best_rule_weight:.3f}"
)

print(
    f"Seuil          : {best_threshold:.6f}"
)

print(
    f"F1-score OOF   : {best_f1:.4f}"
)

print(
    f"Precision OOF  : {best_precision:.4f}"
)

print(
    f"Recall OOF     : {best_recall:.4f}"
)


# ============================================================
# 15. ENTRAINEMENT FINAL SUR LES 80 % DE DEVELOPPEMENT
# ============================================================

print("\n" + "=" * 80)
print("REENTRAINEMENT FINAL SUR LES DONNEES DE DEVELOPPEMENT")
print("=" * 80)

# Maintenant que alpha et le seuil sont fixés, nous pouvons
# utiliser les 80 % de développement dans leur totalité
# pour entraîner les modèles finaux.
#
# Le test reste totalement indépendant.

final_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

final_X_development = final_vectorizer.fit_transform(
    development_df["message"]
)

final_X_test = final_vectorizer.transform(
    test_df["message"]
)


# ------------------------------------------------------------
# Modèle ML final
# ------------------------------------------------------------

final_ml_model = LogisticRegression(
    max_iter=1000
)

final_ml_model.fit(
    final_X_development,
    y_development
)


final_test_ml_probability = (
    final_ml_model.predict_proba(
        final_X_test
    )[:, 1]
)


# ------------------------------------------------------------
# Modèle Rule final
# ------------------------------------------------------------

development_rules = create_rule_features(
    development_df
)

test_rules = create_rule_features(
    test_df
)


final_rule_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

final_rule_model.fit(
    development_rules,
    y_development
)


final_test_rule_probability = (
    final_rule_model.predict_proba(
        test_rules
    )[:, 1]
)


# ============================================================
# 16. COEFFICIENTS DU RULE ANALYZER
# ============================================================

print("\n" + "=" * 80)
print("COEFFICIENTS DU RULE ANALYZER FINAL")
print("=" * 80)

for feature, coefficient in sorted(
    zip(
        development_rules.columns,
        final_rule_model.coef_[0]
    ),
    key=lambda x: abs(x[1]),
    reverse=True
):

    print(
        f"{feature:<25} : {coefficient:+.4f}"
    )


# ============================================================
# 17. MODELE ML SEUL SUR LE TEST
# ============================================================

# Pour comparer correctement l'hybride au ML seul,
# le modèle ML seul utilise son seuil standard de 0.50.

final_test_ml_prediction = (
    final_test_ml_probability >= 0.50
).astype(int)


ml_test_metrics = evaluate(
    y_test,
    final_test_ml_prediction
)


# ============================================================
# 18. SYSTEME HYBRIDE SUR LE TEST
# ============================================================

# Le poids et le seuil ont été déterminés AVANT d'accéder
# au test.
#
# Nous pouvons maintenant appliquer la configuration finale.

final_test_hybrid_score = (
    best_alpha * final_test_ml_probability
    +
    best_rule_weight * final_test_rule_probability
)


final_test_hybrid_prediction = (
    final_test_hybrid_score >= best_threshold
).astype(int)


hybrid_test_metrics = evaluate(
    y_test,
    final_test_hybrid_prediction
)


# ============================================================
# 19. RESULTATS FINAUX
# ============================================================

print("\n" + "=" * 80)
print("RESULTATS FINAUX SUR LE TEST INDEPENDANT")
print("=" * 80)


print("\n--- ML SEUL ---")

print(
    f"Accuracy  : "
    f"{ml_test_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{ml_test_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{ml_test_metrics['recall']:.4f}"
)

print(
    f"F1-score  : "
    f"{ml_test_metrics['f1']:.4f}"
)

print("\nMatrice de confusion :")

print(
    ml_test_metrics["confusion"]
)


print("\n--- SYSTEME HYBRIDE ---")

print(
    f"Accuracy  : "
    f"{hybrid_test_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{hybrid_test_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{hybrid_test_metrics['recall']:.4f}"
)

print(
    f"F1-score  : "
    f"{hybrid_test_metrics['f1']:.4f}"
)

print("\nMatrice de confusion :")

print(
    hybrid_test_metrics["confusion"]
)


# ============================================================
# 20. COMPARAISON ML / HYBRIDE
# ============================================================

print("\n" + "=" * 80)
print("COMPARAISON FINALE")
print("=" * 80)

print(
    f"Delta Accuracy  : "
    f"{hybrid_test_metrics['accuracy'] - ml_test_metrics['accuracy']:+.4f}"
)

print(
    f"Delta Precision : "
    f"{hybrid_test_metrics['precision'] - ml_test_metrics['precision']:+.4f}"
)

print(
    f"Delta Recall    : "
    f"{hybrid_test_metrics['recall'] - ml_test_metrics['recall']:+.4f}"
)

print(
    f"Delta F1        : "
    f"{hybrid_test_metrics['f1'] - ml_test_metrics['f1']:+.4f}"
)


# ============================================================
# 21. SAUVEGARDE DES RESULTATS
# ============================================================

results = test_df.copy()

results["ml_probability"] = (
    final_test_ml_probability
)

results["rule_probability"] = (
    final_test_rule_probability
)

results["hybrid_probability"] = (
    final_test_hybrid_score
)

results["ml_prediction"] = np.where(
    final_test_ml_prediction == 1,
    "spam",
    "ham"
)

results["hybrid_prediction"] = np.where(
    final_test_hybrid_prediction == 1,
    "spam",
    "ham"
)


results.to_csv(
    "data/test_results_hybrid.csv",
    index=False,
    encoding="utf-8"
)


print("\n" + "=" * 80)
print("PIPELINE TERMINEE")
print("=" * 80)

print(
    "Résultats détaillés : "
    "data/test_results_hybrid.csv"
)

# ============================================================
# 22. EXPORT DU MODELE POUR ANDROID
# ============================================================

print("\n" + "=" * 80)
print("EXPORT DU MODELE POUR ANDROID")
print("=" * 80)

# Création du dossier de sortie s'il n'existe pas encore.
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)


# ------------------------------------------------------------
# 22.1 EXPORT DU VOCABULAIRE TF-IDF
# ------------------------------------------------------------

# Le vocabulaire associe chaque token/n-gramme à son index
# dans le vecteur TF-IDF.
with open(
    os.path.join(models_dir, "tfidf_vocabulary.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_vectorizer.vocabulary_,
        file,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 22.2 EXPORT DES IDF
# ------------------------------------------------------------

# Les valeurs IDF sont nécessaires pour reproduire le calcul
# TF-IDF directement dans l'application Android.
with open(
    os.path.join(models_dir, "tfidf_idf.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_vectorizer.idf_.tolist(),
        file
    )


# ------------------------------------------------------------
# 22.3 EXPORT DU MODELE ML
# ------------------------------------------------------------

# La Logistic Regression utilise :
#     score = intercept + somme(coefficient * feature)
#
# Puis :
#     probability = sigmoid(score)

ml_model_data = {
    "coefficients": final_ml_model.coef_[0].tolist(),
    "intercept": float(final_ml_model.intercept_[0]),
    "classes": final_ml_model.classes_.tolist()
}

with open(
    os.path.join(models_dir, "ml_model.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        ml_model_data,
        file,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 22.4 EXPORT DU RULE MODEL
# ------------------------------------------------------------

# Les noms des features doivent être conservés dans le même
# ordre que celui utilisé pendant l'entraînement.
rule_feature_names = development_rules.columns.tolist()

rule_model_data = {
    "features": rule_feature_names,
    "coefficients": final_rule_model.coef_[0].tolist(),
    "intercept": float(final_rule_model.intercept_[0]),
    "classes": final_rule_model.classes_.tolist()
}

with open(
    os.path.join(models_dir, "rule_model.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        rule_model_data,
        file,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 22.5 EXPORT DE LA CONFIGURATION HYBRIDE
# ------------------------------------------------------------

# Ces valeurs ont été déterminées par cross-validation
# sur l'ensemble de développement.
#
# Elles seront utilisées telles quelles dans Android.

config_data = {
    "alpha": float(best_alpha),
    "rule_weight": float(best_rule_weight),
    "threshold": float(best_threshold),

    "tfidf": {
        "lowercase": True,
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95
    }
}

with open(
    os.path.join(models_dir, "model_config.json"),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 22.6 CONFIRMATION
# ------------------------------------------------------------

print("\nFichiers du modèle exportés :")

print(
    os.path.join(
        models_dir,
        "tfidf_vocabulary.json"
    )
)

print(
    os.path.join(
        models_dir,
        "tfidf_idf.json"
    )
)

print(
    os.path.join(
        models_dir,
        "ml_model.json"
    )
)

print(
    os.path.join(
        models_dir,
        "rule_model.json"
    )
)

print(
    os.path.join(
        models_dir,
        "model_config.json"
    )
)

print("\nExport terminé.")