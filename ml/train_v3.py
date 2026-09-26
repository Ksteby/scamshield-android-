import pandas as pd
import numpy as np
import re
import json
import os

from scipy.sparse import hstack

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
# 1. CARICAMENTO DEL DATASET INGLESE
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
# 2. CARICAMENTO DEL DATASET ITALIANO
# ============================================================

italian_path = "data/italian_sms.csv"

italian_df = pd.read_csv(
    italian_path
)

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
# 3.CAMPIONAMENTO  ITALIANO
# ============================================================

# Manteniamo esattamente la stessa strategia utilizzata
# nelle precedenti versioni del modello.

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
# 4. DATASET FINALE
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
print("V3 - DATASET")
print("=" * 80)

print(
    f"Nombre total de messages : {len(combined_df)}"
)

print("\nRépartition des classes :")
print(
    combined_df["label"].value_counts()
)


# ============================================================
# 5. SEPARAZIONE DEVELOPPEMENT / TEST
# ============================================================


development_df, test_df = train_test_split(
    combined_df,
    test_size=0.20,
    random_state=42,
    stratify=combined_df["label"]
)


print("\n" + "=" * 80)
print("V3 - SEPARATION DEVELOPPEMENT / TEST")
print("=" * 80)

print(
    f"Développement : {len(development_df)}"
)

print(
    f"Test final    : {len(test_df)}"
)


# ============================================================
# 6. DEFINIZIONE DEI SEGNALI DEL RULE ANALYZER V2
# ============================================================

def has_urgency(text):
    # Rileva espressioni che possono creare un senso di urgenza.
    pattern = (
        r"\burgente\b|\burgenti\b|\burgent\b|\bsubito\b|"
        r"\bimmediatamente\b|\bscadenza\b|\boggi\b|\bora\b|"
        r"\bnow\b|\bimmediately\b|\btoday\b|\bentro\b"
    )

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_action(text):
    # Rileva inviti espliciti a compiere un'azione.
    pattern = (
        r"\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|"
        r"\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|"
        r"\bscarica\b|\bdownload\b|\bapri\b|\bopen\b"
    )

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_contact(text):
    # Rileva richieste esplicite di contatto o richiamata.
    pattern = (
        r"\bcontatta\b|\bcontattare\b|\bcontattami\b|"
        r"\bcontattaci\b|\bchiamare\b|\bchiamami\b|"
        r"\bchiama\b|\brichiamare\b|\brichiamami\b|"
        r"\bcall\b|\bcontact\b"
    )

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_phone(text):
    # Rileva un numero di telefono quando è associato
    # a parole come "numero", "telefono" o "cellulare".
    pattern = (
        r"\b(?:numero|num\.?|tel(?:efono)?|cellulare|mobile)"
        r"\s*(?:[:#-]\s*)?"
        r"(?:\+?\d[\d\s().-]{6,}\d)\b"
    )

    return bool(
        re.search(
            pattern,
            text
        )
    )


def has_money(text):
    # Rileva simboli e parole associate al denaro.
    pattern = r"€|\$|£|\beuro\b|\beuros\b"

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_sensitive(text):
    # Rileva riferimenti a dati bancari, credenziali
    # o informazioni personali sensibili.
    pattern = (
        r"\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|"
        r"\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|"
        r"\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b"
    )

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


def has_url(text):
    # Rileva la presenza di un URL.
    pattern = r"https?://|www\."

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )


# ============================================================
# 7. CREATZIONE DELLE VARIABILE DELLA REGOLA
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

    features["contact"] = df["message"].apply(
        has_contact
    ).astype(int)

    features["phone"] = df["message"].apply(
        has_phone
    ).astype(int)


    # --------------------------------------------------------
    # Interazioni tra i segnali.
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

    features["phone_contact"] = (
        features["phone"] &
        features["contact"]
    ).astype(int)

    features["phone_urgency"] = (
        features["phone"] &
        features["urgency"]
    ).astype(int)

    features["contact_urgency"] = (
        features["contact"] &
        features["urgency"]
    ).astype(int)

    return features


# ============================================================
# 8. FONCTIONE D'EVALUAZIONE
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
# 10. PREPARAZIONE DELLA CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("V3 - 5-FOLD CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Probabilita OOF del modello ML.
oof_ml_probability = np.zeros(
    len(development_df)
)

# Probabilita OOF del Rule Analyzer.
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
    print(f"V3 - FOLD {fold_number}/5")
    print("-" * 80)


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
    # 11.1 TF-IDF PER PAROLE
    # ========================================================

    # Questa rappresentazione conserva gli unigrammi e i
    # bigrammi utilizzati nelle versioni precedenti.

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_word_train = word_vectorizer.fit_transform(
        fold_train["message"]
    )

    X_word_validation = word_vectorizer.transform(
        fold_validation["message"]
    )


    # ========================================================
    # 11.2 TF-IDF PER CARACTERI
    # ========================================================

    # La rappresentazione a caratteri permette di catturare
    # frammenti di parole, abbreviazioni, variazioni
    # ortografiche e strutture tipiche degli SMS.

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_char_train = char_vectorizer.fit_transform(
        fold_train["message"]
    )

    X_char_validation = char_vectorizer.transform(
        fold_validation["message"]
    )


    # ========================================================
    # 11.3 COMBINAZIONE DELLE DUE RAPPRESENTAZIONI
    # ========================================================

    # Le deux représentations sont concaténées horizontalement.
    #
    # Le modèle ML dispose ainsi :
    # - des informations lexicales des mots ;
    # - des informations morphologiques des caractères.

    X_train = hstack([
        X_word_train,
        X_char_train
    ]).tocsr()

    X_validation = hstack([
        X_word_validation,
        X_char_validation
    ]).tocsr()


    # ========================================================
    # 11.4 MODELLO ML
    # ========================================================

    fold_ml_model = LogisticRegression(
        max_iter=1000
    )

    fold_ml_model.fit(
        X_train,
        y_fold_train
    )

    fold_ml_probability = (
        fold_ml_model.predict_proba(
            X_validation
        )[:, 1]
    )


    # ========================================================
    # 11.5 MODELLO RULE ANALYZER
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

    fold_rule_probability = (
        fold_rule_model.predict_proba(
            fold_validation_rules
        )[:, 1]
    )


    # ========================================================
    # 11.6 STOCKAGGIO DELLE PREDIZIONI
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

    print(
        f"Features mots         : {X_word_train.shape[1]}"
    )

    print(
        f"Features caractères   : {X_char_train.shape[1]}"
    )

    print(
        f"Features totales      : {X_train.shape[1]}"
    )


print("\nCross-validation V3 terminée.")


# ============================================================
# 12. REicerca della soglia
# ============================================================

def find_best_threshold(y_true, scores):

    # Ordina i punteggi dal più alto al più basso.
    order = np.argsort(-scores)

    sorted_scores = scores[order]
    sorted_y = y_true[order]

    # Calcola cumulativamente veri positivi e falsi positivi.
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


    # Conserviamo una sola volta ogni valore di soglia distinto.

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


    # Calcolo della precisione.

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fp) != 0
    )


    # Calcolo del recall.

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(
            tp,
            dtype=float
        ),
        where=(tp + fn) != 0
    )


    # Calcolo dell'F1-score.

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            precision,
            dtype=float
        ),
        where=(precision + recall) != 0
    )


    # Selezione della configurazione migliore:
    # 1. F1 massimo
    # 2. precisione massima
    # 3. recall massimo

    best_f1 = np.max(
        f1
    )

    best_indices = np.where(
        np.isclose(
            f1,
            best_f1
        )
    )[0]


    best_precision = np.max(
        precision[best_indices]
    )

    best_indices = best_indices[
        np.isclose(
            precision[best_indices],
            best_precision
        )
    ]


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
# 13. Ricerca di aplha e soglia
# ============================================================

print("\n" + "=" * 80)
print("V3 - Ricerca di aplha e sogliaL")
print("=" * 80)


# alpha rappresenta il peso del modello ML.
#
# Il peso delle regole è:
#     1 - alpha

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


    # Calcolo del punteggio ibrido sulle predizioni OOF.

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
    # selezioniamo a miglior configurazione
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


best_rule_weight = (
    1.0 - best_alpha
)


# ============================================================
# 14. CONFIGURAZIONI  V3 RITENUE
# ============================================================

print("\n" + "-" * 80)
print("V3 - CONFIGURAZIONEIOPTIMALE")
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
# 15. READDESTRAMENTO FINALE
# ============================================================

print("\n" + "=" * 80)
print("V3 - READDESTRAMENTO FINALE")
print("=" * 80)


# ============================================================
# 15.1 TF-IDF FINAL words
# ============================================================

final_word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

final_X_word_development = (
    final_word_vectorizer.fit_transform(
        development_df["message"]
    )
)

final_X_word_test = (
    final_word_vectorizer.transform(
        test_df["message"]
    )
)


# ============================================================
# 15.2 TF-IDF  FINAL CHARACTERS
# ============================================================

final_char_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

final_X_char_development = (
    final_char_vectorizer.fit_transform(
        development_df["message"]
    )
)

final_X_char_test = (
    final_char_vectorizer.transform(
        test_df["message"]
    )
)


# ============================================================
# 15.3 COMBINAZIONE FINALE
# ============================================================

final_X_development = hstack([
    final_X_word_development,
    final_X_char_development
]).tocsr()

final_X_test = hstack([
    final_X_word_test,
    final_X_char_test
]).tocsr()


print(
    f"Features mots       : "
    f"{final_X_word_development.shape[1]}"
)

print(
    f"Features caractères : "
    f"{final_X_char_development.shape[1]}"
)

print(
    f"Features totales    : "
    f"{final_X_development.shape[1]}"
)


# ============================================================
# 15.4 MODELE ML FINALE
# ============================================================

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


# ============================================================
# 15.5 RULE MODEL FINALE
# ============================================================

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
# 16. COEFFICIENTS Del RULE ANALYZER
# ============================================================

print("\n" + "=" * 80)
print("V3 - COEFFICIENTS DEL RULE ANALYZER")
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
# 17. ML sul  TEST
# ============================================================

final_test_ml_prediction = (
    final_test_ml_probability >= 0.50
).astype(int)


ml_test_metrics = evaluate(
    y_test,
    final_test_ml_prediction
)


# ============================================================
# 18. Sistema ibride dul test
# ============================================================

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
# 19. Risultati finali
# ============================================================

print("\n" + "=" * 80)
print("V3 - Risultati finali sul test")
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

print("\n confusion matrice :")

print(
    ml_test_metrics["confusion"]
)


print("\n--- SISTEMA IBRIDE V3 ---")

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
# 20. CONFRONTO ML / HYBRID
# ============================================================

print("\n" + "=" * 80)
print("V3 - COMPARAISON FINALE")
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
# 21. TEST Del messaggio ASL
# ============================================================

print("\n" + "=" * 80)
print("V3 - TEST DU MESSAGE ASL")
print("=" * 80)


asl_message = (
    "Servizio Sanitario: Si prega di contattare la propria "
    "sede ASL di riferimento al numero 89347782 per "
    "comunicazioni urgenti che la riguardano."
)


# ------------------------------------------------------------
# TF-IDF words
# ------------------------------------------------------------

asl_word = final_word_vectorizer.transform(
    [asl_message]
)


# ------------------------------------------------------------
# TF-IDF characters
# ------------------------------------------------------------

asl_char = final_char_vectorizer.transform(
    [asl_message]
)


# ------------------------------------------------------------
# Combinazioni
# ------------------------------------------------------------

asl_X = hstack([
    asl_word,
    asl_char
]).tocsr()


# ------------------------------------------------------------
# Score ML
# ------------------------------------------------------------

asl_ml_probability = (
    final_ml_model.predict_proba(
        asl_X
    )[:, 1][0]
)


# ------------------------------------------------------------
# Features Rule
# ------------------------------------------------------------

asl_df = pd.DataFrame({
    "message": [asl_message]
})

asl_rule_features = create_rule_features(
    asl_df
)


# ------------------------------------------------------------
# Score Rule
# ------------------------------------------------------------

asl_rule_probability = (
    final_rule_model.predict_proba(
        asl_rule_features
    )[:, 1][0]
)


# ------------------------------------------------------------
# Score IBRIDE
# ------------------------------------------------------------

asl_hybrid_score = (
    best_alpha * asl_ml_probability
    +
    best_rule_weight * asl_rule_probability
)


asl_is_spam = (
    asl_hybrid_score >= best_threshold
)


print(
    f"\nMessage :\n{asl_message}"
)

print("\nFeatures Rule :")

print(
    asl_rule_features.iloc[0].to_numpy()
)


print("\n--- SCORES V3 ---")

print(
    f"ML score       : "
    f"{asl_ml_probability:.12f}"
)

print(
    f"Rule score     : "
    f"{asl_rule_probability:.12f}"
)

print(
    f"Hybrid score   : "
    f"{asl_hybrid_score:.12f}"
)

print(
    f"Score /100     : "
    f"{asl_hybrid_score * 100:.4f}"
)

print(
    f"Threshold      : "
    f"{best_threshold:.12f}"
)

print(
    f"Decision       : "
    f"{'SPAM' if asl_is_spam else 'HAM'}"
)


# ============================================================
# 22. STOCKAGGIO DEI RISUTATI DEL TEST
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
    "data/test_results_hybrid_v3.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 23. EXPORT V3
# ============================================================

print("\n" + "=" * 80)
print("V3 - EXPORT DES MODELES")
print("=" * 80)



models_dir = "models_v3"

os.makedirs(
    models_dir,
    exist_ok=True
)


# ============================================================
# 23.1 EXPORT DEL VOCABOLARIO PAROLE
# ============================================================

with open(
    os.path.join(
        models_dir,
        "tfidf_word_vocabulary.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_word_vectorizer.vocabulary_,
        file,
        ensure_ascii=False
    )


# ============================================================
# 23.2 EXPORT DELLE PAROLE IDF
# ============================================================

with open(
    os.path.join(
        models_dir,
        "tfidf_word_idf.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_word_vectorizer.idf_.tolist(),
        file
    )


# ============================================================
# 23.3 EXPORT DEL VOCABOLARIO CARACTERS
# ============================================================

with open(
    os.path.join(
        models_dir,
        "tfidf_char_vocabulary.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_char_vectorizer.vocabulary_,
        file,
        ensure_ascii=False
    )


# ============================================================
# 23.4 EXPORT DEI IDF CARACTERS
# ============================================================

with open(
    os.path.join(
        models_dir,
        "tfidf_char_idf.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_char_vectorizer.idf_.tolist(),
        file
    )


# ============================================================
# 23.5 EXPORT DEL MODELLO ML
# ============================================================

ml_model_data = {
    "coefficients": final_ml_model.coef_[0].tolist(),
    "intercept": float(
        final_ml_model.intercept_[0]
    ),
    "classes": final_ml_model.classes_.tolist()
}


with open(
    os.path.join(
        models_dir,
        "ml_model.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        ml_model_data,
        file,
        ensure_ascii=False
    )


# ============================================================
# 23.6 EXPORT Del RULE MODEL
# ============================================================

rule_feature_names = (
    development_rules.columns.tolist()
)


rule_model_data = {
    "features": rule_feature_names,
    "coefficients": final_rule_model.coef_[0].tolist(),
    "intercept": float(
        final_rule_model.intercept_[0]
    ),
    "classes": final_rule_model.classes_.tolist()
}


with open(
    os.path.join(
        models_dir,
        "rule_model.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        rule_model_data,
        file,
        ensure_ascii=False
    )


# ============================================================
# 23.7 EXPORT DELLA CONFIGURAZIONE
# ============================================================

config_data = {

    "alpha": float(
        best_alpha
    ),

    "rule_weight": float(
        best_rule_weight
    ),

    "threshold": float(
        best_threshold
    ),

    "tfidf_word": {
        "lowercase": True,
        "analyzer": "word",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95
    },

    "tfidf_char": {
        "lowercase": True,
        "analyzer": "char",
        "ngram_range": [3, 5],
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True
    }
}


with open(
    os.path.join(
        models_dir,
        "model_config.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 23.8 CONFIRMAZIONE
# ============================================================

print("\nFile V3 esportato in  :")

print(
    f"{models_dir}/"
)

print("\n" + "=" * 80)
print("V3 - PIPELINE TERMINATO")
print("=" * 80)