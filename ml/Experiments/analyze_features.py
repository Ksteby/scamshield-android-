import re
import pandas as pd


# ============================================================
# 1. CARICAMENTO DEI RISULTATI DEL MODELLO
# ============================================================

# Il file è stato creato con evaluate_errors.py quindi contiene già le predizioni del modello
results_path = "data/test_results.csv"

results = pd.read_csv(
    results_path
)


# ============================================================
# 2. IDENTIFICAZIONE DEI FALSE NEGATIVES
# ============================================================

# FN= spam classificatocomme ham
false_negatives = results[
    (results["label"] == "spam") &
    (results["prediction"] == "ham")
].copy()


# ============================================================
# 3. FUNZIONE DI ANALISI DI UN MESSAGGIO
# ============================================================

def analyze_message(message):

    # Facciamo la conversione per facilitare la ricerca delle parole chiave

    message_lower = message.lower()

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    has_url = bool(
        re.search(
            r"https?://|www\.",
            message,
            re.IGNORECASE
        )
    )

    # --------------------------------------------------------
    # Numero di telefono
    # --------------------------------------------------------

    has_phone = bool(
        re.search(
            r"(?:\+\d{8,15}|\b\d{8,15}\b)",
            message
        )
    )

    # --------------------------------------------------------
    # Indirizzo mail
    # --------------------------------------------------------

    has_email = bool(
        re.search(
            r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
            message
        )
    )

    # --------------------------------------------------------
    # Short URL
    # --------------------------------------------------------

    has_short_url = bool(
        re.search(
            r"bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|is\.gd",
            message_lower
        )
    )

    # --------------------------------------------------------
    # Soldi
    # --------------------------------------------------------

    has_money = bool(
        re.search(
            r"€|\$|£|\beuro\b|\beuros\b",
            message_lower
        )
    )

    # --------------------------------------------------------
    # Numero di numeri presenti nel messaggio
    # --------------------------------------------------------

    number_count = len(
        re.findall(
            r"\d+",
            message
        )
    )

    # --------------------------------------------------------
    # Signali d'emergenza
    # --------------------------------------------------------

    urgency_words = [
        "urgente",
        "urgent",
        "subito",
        "immediatamente",
        "scadenza",
        "oggi",
        "now",
        "immediately",
        "today",
        "entro"
    ]

    has_urgency = any(
        word in message_lower
        for word in urgency_words
    )

    # --------------------------------------------------------
    # Signali d'azione
    # --------------------------------------------------------

    action_words = [
        "clicca",
        "click",
        "rispondi",
        "reply",
        "chiama",
        "call",
        "conferma",
        "confirm",
        "verifica",
        "verify",
        "scarica",
        "download",
        "apri",
        "open"
    ]

    has_action = any(
        word in message_lower
        for word in action_words
    )

    # --------------------------------------------------------
    # Dati sensibili /finanziari
    # --------------------------------------------------------

    sensitive_words = [
        "banca",
        "bank",
        "conto",
        "account",
        "password",
        "codice",
        "code",
        "pin",
        "credenziali",
        "credentials",
        "carta",
        "card",
        "pagamento",
        "payment"
    ]

    has_sensitive_term = any(
        word in message_lower
        for word in sensitive_words
    )

    # --------------------------------------------------------
    # Percentuale di maiuscole
    # --------------------------------------------------------

    uppercase_count = sum(
        1
        for character in message
        if character.isupper()
    )

    letters = [
        character
        for character in message
        if character.isalpha()
    ]

    uppercase_ratio = (
        uppercase_count / len(letters)
        if letters
        else 0
    )

    return {
        "url": has_url,
        "phone": has_phone,
        "email": has_email,
        "short_url": has_short_url,
        "money": has_money,
        "numbers": number_count,
        "urgency": has_urgency,
        "action": has_action,
        "sensitive": has_sensitive_term,
        "uppercase_ratio": uppercase_ratio,
        "length": len(message)
    }


# ============================================================
# 4. ANALISI DEI FN
# ============================================================

features = false_negatives["message"].apply(
    analyze_message
)

features_df = pd.DataFrame(
    features.tolist()
)


# ============================================================
# 5. STATISTICHE GLOBALI
# ============================================================

print("\n")
print("==============================================")
print("CARACTERISTICI DEI FN")
print("==============================================")

print(
    "\nNumero di FN:",
    len(false_negatives)
)


boolean_features = [
    "url",
    "phone",
    "email",
    "short_url",
    "money",
    "urgency",
    "action",
    "sensitive"
]


print("\nPresenza dei caratteristici :")

for feature in boolean_features:

    count = features_df[feature].sum()

    percentage = (
        count / len(features_df) * 100
    )

    print(
        f"{feature:15} : "
        f"{count:2} "
        f"({percentage:.1f} %)"
    )


print("\nMedia dei numeri :")

print(
    features_df["numbers"].mean()
)


print("\nLunghezza media dei msg :")

print(
    features_df["length"].mean()
)


print("\nRatio medio dei maiuscole :")

print(
    features_df["uppercase_ratio"].mean()
)


# ============================================================
# 6. ANALISI DEI FN ITALIANI
# ============================================================

italian_fn = false_negatives[
    false_negatives["language"] == "italian"
].copy()


italian_features = italian_fn["message"].apply(
    analyze_message
)

italian_features_df = pd.DataFrame(
    italian_features.tolist()
)


print("\n")
print("==============================================")
print("FN ITALIANI")
print("==============================================")

print(
    "\nNumero :",
    len(italian_fn)
)


for feature in boolean_features:

    count = italian_features_df[feature].sum()

    percentage = (
        count / len(italian_features_df) * 100
    )

    print(
        f"{feature:15} : "
        f"{count:2} "
        f"({percentage:.1f} %)"
    )


# ============================================================
# 7. FINE
# ============================================================

print("\n")
print("==============================================")
print("FINE DEL ANALISI")
print("==============================================")