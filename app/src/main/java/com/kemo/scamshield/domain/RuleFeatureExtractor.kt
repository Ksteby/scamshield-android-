package com.kemo.scamshield.domain

/**
 * Estrae le 14 feature utilizzate dal Rule Model Python V3.
 *
 * Le prime 7 feature corrispondono ai segnali
 * rilevati direttamente nel messaggio.
 *
 * Le 7 feature successive corrispondono alle interazioni
 * tra più segnali.
 */
class RuleFeatureExtractor {

    /**
     * Nomi delle feature nello stesso identico ordine
     * utilizzato durante l'addestramento Python.
     *
     * Questo ordine è molto importante perché ogni coefficiente
     * del modello corrisponde a una posizione precisa.
     */
    val featureNames = listOf(
        "urgency",
        "action",
        "money",
        "sensitive",
        "url",
        "contact",
        "phone",
        "urgency_sensitive",
        "action_money",
        "action_sensitive",
        "urgency_money",
        "phone_contact",
        "phone_urgency",
        "contact_urgency"
    )

    /**
     * Estrae le 14 feature numeriche dal messaggio.
     *
     * Ogni feature vale:
     *
     * 0 = segnale assente
     * 1 = segnale presente
     */
    fun extractFeatures(message: String): DoubleArray {

        // ----------------------------------------------------
        // 1. Rilevamento dei sette segnali principali
        // ----------------------------------------------------

        val urgency = hasUrgency(message)

        val action = hasAction(message)

        val money = hasMoney(message)

        val sensitive = hasSensitive(message)

        val url = hasUrl(message)

        val contact = hasContact(message)

        val phone = hasPhone(message)


        // ----------------------------------------------------
        // 2. Creazione delle interazioni
        // ----------------------------------------------------

        /*
         * Un'interazione vale 1 quando entrambi
         * i segnali corrispondenti sono presenti.
         */

        val urgencySensitive =
            urgency && sensitive

        val actionMoney =
            action && money

        val actionSensitive =
            action && sensitive

        val urgencyMoney =
            urgency && money

        val phoneContact =
            phone && contact

        val phoneUrgency =
            phone && urgency

        val contactUrgency =
            contact && urgency


        // ----------------------------------------------------
        // 3. Conversione Boolean → Double
        // ----------------------------------------------------

        return doubleArrayOf(
            urgency.toDouble(),
            action.toDouble(),
            money.toDouble(),
            sensitive.toDouble(),
            url.toDouble(),
            contact.toDouble(),
            phone.toDouble(),
            urgencySensitive.toDouble(),
            actionMoney.toDouble(),
            actionSensitive.toDouble(),
            urgencyMoney.toDouble(),
            phoneContact.toDouble(),
            phoneUrgency.toDouble(),
            contactUrgency.toDouble()
        )
    }


    /**
     * Rileva il vocabolario associato all'urgenza.
     */
    private fun hasUrgency(message: String): Boolean {

        val pattern = Regex(
            """\burgente\b|\burgenti\b|\burgent\b|\bsubito\b|\bimmediatamente\b|\bscadenza\b|\boggi\b|\bora\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva le richieste di azione.
     */
    private fun hasAction(message: String): Boolean {

        val pattern = Regex(
            """\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|\bscarica\b|\bdownload\b|\bapri\b|\bopen\b""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva la presenza di un'informazione monetaria.
     */
    private fun hasMoney(message: String): Boolean {

        val pattern = Regex(
            """€|\$|£|\beuro\b|\beuros\b""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva informazioni sensibili o bancarie.
     */
    private fun hasSensitive(message: String): Boolean {

        val pattern = Regex(
            """\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva la presenza di un URL.
     */
    private fun hasUrl(message: String): Boolean {

        val pattern = Regex(
            """https?://|www\.""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva le parole associate a un metodo di contatto.
     */
    private fun hasContact(message: String): Boolean {

        val pattern = Regex(
            """\bcontatta\b|\bcontattare\b|\bcontatto\b|\bcontatti\b|\bchiamaci\b|\bcall\b|\bcontact\b|\bcontatta\s+la\s+propria\s+sede\b""",
            RegexOption.IGNORE_CASE
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Rileva la presenza di un numero di telefono.
     */
    private fun hasPhone(message: String): Boolean {

        val pattern = Regex(
            """(?<!\d)\+?\d[\d\s().-]{6,}\d(?!\d)"""
        )

        return pattern.containsMatchIn(message)
    }


    /**
     * Converte un Boolean in un valore numerico.
     *
     * false → 0.0
     * true  → 1.0
     */
    private fun Boolean.toDouble(): Double {
        return if (this) 1.0 else 0.0
    }
}