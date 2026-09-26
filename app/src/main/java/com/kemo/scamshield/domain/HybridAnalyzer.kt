package com.kemo.scamshield.domain

/**
 * Motore principale di analisi di ScamShield.
 *
 * La versione V3 combina:
 *
 * 1. Un modello di apprendimento automatico che utilizza:
 * - TF-IDF basato sulle parole
 * - TF-IDF basato sui caratteri
 * - Regressione logistica
 *
 * 2. Un modello basato su regole + regressione logistica.
 *
 * I due risultati vengono poi combinati per ottenere
 * il punteggio finale del sistema ibrido.
 */
class HybridAnalyzer(
    private val modelLoader: ModelLoader
) {

    // --------------------------------------------------------
    // Caricamento dei parametri del modello
    // --------------------------------------------------------

    private val wordVocabulary =
        modelLoader.loadWordVocabulary()

    private val wordIdf =
        modelLoader.loadWordIdf()

    private val characterVocabulary =
        modelLoader.loadCharacterVocabulary()

    private val characterIdf =
        modelLoader.loadCharacterIdf()

    private val mlModelData =
        modelLoader.loadMlModel()

    private val ruleModelData =
        modelLoader.loadRuleModel()

    private val config =
        modelLoader.loadConfig()


    // --------------------------------------------------------
    // Creazione dei componenti di analisi
    // --------------------------------------------------------

    /**
     * Vectorializzatore TF-IDF basato dulle parole.
     */
    private val wordVectorizer =
        TfidfVectorizer(
            vocabulary = wordVocabulary,
            idf = wordIdf
        )

    /*
     * Vettorializzatore TF-IDF basato su n-grammi di caratteri.
     */
    private val characterVectorizer =
        CharacterTfidfVectorizer(
            vocabulary = characterVocabulary,
            idf = characterIdf
        )

    /*
     * Modello Logistic Regression principale.
     */
    private val mlModel =
        LogisticRegressionModel(
            coefficients = mlModelData.coefficients,
            intercept = mlModelData.intercept
        )

    /**
     * Estrazione dei features du Rule Analyzer.
     */
    private val ruleFeatureExtractor =
        RuleFeatureExtractor()

    /**
     * Modello Logistic Regression del Rule Analyzer.
     */
    private val ruleModel =
        RuleModel(ruleModelData)


    /**
     * @param message testo da analizzare
     *
     * @return risultato completo dell'analisi.
     */
    fun analyze(message: String): HybridAnalysis {

        // ----------------------------------------------------
        // 1. Creazione del vettore Word TF-IDF
        // ----------------------------------------------------

        val wordVector =
            wordVectorizer.transform(message)


        // ----------------------------------------------------
        // 2. Creazione del vettore Character TF-IDF
        // ----------------------------------------------------

        val characterVector =
            characterVectorizer.transform(message)


        // ----------------------------------------------------
        // 3. Combinazione dei due vettori
        // ----------------------------------------------------

        /*
         * Il modello Python V3 è stato addestrato con:

         * [Word TF-IDF][Character TF-IDF]

         * I due vettori devono quindi essere inseriti in questo ordine.
         */
        val combinedVector =
            DoubleArray(
                wordVector.size + characterVector.size
            )

        for (i in wordVector.indices) {
            combinedVector[i] = wordVector[i]
        }

        for (i in characterVector.indices) {
            combinedVector[wordVector.size + i] =
                characterVector[i]
        }


        // ----------------------------------------------------
        // 4. Analyse ML
        // ----------------------------------------------------

        val mlProbability =
            mlModel.predictProbability(combinedVector)


        // ----------------------------------------------------
        // 5. Analyse Rule
        // ----------------------------------------------------

        val ruleFeatures =
            ruleFeatureExtractor.extractFeatures(message)

        val ruleProbability =
            ruleModel.predictProbability(ruleFeatures)


        // ----------------------------------------------------
        // 6. Combinazione dei due modelli
        // ----------------------------------------------------

        /*
         * Carichiamo i pesi e le soglie dalla configurazione del modello
         */
        val hybridScore =
            config.alpha * mlProbability +
                    config.ruleWeight * ruleProbability


        // ----------------------------------------------------
        // 7. Décisione finale
        // ----------------------------------------------------

        val isSpam =
            hybridScore >= config.threshold


        // ----------------------------------------------------
        // 8. Rilevamento dei segnali leggibli
        // ----------------------------------------------------

        val detectedSignals =
            detectSignals(message)


        return HybridAnalysis(
            message = message,
            mlProbability = mlProbability,
            ruleProbability = ruleProbability,
            hybridScore = hybridScore,
            isSpam = isSpam,
            detectedSignals = detectedSignals
        )
    }


    /*Rileva i segnali leggibili dall'utente.

     Questa sezione serve a spiegare alcuni
     elementi sospetti rilevati nel messaggio.
      */
    private fun detectSignals(message: String): List<String> {

        val signals = mutableListOf<String>()

        if (
            Regex(
                """\burgente\b|\burgent\b|\bsubito\b|\bimmediatamente\b|\bscadenza\b|\boggi\b|\bnow\b|\bimmediately\b|\btoday\b|\bentro\b""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(message)
        ) {
            signals.add("Vocabolario urgente")
        }

        if (
            Regex(
                """\bclicca\b|\bclick\b|\brispondi\b|\breply\b|\bchiama\b|\bcall\b|\bconferma\b|\bconfirm\b|\bverifica\b|\bverify\b|\bscarica\b|\bdownload\b|\bapri\b|\bopen\b""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(message)
        ) {
            signals.add("Invito a compiere un'azione")
        }

        if (
            Regex(
                """€|\$|£|\beuro\b|\beuros\b""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(message)
        ) {
            signals.add("Presenza di denaro")
        }

        if (
            Regex(
                """\bbanca\b|\bbank\b|\bconto\b|\baccount\b|\bpassword\b|\bcodice\b|\bcode\b|\bpin\b|\bcredenziali\b|\bcredentials\b|\bcarta\b|\bcard\b|\bpagamento\b|\bpayment\b""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(message)
        ) {
            signals.add("Richiesta o riferimento a dati sensibili")
        }

        if (
            Regex(
                """https?://|www\.""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(message)
        ) {
            signals.add("Presenza di un URL")
        }

        return signals
    }
}


/**
  Risultato completo prodotto dal motore ibrido
 */
data class HybridAnalysis(
    val message: String,
    val mlProbability: Double,
    val ruleProbability: Double,
    val hybridScore: Double,
    val isSpam: Boolean,
    val detectedSignals: List<String>
)