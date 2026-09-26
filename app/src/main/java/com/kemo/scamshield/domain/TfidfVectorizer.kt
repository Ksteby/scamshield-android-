package com.kemo.scamshield.domain

import kotlin.math.sqrt

/**
 * Riproduce il funzionamento del TfidfVectorizer
 * utilizzato nel nostro modello Python.
 *
 * Configurazione Python:
 *
 * lowercase = True
 * ngram_range = (1, 2)
 * min_df = 2
 * max_df = 0.95
 * norm = "l2" (valore predefinito di Scikit-learn)
 *
 * Il vocabolario e i valori IDF sono quelli del modello
 * finale esportato da Python.
 */
class TfidfVectorizer(
    private val vocabulary: Map<String, Int>,
    private val idf: List<Double>
) {

    /**
     * Trasforma un messaggio in un vettore TF-IDF.
     *
     * Il risultato è un array in cui ogni posizione
     * corrisponde a una caratteristica del vocabolario Python.
     */
    fun transform(message: String): DoubleArray {

        // ----------------------------------------------------
        // 1. Pre-addestramento
        // ----------------------------------------------------

        // Scikit-learn utilise lowercase=True.
        val text = message.lowercase()

        // ----------------------------------------------------
        // 2. Tokenisation
        // ----------------------------------------------------

        /*
         *conserviamo
         * le sequenze contenenti almeno due caratteri
         * alfanumerici.
         */
        val tokens = Regex("""[\p{L}\p{N}_]{2,}""")
            .findAll(text)
            .map { it.value }
            .toList()
        // ----------------------------------------------------
        // 3. Creaazione dei n-grammi
        // ----------------------------------------------------

        /*
         * Il nostro modelle utilizza
         *
         * ngram_range=(1, 2)
         *
         * Nquindi creamo gli unigrammi e gli bigrammi
         */
        val ngrams = mutableListOf<String>()

        // Unigrams
        ngrams.addAll(tokens)

        // Bigrams
        for (i in 0 until tokens.size - 1) {
            ngrams.add(
                "${tokens[i]} ${tokens[i + 1]}"
            )
        }

        // ----------------------------------------------------
        // 4. Contaggio dei termini
        // ----------------------------------------------------

        /*
         * Utilizziamo una tabella le cui dimensioni corrispondono
         * al numero di feature del vocabolario.
         */
        val termFrequency = DoubleArray(idf.size)

        for (ngram in ngrams) {

            val index = vocabulary[ngram]

            if (index != null) {
                termFrequency[index] += 1.0
            }
        }

        // ----------------------------------------------------
        // 5. Applicazione dell'IDF
        // ----------------------------------------------------

        /*
         * TF-IDF = TF × IDF
         *
         * I valori IDF provengono direttamente da Scikit-learn e sono stati esportati da Python.
         */
        for (i in termFrequency.indices) {
            termFrequency[i] *= idf[i]
        }

        // ----------------------------------------------------
        // 6. Normalizzazione L2
        // ----------------------------------------------------

        var squaredSum = 0.0

        for (value in termFrequency) {
            squaredSum += value * value
        }

        val norm = sqrt(squaredSum)


        if (norm > 0.0) {
            for (i in termFrequency.indices) {
                termFrequency[i] /= norm
            }
        }

        return termFrequency
    }
}