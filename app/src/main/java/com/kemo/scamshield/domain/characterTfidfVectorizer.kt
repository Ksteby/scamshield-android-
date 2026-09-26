package com.kemo.scamshield.domain

import kotlin.math.ln
import kotlin.math.sqrt

/**
 *Vettorializzatore TF-IDF basato su n-grammi di caratteri.
 *
 * Riproduce il funzionamento del vettorializzatore Python V3:
 * n-grammi da 3 a 5 caratteri e TF sublineare.
 */
class CharacterTfidfVectorizer(
    private val vocabulary: Map<String, Int>,
    private val idf: List<Double>
) {

    fun transform(text: String): DoubleArray {

        // Conversionne del testo in mnuscolo come durante l'addestramento.
        val normalizedText = text.lowercase()

        // Creazione del vettore TF con la lunghezza del vocabolario.
        val tf = DoubleArray(idf.size)

        // Creazione dei n-grammi di caratteri da 3 a 5 caracteri.
        for (size in 3..5) {

            if (normalizedText.length < size) {
                continue
            }

            for (i in 0..normalizedText.length - size) {

                val ngram = normalizedText.substring(
                    i,
                    i + size
                )

                val index = vocabulary[ngram]

                if (index != null) {
                    tf[index] += 1.0
                }
            }
        }

        // Applicazione del TF sublineare utilizzato da Python.
        for (i in tf.indices) {
            if (tf[i] > 0.0) {
                tf[i] = 1.0 + ln(tf[i])
            }
        }

        // Applicazione dell'IDF.
        for (i in tf.indices) {
            tf[i] *= idf[i]
        }

        // Calcolo della norma L2.
        var sum = 0.0

        for (value in tf) {
            sum += value * value
        }

        val norm = sqrt(sum)

        // Normalizzazione del vettore.
        if (norm > 0.0) {
            for (i in tf.indices) {
                tf[i] /= norm
            }
        }

        return tf
    }
}