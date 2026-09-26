package com.kemo.scamshield.domain

import kotlin.math.exp

/**
 * Riproduce l'inferenza di una Logistic Regression
 * addestrata con Scikit-learn.
 *
 * Il modello Python calcola:
 *
 *     z = intercept + Σ(coefficiente × feature)
 *
 * e applica successivamente la funzione sigmoide:
 *
 *     probability = 1 / (1 + exp(-z))
 *
 * Il valore restituito corrisponde alla probabilità
 * della classe positiva: SPAM.
 */
class LogisticRegressionModel(
    private val coefficients: List<Double>,
    private val intercept: Double
) {

    /**
     * Calcola la probabilità che il vettore di input
     * appartenga alla classe SPAM.
     */
    fun predictProbability(features: DoubleArray): Double {

        // Il numero di coefficienti deve corrispondere
        // al numero di feature del vettore di input.
        require(features.size == coefficients.size) {
            "Numero di feature non corretto: " +
                    "${features.size} ricevute, " +
                    "${coefficients.size} attese."
        }

        // Inizia con l'intercetta del modello.
        var z = intercept

        // Calcola:
        //
        // intercept + coefficient[0] * feature[0]
        //           + coefficient[1] * feature[1]
        //           + ...
        for (i in features.indices) {
            z += coefficients[i] * features[i]
        }

        // Funzione sigmoide.
        //
        // Trasforma il valore z in un valore
        // compreso tra 0 e 1.
        return sigmoid(z)
    }

    /**
     * Funzione sigmoide:
     *
     *              1
     * σ(z) = -------------
     *         1 + e^(-z)
     */
    private fun sigmoid(z: Double): Double {

        return 1.0 / (1.0 + exp(-z))
    }
}