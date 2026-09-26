package com.kemo.scamshield.domain

/**
 * Modello di classificazione basato su regole.
 *
 * Questa classe funge da collegamento tra:
 *
 *   1. RuleFeatureExtractor
 *         ↓
 *  9 caratteristiche
 *            ↓
 * * 2. LogisticRegressionModel
 *            ↓
 *  probabilità di SPAM
 *
 *  I coefficienti utilizzati da questa classe sono quelli
 *  del modello addestrato con Scikit-learn e salvato
 * nel file rule_model.json.
 */
class RuleModel(
    private val modelData: RuleModelData
) {

    // Modello di regressione logistica utilizzato per calcolare
    // la probabilità che il messaggio sia spam.
    private val classifier = LogisticRegressionModel(
        coefficients = modelData.coefficients,
        intercept = modelData.intercept
    )

    /**
     * Calcula la probabilità che un messaggio sia un SPAM
     *
     * @param features i 9 features estrati dal messaggio.
     *
     * @return valore compreso tra 0 e 1.
     */
    fun predictProbability(features: DoubleArray): Double {

        return classifier.predictProbability(features)
    }
}