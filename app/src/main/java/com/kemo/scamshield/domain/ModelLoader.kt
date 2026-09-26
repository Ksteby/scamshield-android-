package com.kemo.scamshield.domain

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Carica i parametri dei modelli Machine Learning
 * salvati nella cartella assets/model/.
 *
 * Questa classe si occupa solamente di leggere
 * i file JSON necessari al motore di analisi.
 */
class ModelLoader(private val context: Context) {

    /**
     * Carica un file JSON dalla cartella assets/model/.
     */
    private fun readAssetFile(fileName: String): String {
        return context.assets
            .open("model/$fileName")
            .bufferedReader()
            .use { it.readText() }
    }

    /**
     * Carica un vocabolario TF-IDF.
     *
     * Il JSON contiene una relazione:
     *
     * token -> indice
     */
    private fun loadVocabularyFile(
        fileName: String
    ): Map<String, Int> {

        val json = JSONObject(
            readAssetFile(fileName)
        )

        val vocabulary = mutableMapOf<String, Int>()

        val keys = json.keys()

        while (keys.hasNext()) {
            val token = keys.next()
            vocabulary[token] = json.getInt(token)
        }

        return vocabulary
    }

    /**
     * Carica il vocabolario TF-IDF delle parole.
     */
    fun loadWordVocabulary(): Map<String, Int> {
        return loadVocabularyFile(
            "v3_tfidf_word_vocabulary.json"
        )
    }

    /**
     * Carica il vocabolario TF-IDF dei caratteri.
     */
    fun loadCharacterVocabulary(): Map<String, Int> {
        return loadVocabularyFile(
            "v3_tfidf_char_vocabulary.json"
        )
    }

    /**
     * Carica una lista di valori IDF.
     */
    private fun loadIdfFile(
        fileName: String
    ): List<Double> {

        val jsonArray = JSONArray(
            readAssetFile(fileName)
        )

        return List(jsonArray.length()) { index ->
            jsonArray.getDouble(index)
        }
    }

    /**
     * Carica gli IDF utilizzati dal modello sulle parole.
     */
    fun loadWordIdf(): List<Double> {
        return loadIdfFile(
            "v3_tfidf_word_idf.json"
        )
    }

    /**
     * Carica gli IDF utilizzati dal modello sui caratteri.
     */
    fun loadCharacterIdf(): List<Double> {
        return loadIdfFile(
            "v3_tfidf_char_idf.json"
        )
    }

    /**
     * Carica i coefficienti del modello ML V3
     * e il relativo intercept.
     */
    fun loadMlModel(): MlModelData {

        val json = JSONObject(
            readAssetFile("v3_ml_model.json")
        )

        val coefficientsJson =
            json.getJSONArray("coefficients")

        val coefficients = List(
            coefficientsJson.length()
        ) { index ->
            coefficientsJson.getDouble(index)
        }

        return MlModelData(
            coefficients = coefficients,
            intercept = json.getDouble("intercept")
        )
    }

    /**
     * Carica i coefficienti del modello basato
     * sulle regole.
     */
    fun loadRuleModel(): RuleModelData {

        val json = JSONObject(
            readAssetFile("v3_rule_model.json")
        )

        val featuresJson =
            json.getJSONArray("features")

        val features = List(
            featuresJson.length()
        ) { index ->
            featuresJson.getString(index)
        }

        val coefficientsJson =
            json.getJSONArray("coefficients")

        val coefficients = List(
            coefficientsJson.length()
        ) { index ->
            coefficientsJson.getDouble(index)
        }

        return RuleModelData(
            features = features,
            coefficients = coefficients,
            intercept = json.getDouble("intercept")
        )
    }

    /**
     * Carica la configurazione del sistema ibrido V3.
     */
    fun loadConfig(): ModelConfig {

        val json = JSONObject(
            readAssetFile("v3_model_config.json")
        )

        return ModelConfig(
            alpha = json.getDouble("alpha"),
            ruleWeight = json.getDouble("rule_weight"),
            threshold = json.getDouble("threshold")
        )
    }
}


/**
 * Parametri del modello Logistic Regression ML.
 */
data class MlModelData(
    val coefficients: List<Double>,
    val intercept: Double
)


/**
 * Parametri del modello Logistic Regression
 * utilizzato dal Rule Analyzer.
 */
data class RuleModelData(
    val features: List<String>,
    val coefficients: List<Double>,
    val intercept: Double
)


/**
 * Configurazione finale del sistema ibrido.
 */
data class ModelConfig(
    val alpha: Double,
    val ruleWeight: Double,
    val threshold: Double
)