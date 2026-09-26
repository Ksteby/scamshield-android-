package com.kemo.scamshield.data.repository

import android.content.Context
import android.util.Log
import com.kemo.scamshield.data.AnalyzeRequest
import com.kemo.scamshield.data.RetrofitClient
import com.kemo.scamshield.domain.AnalysisResult
import com.kemo.scamshield.domain.HybridAnalyzer
import com.kemo.scamshield.domain.MessageCategory
import com.kemo.scamshield.domain.ModelLoader
import com.kemo.scamshield.domain.RiskLevel

/**
 * Repository responsabile dell'analisi dei messaggi.
 *
 * Il Repository gestisce due modalità:
 *
 * - analisi offline con il modello V3 presente nell'app;
 * - analisi online tramite Retrofit e server FastAPI.
 */
class AnalysisRepository(
    context: Context
) {

    /**
     * Carica i file del modello V3 da assets/model/.
     */
    private val modelLoader =
        ModelLoader(context)

    /**
     * Motore di analisi locale utilizzato nella modalità offline.
     */
    private val hybridAnalyzer =
        HybridAnalyzer(modelLoader)

    /**
     * Servizio Retrofit utilizzato per comunicare con FastAPI.
     */
    private val apiService =
        RetrofitClient.apiService

    /**
     * Analizza il messaggio utilizzando il modello locale V3.
     *
     * Questa funzione mantiene il funzionamento offline
     * già presente nell'applicazione.
     */
    fun analyzeOffline(message: String): AnalysisResult {

        // Lanciamo l'analisi con il modello V3 presente sul dispositivo.
        val hybridResult =
            hybridAnalyzer.analyze(message)

        Log.d(
            "ScamShieldModel",
            """
            Modalità      : OFFLINE
            Message       : $message
            ML probability: ${hybridResult.mlProbability}
            Rule probability: ${hybridResult.ruleProbability}
            Hybrid score  : ${hybridResult.hybridScore}
            Is spam       : ${hybridResult.isSpam}
            """.trimIndent()
        )

        return createAnalysisResult(
            message = message,
            scoreValue = hybridResult.hybridScore,
            isSpam = hybridResult.isSpam,
            signals = hybridResult.detectedSignals
        )
    }


    /**
     * Analizza il messaggio utilizzando il server FastAPI.
     *
     * Retrofit invia il messaggio al server e riceve
     * il risultato dell'analisi in formato JSON.
     */
    suspend fun analyzeOnline(message: String): AnalysisResult {

        // Creiamo la richiesta contenente il messaggio da analizzare.
        val request =
            AnalyzeRequest(message)

        // Inviamo la requête au serveur FastAPI.
        val response =
            apiService.analyzeMessage(request)

        Log.d(
            "ScamShieldModel",
            """
            Modalità      : ONLINE
            Message       : $message
            ML probability: ${response.ml_probability}
            Rule probability: ${response.rule_probability}
            Hybrid score  : ${response.score}
            Is spam       : ${response.is_spam}
            """.trimIndent()
        )

        /*
         * Il server restituisce già il punteggio finale su 0-100.
         * Lo utilizziamo direttamente per l'interfaccia.
         */


        return createAnalysisResult(
            message = message,
            scoreValue = response.score / 100.0,
            isSpam = response.is_spam,
            signals = emptyList()
        )
    }


    /**
     * Trasforma il risultato del modello in un AnalysisResult
     * utilizzato dall'interfaccia dell'applicazione.
     */
    private fun createAnalysisResult(
        message: String,
        scoreValue: Double,
        isSpam: Boolean,
        signals: List<String>
    ): AnalysisResult {

        // Convertiamo il punteggio da 0-1 a un valore compreso tra 0 e 100.
        val score =
            (scoreValue * 100)
                .toInt()
                .coerceIn(0, 100)

        // Determiniamo il livello di rischio in base al punteggio.
        val riskLevel =
            when {

                score >= 80 ->
                    RiskLevel.CRITICAL

                score >= 60 ->
                    RiskLevel.HIGH

                score >= 30 ->
                    RiskLevel.MODERATE

                else ->
                    RiskLevel.LOW
            }

        // Creiamo l'explication affichée à l'utilisateur.
        val explanation =
            createExplanation(
                isSpam = isSpam,
                signals = signals
            )

        return AnalysisResult(
            message = message,
            score = score,
            riskLevel = riskLevel,
            category = MessageCategory.OTHER_SCAM,
            detectedSignals = signals,
            explanation = explanation
        )
    }


    /**
     * Genera una spiegazione leggibile dall'utente.
     *
     * Questa funzione serve solamente alla presentazione
     * del risultato e non modifica la classificazione.
     */
    private fun createExplanation(
        isSpam: Boolean,
        signals: List<String>
    ): String {

        // Nessun segnale particolare è stato rilevato.
        if (signals.isEmpty()) {

            return if (isSpam) {
                "Il messaggio presenta caratteristiche compatibili con un possibile messaggio fraudolento."
            } else {
                "Non sono stati rilevati elementi particolarmente sospetti."
            }
        }

        // Sono stati rilevati diversi segnali.
        if (signals.size >= 2) {

            return if (isSpam) {
                "Sono stati rilevati diversi elementi frequentemente associati a messaggi fraudolenti."
            } else {
                "Sono stati rilevati alcuni elementi potenzialmente sospetti, ma il messaggio non è stato classificato come fraudolento."
            }
        }

        // È stato rilevato un solo segnale.
        return if (isSpam) {
            "È stato rilevato un elemento potenzialmente associato a un messaggio fraudolento."
        } else {
            "È stato rilevato un elemento potenzialmente sospetto."
        }
    }
}