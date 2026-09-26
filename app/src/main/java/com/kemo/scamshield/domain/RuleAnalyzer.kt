package com.kemo.scamshield.domain

class RuleAnalyzer {

    fun analyze(message: String): AnalysisResult {

        val signals = mutableListOf<String>()
        var score = 0

        // Individua un linguaggio che mira a creare un senso di urgenza.
        if (
            message.contains("urgente", ignoreCase = true) ||
            message.contains("urgent", ignoreCase = true) ||
            message.contains("subito", ignoreCase = true)
        ) {
            signals.add("Vocabolario urgente")
            score += 20
        }

        // Rileva un invito a cliccare su un link.
        if (
            message.contains("clicca", ignoreCase = true) ||
            message.contains("cliccare", ignoreCase = true) ||
            message.contains("link", ignoreCase = true)
        ) {
            signals.add("Invito a cliccare su un link")
            score += 20
        }


        if (message.contains("http://", ignoreCase = true)) {
            signals.add("Link HTTP non sicuro")
            score += 25
        }

        // Rileva potenziali richieste di informazioni sensibili.
        if (
            message.contains("password", ignoreCase = true) ||
            message.contains("codice", ignoreCase = true) ||
            message.contains("credenziali", ignoreCase = true)
        ) {
            signals.add("Richiesta di dati personali")
            score += 25
        }

        // Il punteggio non deve superare 100.
        score = score.coerceAtMost(100)

        // Transforma il punteggio in livello di rischio
        val riskLevel = when {
            score >= 80 -> RiskLevel.CRITICAL
            score >= 60 -> RiskLevel.HIGH
            score >= 30 -> RiskLevel.MODERATE
            else -> RiskLevel.LOW
        }

        return AnalysisResult(
            message = message,
            score = score,
            riskLevel = riskLevel,
            category = MessageCategory.OTHER_SCAM,
            detectedSignals = signals,
            explanation = createExplanation(signals)
        )
    }

    // Genera una spiegazione sulla base dei segnali rilevati.
    private fun createExplanation(signals: List<String>): String {

        return when {
            signals.isEmpty() ->
                "Non sono stati rilevati elementi particolarmente sospetti."

            signals.size == 1 ->
                "È stato rilevato un elemento potenzialmente sospetto."

            else ->
                "Sono stati rilevati diversi elementi frequentemente associati a messaggi fraudolenti."
        }
    }
}