package com.kemo.scamshield.domain

data class AnalysisResult(

    // Messaggio originale inviato all'analizzatore
    val message: String,

    // Punteggio di rischio compreso tra 0 e 100
    val score: Int,

    // Livello di rischio corrispondente al punteggio
    val riskLevel: RiskLevel,

    // Tipo di messaggio rilevato
    val category: MessageCategory,

    // Elenco degli elementi sospeti rilevati
    val detectedSignals: List<String>,

    // Spiegezioni generali destinati all'utente
    val explanation: String
)

