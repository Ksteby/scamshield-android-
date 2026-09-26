package com.kemo.scamshield.data

data class AnalyzeResponse(
    val ml_probability: Double,
    val rule_probability: Double,
    val score: Double,
    val is_spam: Boolean,
    val threshold: Double
)