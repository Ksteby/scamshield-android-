package com.kemo.scamshield.data

import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {

    // Invia il messaggio al server FastAPI per l'analisi online.
    @POST("analyze")
    suspend fun analyzeMessage(
        @Body request: AnalyzeRequest
    ): AnalyzeResponse
}