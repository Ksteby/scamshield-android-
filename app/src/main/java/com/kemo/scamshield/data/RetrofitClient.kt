package com.kemo.scamshield.data

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitClient {

    // Indirizzo del server FastAPI quando utilizziamo l'emulatore Android
    private const val BASE_URL = "http://127.0.0.1:8000/"

    // Creiamo una sola istanza di Retrofit per tutta l'applicazione
    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    // Creiamo l'implementazione dell'interfaccia ApiService
    val apiService: ApiService = retrofit.create(ApiService::class.java)
}

