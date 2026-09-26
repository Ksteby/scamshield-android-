package com.kemo.scamshield.userInterface.scanner

import android.app.Application
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.kemo.scamshield.data.repository.AnalysisRepository
import com.kemo.scamshield.domain.AnalysisResult
import kotlinx.coroutines.launch

/**
 * ViewModel della schermata Scanner.
 *
 * Gestisce il messaggio, la modalità di analisi
 * e il risultato restituito dal Repository.
 */
class ScannerViewModel(
    application: Application
) : AndroidViewModel(application) {

    /**
     * Repository responsabile dell'analisi.
     */
    private val repository =
        AnalysisRepository(application)


    /**
     * Messaggio inserito dall'utente.
     */
    var message by mutableStateOf("")
        private set


    /**
     * Risultato dell'ultima analisi.
     */
    var analysisResult by mutableStateOf<AnalysisResult?>(null)
        private set


    /**
     * false = modalità offline
     * true = modalità online
     */
    var isOnlineMode by mutableStateOf(false)
        private set


    /**
     * Indica se un'analisi online è in corso.
     */
    var isAnalyzing by mutableStateOf(false)
        private set


    /**
     * Aggiorna il messaggio inserito dall'utente.
     */
    fun onMessageChange(newMessage: String) {
        message = newMessage
    }


    /**
     * Cambia la modalità di analisi.
     */
    fun changeOnlineMode(online: Boolean) {
        isOnlineMode = online
    }


    /**
     * Avvia l'analisi del messaggio.
     *
     * @param onAnalysisComplete funzione chiamata quando
     * l'analisi è terminata.
     */
    fun analyzeMessage(
        onAnalysisComplete: () -> Unit
    ): Boolean {

        // Evitiamo di analizzare un messaggio vuoto.
        if (message.isBlank()) {
            return false
        }


        // ----------------------------------------------------
        // MODALITÀ OFFLINE
        // ----------------------------------------------------

        if (!isOnlineMode) {

            analysisResult =
                repository.analyzeOffline(message)

            // Il risultato offline è disponibile immediatamente.
            onAnalysisComplete()

            return true
        }


        // ----------------------------------------------------
        // MODALITÀ ONLINE
        // ----------------------------------------------------


        viewModelScope.launch {

            isAnalyzing = true

            try {
                // Inviamo il messaggio al server FastAPI.
                analysisResult =
                    repository.analyzeOnline(message)

                // Passiamo al risultato dopo la risposta del server.
                onAnalysisComplete()
            } catch (e: Exception) {

                Log.e(
                    "ScamShieldAPI",
                    "Errore durante l'analisi online",
                    e
                )

            } finally {

                isAnalyzing = false
            }
        }

        return true
    }
}