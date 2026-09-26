package com.kemo.scamshield

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.mutableStateOf
import com.kemo.scamshield.navigation.AppNavigation
import com.kemo.scamshield.ui.theme.ScamShieldTheme

class MainActivity : ComponentActivity() {

    // Contiene il testo ricevuto tramite la funzione "Condividi" di Android.
    private val sharedText = mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Controlla se l'app è stata aperta tramite la condivisione di testo.
        sharedText.value = extractSharedText(intent)

        setContent {
            ScamShieldTheme {

                AppNavigation(
                    sharedText = sharedText.value,

                    // Segnala che il testo condiviso è stato elaborato.
                    onSharedTextConsumed = {
                        sharedText.value = null
                    }
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)

        // Aggiorna l'Intent corrente con il nuovo contenuto ricevuto.
        setIntent(intent)

        // Recupera il nuovo testo condiviso.
        sharedText.value = extractSharedText(intent)
    }

    /**
     * Recupera il testo contenuto nell'Intent di tipo SEND.
     *
     * @return il testo condiviso oppure null se non è presente.
     */
    private fun extractSharedText(intent: Intent?): String? {

        // Verifica che l'Intent provenga dalla funzione "Condividi".
        if (intent?.action != Intent.ACTION_SEND) {
            return null
        }

        // Recupera il testo condiviso.
        return intent.getStringExtra(Intent.EXTRA_TEXT)
    }
}