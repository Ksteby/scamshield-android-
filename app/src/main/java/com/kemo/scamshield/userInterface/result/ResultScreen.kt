package com.kemo.scamshield.userInterface.result

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.kemo.scamshield.domain.AnalysisResult
import com.kemo.scamshield.domain.RiskLevel

/**
 * Schermata che mostra il risultato dell'analisi.
 *
 * Il risultato proviene dal sistema ibrido V3.
 */
@Composable
fun ResultScreen(
    result: AnalysisResult,
    onBackToScanner: () -> Unit
) {

    // Permette di scorrere la schermata sui dispositivi più piccoli.
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp),
        verticalArrangement = Arrangement.Top
    ) {

        // ----------------------------------------------------
        // TITLE
        // ----------------------------------------------------

        Text(
            text = "🛡️ Risultato dell'analisi",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )

        Text(
            text = "ScamShield ha completato l'analisi del messaggio.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(
            modifier = Modifier.height(20.dp)
        )


        // ----------------------------------------------------
        // CARTE DEL RISULTATO
        // ----------------------------------------------------

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        ) {

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                Text(
                    text = "Punteggio di rischio",
                    style = MaterialTheme.typography.titleMedium
                )

                Spacer(
                    modifier = Modifier.height(8.dp)
                )

                Text(
                    text = "${result.score.toInt()}/100",
                    style = MaterialTheme.typography.displaySmall,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(
                    modifier = Modifier.height(4.dp)
                )

                Text(
                    text = getRiskLabel(result.riskLevel),
                    style = MaterialTheme.typography.titleLarge
                )

                Spacer(
                    modifier = Modifier.height(14.dp)
                )

                // Il progress indicator utilizza un valore compreso
                // tra 0 e 1.
                LinearProgressIndicator(
                    progress = {
                        (result.score / 100f).coerceIn(0f, 1f)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(10.dp)
                )
            }
        }

        Spacer(
            modifier = Modifier.height(16.dp)
        )


        // ----------------------------------------------------
        // DECISIONE DEL MODELLO
        // ----------------------------------------------------

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = if (isProbablySpam(result)) {
                    MaterialTheme.colorScheme.errorContainer
                } else {
                    MaterialTheme.colorScheme.secondaryContainer
                }
            )
        ) {

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.Center
            ) {

                Text(
                    text = if (isProbablySpam(result)) {
                        "⚠️ POSSIBILE MESSAGGIO FRAUDOLENTO"
                    } else {
                        "✓ MESSAGGIO PROBABILMENTE SICURO"
                    },
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }

        Spacer(
            modifier = Modifier.height(16.dp)
        )


        // ----------------------------------------------------
        // CARTE "PERCHÉ?"
        // ----------------------------------------------------

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp)
            ) {

                Text(
                    text = "🔎 Perché?",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(
                    modifier = Modifier.height(10.dp)
                )

                // Mostra la spiegazione prodotta dal Repository.
                Text(
                    text = result.explanation,
                    style = MaterialTheme.typography.bodyLarge
                )

                // Mostra i segnali rilevati dal sistema.
                if (result.detectedSignals.isNotEmpty()) {

                    Spacer(
                        modifier = Modifier.height(12.dp)
                    )

                    Text(
                        text = "Segnali rilevati:",
                        style = MaterialTheme.typography.titleMedium
                    )

                    result.detectedSignals.forEach { signal ->

                        Text(
                            text = "• $signal",
                            modifier = Modifier.padding(top = 5.dp),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }

        Spacer(
            modifier = Modifier.height(16.dp)
        )


        // ----------------------------------------------------
        // CARTE "COSA FARE?"
        // ----------------------------------------------------

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {

            Column(
                modifier = Modifier.padding(16.dp)
            ) {

                Text(
                    text = "💡 Cosa fare?",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.primary
                )

                Spacer(
                    modifier = Modifier.height(10.dp)
                )

                if (isProbablySpam(result)) {

                    Text(
                        text = "• Non cliccare sui link sospetti."
                    )

                    Text(
                        text = "• Non fornire dati personali.",
                        modifier = Modifier.padding(top = 5.dp)
                    )

                    Text(
                        text = "• Verifica il mittente del messaggio.",
                        modifier = Modifier.padding(top = 5.dp)
                    )

                    Text(
                        text = "• Non effettuare pagamenti richiesti dal messaggio.",
                        modifier = Modifier.padding(top = 5.dp)
                    )

                } else {

                    Text(
                        text = "• Il messaggio non presenta elementi fortemente sospetti."
                    )

                    Text(
                        text = "• Verifica comunque il mittente.",
                        modifier = Modifier.padding(top = 5.dp)
                    )

                    Text(
                        text = "• Evita di condividere informazioni personali.",
                        modifier = Modifier.padding(top = 5.dp)
                    )
                }
            }
        }

        Spacer(
            modifier = Modifier.height(20.dp)
        )


        // ----------------------------------------------------
        // BUTTON
        // ----------------------------------------------------

        Button(
            onClick = onBackToScanner,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {

            Text(
                text = "Analizza un altro messaggio",
                style = MaterialTheme.typography.titleMedium
            )
        }

        Spacer(
            modifier = Modifier.height(12.dp)
        )

        Text(
            text = "Analisi effettuata con un modello ibrido di Machine Learning e regole.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )
    }
}


/**
 * Determina se il risultato supera la soglia del modello V3.
 *
 * La soglia V3 è circa 38.92/100.
 * Utilizziamo quindi 39 come limite di visualizzazione.
 */
private fun isProbablySpam(
    result: AnalysisResult
): Boolean {

    return result.score >= 39
}


/**
 * Converte il livello di rischio in un testo comprensibile.
 */
private fun getRiskLabel(
    riskLevel: RiskLevel
): String {

    return when (riskLevel) {

        RiskLevel.LOW ->
            "RISCHIO BASSO"

        RiskLevel.MODERATE ->
            "RISCHIO MODERATO"

        RiskLevel.HIGH ->
            "RISCHIO ALTO"

        RiskLevel.CRITICAL ->
            "RISCHIO CRITICO"
    }
}