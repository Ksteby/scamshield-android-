package com.kemo.scamshield.userInterface.scanner

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ScannerScreen(
    viewModel: ScannerViewModel,
    onAnalysisComplete: () -> Unit
) {
    val scrollState = rememberScrollState()

    // Il pulsante di analisi è attivo solo quando è presente un messaggio.
    val canAnalyze = viewModel.message.isNotBlank()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(24.dp)
    ) {

        // ------------------------------------------------------------
        // INTESTAZIONE
        // ------------------------------------------------------------

        Text(
            text = "🛡️ ScamShield",
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "Proteggi i tuoi messaggi dalle truffe.",
            style = MaterialTheme.typography.titleMedium
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = "Incolla un messaggio sospetto e ScamShield analizzerà il suo livello di rischio.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ------------------------------------------------------------
        // CAMPO DEL MESSAGGIO
        // ------------------------------------------------------------

        OutlinedTextField(
            value = viewModel.message,
            onValueChange = {
                // Limitiamo il testo a 2000 caratteri.
                if (it.length <= 2000) {
                    viewModel.onMessageChange(it)
                }
            },
            modifier = Modifier.fillMaxWidth(),
            label = {
                Text("Messaggio da analizzare")
            },
            placeholder = {
                Text("Incolla qui il messaggio sospetto...")
            },
            minLines = 6,
            maxLines = 10,
            trailingIcon = {
                if (viewModel.message.isNotEmpty()) {
                    TextButton(
                        onClick = {
                            viewModel.onMessageChange("")
                        }
                    ) {
                        Text("Cancella")
                    }
                }
            },
            supportingText = {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    Text(
                        text = "${viewModel.message.length}/2000"
                    )
                }
            }
        )

        Spacer(modifier = Modifier.height(22.dp))

        // ------------------------------------------------------------
        // SELEZIONE DELLA MODALITÀ
        // ------------------------------------------------------------

        Text(
            text = "Modalità di analisi",
            style = MaterialTheme.typography.titleMedium
        )

        Spacer(modifier = Modifier.height(10.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {

            FilterChip(
                selected = !viewModel.isOnlineMode,
                onClick = {
                    viewModel.changeOnlineMode(false)
                },
                label = {
                    Text("Offline")
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor =
                        MaterialTheme.colorScheme.primaryContainer,
                    selectedLabelColor =
                        MaterialTheme.colorScheme.onPrimaryContainer
                )
            )

            FilterChip(
                selected = viewModel.isOnlineMode,
                onClick = {
                    viewModel.changeOnlineMode(true)
                },
                label = {
                    Text("Online")
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor =
                        MaterialTheme.colorScheme.primaryContainer,
                    selectedLabelColor =
                        MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // ------------------------------------------------------------
        // INFORMAZIONI SULLA MODALITÀ SELEZIONATA
        // ------------------------------------------------------------

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {

                if (!viewModel.isOnlineMode) {

                    Text(
                        text = "📱 Analisi offline",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "Il messaggio viene analizzato direttamente sul dispositivo. Non è necessaria una connessione Internet.",
                        style = MaterialTheme.typography.bodyMedium
                    )

                } else {

                    Text(
                        text = "🌐 Analisi online",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary
                    )

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "È richiesta una connessione Internet.",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // ------------------------------------------------------------
        // PULSANTE DI ANALISI
        // ------------------------------------------------------------

        Button(
            onClick = {
                viewModel.analyzeMessage {
                    onAnalysisComplete()
                }
            },
            enabled = canAnalyze && !viewModel.isAnalyzing,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {

            if (viewModel.isAnalyzing) {

                CircularProgressIndicator(
                    modifier = Modifier.height(20.dp),
                    strokeWidth = 2.dp
                )

                Spacer(modifier = Modifier.padding(horizontal = 6.dp))

                Text("Analisi in corso...")

            } else {

                Text(
                    text = "Analizza messaggio",
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Piccola informazione sulla privacy.
        Text(
            text = if (!viewModel.isOnlineMode) {
                "In modalità offline il messaggio rimane sul dispositivo."
            } else {
                "In modalità online il messaggio viene inviato al server ScamShield."
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(16.dp))
    }
}