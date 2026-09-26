package com.kemo.scamshield.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.kemo.scamshield.userInterface.result.ResultScreen
import com.kemo.scamshield.userInterface.scanner.ScannerScreen
import com.kemo.scamshield.userInterface.scanner.ScannerViewModel

@Composable
fun AppNavigation(
    sharedText: String? = null,
    onSharedTextConsumed: () -> Unit = {}
) {

    val navController = rememberNavController()

    /*
     * Lo stesso ViewModel viene condiviso tra ScannerScreen
     * e ResultScreen per conservare il risultato dell'analisi.
     */
    val scannerViewModel: ScannerViewModel = viewModel()

    /*
     * Gestisce il testo ricevuto tramite la funzione "Condividi".
     *
     * Il testo viene inserito nello Scanner e poi considerato
     * come elaborato per evitare di gestirlo nuovamente.
     */
    LaunchedEffect(sharedText) {

        if (!sharedText.isNullOrBlank()) {

            // Inserisce automaticamente il testo ricevuto
            // nel campo di analisi dello Scanner.
            scannerViewModel.onMessageChange(sharedText)

            // Segnala a MainActivity che il testo è stato elaborato.
            onSharedTextConsumed()

            // Mostra la schermata Scanner.
            navController.navigate("scanner") {

                // Evita di creare più copie dello Scanner
                // nello stack di navigazione.
                launchSingleTop = true
            }
        }
    }

    NavHost(
        navController = navController,
        startDestination = "scanner"
    ) {

        // Schermata principale dell'applicazione.
        composable("scanner") {

            ScannerScreen(
                viewModel = scannerViewModel,

                // Passa alla schermata del risultato
                // dopo un'analisi completata correttamente.
                onAnalysisComplete = {
                    navController.navigate("result")
                }
            )
        }

        // Schermata che visualizza il risultato dell'analisi.
        composable("result") {

            val result =
                scannerViewModel.analysisResult

            if (result != null) {

                ResultScreen(
                    result = result,

                    // Torna allo Scanner per analizzare
                    // un nuovo messaggio.
                    onBackToScanner = {
                        navController.popBackStack()
                    }
                )
            }
        }
    }
}