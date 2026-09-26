package com.kemo.scamshield.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext


// ------------------------------------------------------------
// COLORI PERSONALIZZATI DI SCAMSHIELD
// ------------------------------------------------------------

private val ScamShieldBlue = Color(0xFF1565C0)
private val ScamShieldBlueLight = Color(0xFF42A5F5)
private val ScamShieldBlueDark = Color(0xFF0D47A1)

private val ScamShieldGreen = Color(0xFF2E7D32)
private val ScamShieldGreenLight = Color(0xFF66BB6A)


// ------------------------------------------------------------
// TEMA SCURO
// ------------------------------------------------------------

private val DarkColorScheme = darkColorScheme(
    primary = ScamShieldBlueLight,
    secondary = ScamShieldGreenLight,
    tertiary = ScamShieldBlue
)


// ------------------------------------------------------------
// TEMA CHIARO
// ------------------------------------------------------------

private val LightColorScheme = lightColorScheme(
    primary = ScamShieldBlue,
    secondary = ScamShieldGreen,
    tertiary = ScamShieldBlueDark
)


@Composable
fun ScamShieldTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),

    // Il colore dinamico viene utilizzato solo su Android 12+.
    dynamicColor: Boolean = false,

    content: @Composable () -> Unit
) {

    val colorScheme = when {

        // Utilizza i colori del dispositivo quando richiesto.
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {

            val context = LocalContext.current

            if (darkTheme) {
                dynamicDarkColorScheme(context)
            } else {
                dynamicLightColorScheme(context)
            }
        }

        // Utilizza il nostro tema scuro.
        darkTheme -> DarkColorScheme

        // Utilizza il nostro tema chiaro.
        else -> LightColorScheme
    }


    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}