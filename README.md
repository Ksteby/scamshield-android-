# 🛡️ ScamShield

> Android application for detecting potentially fraudulent messages using a hybrid Machine Learning and rule-based approach.

## 1. Project Overview

ScamShield is an Android application developed in Kotlin to help users identify potentially fraudulent or suspicious messages.

The application analyzes the content of a message and returns a **risk score from 0 to 100**, together with a risk level and useful recommendations for the user.

The project implements a hybrid analysis system combining:

- Machine Learning
- Rule-based analysis

### Main features

- 📱 Offline analysis directly on the device
- 🌐 Online analysis through a REST API
- 🇬🇧 Support for English messages
- 🇮🇹 Support for Italian messages
- 📤 Message reception through the Android Share menu
- 📊 Risk score and detected suspicious signals

---

## 2. Main Features

### Offline Analysis

The offline mode uses the **V3 model** integrated into the application's assets.

The message is analyzed directly on the Android device, so no Internet connection is required.

### Online Analysis

The online mode sends the message to a **FastAPI server** through Retrofit.

The server uses the same V3 model as the offline mode.

### Message Sharing

ScamShield can receive a message through Android's standard **Share** menu.

This allows users to send text from another application directly to ScamShield for analysis.

### Analysis Result

The application displays:

- Risk score from 0 to 100
- Risk level
- Message classification
- General explanation
- Detected suspicious signals
- Recommendations for the user

---

## 3. Architecture

```text
ScannerScreen
      ↓
ScannerViewModel
      ↓
AnalysisRepository
      ↓
 ┌───────────────┬────────────────┐
 │               │                │
Offline        Online             │
 │               ↓                │
 ↓            Retrofit            │
HybridAnalyzer  ↓                 │
 │            FastAPI             │
 │               ↓                │
 └───────────────┴────────────────┘
                 ↓
          AnalysisResult
                 ↓
           ResultScreen
```

The offline and online modes use the same V3 model, allowing the two modes to produce consistent analysis results.

--- 

## 4. Machine Learning

Dataset

The combined dataset contains 9,572 messages:

- 6,825 legitimate messages (ham)
- 2,747 spam messages

The dataset contains messages in both English and Italian.

The dataset was divided into:

- 80% development data
- 20% independent final test set
- V3 Model

The final V3 system uses two complementary text representations.

1. Word TF-IDF
Unigrams and bigrams
ngram_range=(1,2)
2. Character TF-IDF
Character n-grams from 3 to 5
sublinear_tf=True

The two representations are concatenated and used by a Logistic Regression classifier.

In parallel, the system uses a Rule Analyzer based on 14 features.

The analyzed signals include:

- Urgency
- Action requests
- Monetary information
- Sensitive information
- URLs
- Contact information
- Phone numbers
- Interactions between different signals

The final score is obtained by combining the Machine Learning model and the Rule Analyzer.

### Final Test Results

| Metric | ML Model | Hybrid V3 System |
|---|---:|---:|
| Accuracy | 96.61% | **97.34%** |
| Precision | 98.21% | **96.99%** |
| Recall | 89.82% | **93.64%** |
| F1-score | 93.83% | **95.28%** |

The system uses a decision threshold of approximately 38.92/100.

Note: the displayed score represents the system's risk score and should not be interpreted as a calibrated probability that a message is actually fraudulent.

## 5. Technologies

- Android
- Kotlin
- Android Studio
- Jetpack Compose
- Material 3
- ViewModel
- Retrofit
- Gson
- Android Intent / Share
- Local application assets
- Machine Learning & Backend
- Python
- FastAPI
- Uvicorn
- NumPy
- SciPy
- Scikit-learn
- Pandas
- Regular Expressions
- Development & Engineering
- Git
- GitHub
- REST API
- Gradle

## 6. Skills Demonstrated

This project demonstrates practical experience in:

- Android application development
- Kotlin and Jetpack Compose
- Software architecture
- Natural Language Processing (NLP)
- Supervised Machine Learning
- Text feature engineering
- TF-IDF representations
- Classification models
- Model evaluation
- Integration of ML models into mobile applications
- REST API development
- Android ↔ API communication
- Git and GitHub

## 7. Project structure


```text
ScamShield/
├── app/
│   └── src/main/
│       ├── java/com/kemo/scamshield/
│       │   ├── data/
│       │   ├── domain/
│       │   ├── navigation/
│       │   ├── userInterface/
│       │   │   ├── scanner/
│       │   │   ├── result/
│       │   │   └── challenges/
│       │   └── utils/
│       └── assets/
│           └── model/
│               ├── v3_tfidf_word_vocabulary.json
│               ├── v3_tfidf_word_idf.json
│               ├── v3_tfidf_char_vocabulary.json
│               ├── v3_tfidf_char_idf.json
│               ├── v3_ml_model.json
│               ├── v3_rule_model.json
│               └── v3_model_config.json
│
└── ml/
    ├── server.py
    ├── models_v3/
    └── ...
```    


## 8. Installation

Requirements: 
- Android Studio
- JDK compatible with the project
- Python 3
- Android device or emulator
- FastAPI server for online analysis

Android Application:

1. Clone the repository:

git clone https://github.com/Ksteby/scamshield-android-.git
cd scamshield-android-

2. Open the project in Android Studio.
3. Wait for Gradle synchronization.
4. Run:

Build → Make Project

5. Launch the application on an Android device or emulator.

## 9. Running the FastAPI Server

Navigate to the Machine Learning directory:

   cd ml

Activate the Python virtual environment:

  .\.venv\Scripts\Activate.ps1

Start the server:

  uvicorn server:app --host 127.0.0.1 --port 8000

The API can be tested by opening:

  http://127.0.0.1:8000

Expected response:

  {
  "application": "ScamShield API",
  "status": "online"
  }
 
 
 ## 10. Online Mode with the Android Emulator

With the Android emulator running, execute:

  adb reverse tcp:8000 tcp:8000

The application uses:
  http://10.0.2.2:8000/
as the Retrofit BASE_URL.

### Procedure
1. Start the FastAPI server.
2. Run adb reverse tcp:8000 tcp:8000.
3. Launch ScamShield.
4. Select Online mode.
5. Enter or share a message.
6. Press Analyze Message.

## 11. Offline Mode

The offline mode does not require the FastAPI server.

- Open ScamShield.
- Select Offline mode.
- Enter a message.
- Press Analyze Message.

The message is analyzed directly on the device using the integrated V3 model.

## 12. Testing

The following aspects were tested during development:

- Spam messages in English and Italian
- Legitimate messages
- URL detection
- Phone number detection
- Urgency signals
- Action requests
- Offline analysis
- Online analysis
- Consistency between offline and online modes
- Android Share functionality
- Screen navigation
- Text input handling
- Analysis button functionality

## 13. Limitations

The model was trained on SMS datasets and cannot guarantee correct detection of every possible type of scam.

The displayed score represents a system risk score, not a calibrated probability that the message is actually fraudulent.

The online mode requires the FastAPI server to be running and reachable.

## 14. Future Improvements

Possible future developments include:

- Using larger and more diverse datasets
- Periodic model updates
- Support for additional languages
- Improved contextual analysis
- Remote deployment of the backend
- Support for additional message types
- Automated ML model deployment
- Continuous model evaluation

## 15. Author

### Kemo Touohou Steby

Master's student in Data Science for Societal Challenges
Université de Tours, France

Background in Software Engineering
Università degli Studi di Parma, Italy

### Areas of interest
- Data Science
- Machine Learning
- ML Engineering
- MLOps
- Software Engineering
- NLP