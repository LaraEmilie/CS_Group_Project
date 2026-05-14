# Riskly – Kreditrisikoanalyse

Riskly ist eine interaktive Web-App, die auf Basis eines trainierten XGBoost-Modells das persönliche Kreditausfallrisiko einschätzt. Als Datenbasis dient der UCI-Datensatz „Default of Credit Card Clients" mit 30.000 anonymisierten Kreditkartenprofilen aus Taiwan. Den Link zum Datensatz finden Sie hier:

https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

Dort finden Sie auch eine detaillierte Erklärung der Bedeutung aller Input-Variablen.
---

## Was die App kann

- **Analyse:** Eigene Kreditdaten eingeben → Risikoklasse (NIEDRIG / MITTEL / HOCH) und Ausfallwahrscheinlichkeit durch das XGBoost-Modell
- **Datenvergleich:** Eigene Rückzahlungsquote im Vergleich zum realen Datensatz
- **Modell-Info:** Modellperformance, Konfusionsmatrix und API-Status
- **Chat:** KI-gestützter Chat (Anthropic Claude) für Rückfragen zum eigenen Risikoprofil

---

## Online ausprobieren

Die App ist deployed und direkt im Browser nutzbar – keine Installation nötig, APIs bereits eingebunden:

🔗 **[Riskly App öffnen](https://csgroupproject-lqbrqwewc2gkfbnqwa6yfx.streamlit.app)**

---

## Installation

Python 3.9 oder höher wird empfohlen.

```bash
pip install streamlit pandas numpy matplotlib requests anthropic joblib xgboost scikit-learn python-calamine openpyxl xlrd
```

---

## App starten

Alle drei Dateien müssen im gleichen Ordner liegen:

```
├── kreditrisiko_app.py
├── best_xgb_model.pkl
└── default of credit card clients.xlsx
```

Dann im Terminal:

```bash
streamlit run kreditrisiko_app.py
```

---

## API-Keys einrichten (optional)

Die App läuft auch ohne API-Keys mit Fallback-Werten. Für den vollen Funktionsumfang (Live-Wechselkurse, freier KI-Chat) können Keys über Streamlit Cloud eingetragen werden:

1. [share.streamlit.io](https://share.streamlit.io) → App → **Settings → Secrets**
2. Eintragen:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
EXCHANGE_API_KEY = "dein-key"
```

> Die Keys niemals direkt in den Code schreiben oder ins GitHub pushen.

---

## Modell

Trainiert im Notebook `credit_default_logreg_xgboost_FINAL.ipynb` als scikit-learn Pipeline:

1. **EngineeringTransformer** – Feature Engineering (Auslastungsquote, Rückzahlungsquote, Zahlungsvolatilität)
2. **XGBClassifier** – mit Class Imbalance Handling, Early Stopping, RandomizedSearchCV

**Performance auf dem Test-Set (n=6.000, Klasse 1 = Zahlungsausfall):**

| Metrik | Wert |
|---|---|
| Accuracy | 76 % |
| Precision | 47 % |
| Recall | 62 % |
| F1-Score | 54 % |
| ROC-AUC | 0.7824 |

Recall ist hier die wichtigste Metrik – das Modell soll möglichst wenige echte Ausfälle verpassen.
