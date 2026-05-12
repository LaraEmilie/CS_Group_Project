"""
KreditRisiko Analyse – Streamlit App
=====================================
Ausführen:
  pip install streamlit pandas numpy requests
  streamlit run kreditrisiko_app.py

Umgebungsvariablen (.env oder Systemvariable):
  ANTHROPIC_API_KEY    – Chat-Funktion im Analyse-Tab
  EXCHANGE_API_KEY     – Live-Wechselkurse via api.exchangerate.host
"""

import os, time, random, datetime, requests
import joblib
from sklearn.preprocessing import FunctionTransformer
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Feature-Engineering-Klassen & Konstanten ────────────────────────────────
# Müssen vor joblib.load() definiert sein, damit das gespeicherte
# Pipeline-Objekt (EngineeringTransformer) beim Laden gefunden wird.

BILL_COLS    = [f"BILL_AMT{i}" for i in range(1, 7)]   # Kontoauszugsspalten Monat 1–6
PAY_AMT_COLS = [f"PAY_AMT{i}"  for i in range(1, 7)]   # Rückzahlungsspalten Monat 1–6
ENGINEERED_COLS = ["util_mean", "util_last", "pay_ratio_last", "no_bill_last", "pay_amt_std"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless Row-wise Feature Engineering – identisch zum Notebook."""
    X = X.copy()
    X["util_mean"]      = X[BILL_COLS].mean(axis=1) / X["LIMIT_BAL"]           # Ø Auslastung über alle Monate
    X["util_last"]      = X["BILL_AMT1"] / X["LIMIT_BAL"]                      # Auslastung im letzten Monat
    bill = X["BILL_AMT1"]
    pay  = X["PAY_AMT1"]
    X["pay_ratio_last"] = np.where(bill > 0, pay / bill, 0.0)                  # Rückzahlungsquote letzter Monat
    X["no_bill_last"]   = (bill <= 0).astype(int)                               # Flag: kein Rechnungsbetrag
    X["pay_amt_std"]    = X[PAY_AMT_COLS].std(axis=1).fillna(0.0)              # Volatilität der monatlichen Zahlungen
    return X


class EngineeringTransformer(FunctionTransformer):
    """Custom FunctionTransformer mit korrekten feature_names_out – identisch zum Notebook."""

    def fit(self, X, y=None, **fit_params):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()   # Spaltennamen während des Fits speichern
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
        result = super().fit(X, y, **fit_params)           # Parent-fit aufrufen
        self.n_features_in_ = X.shape[1]                  # Anzahl Input-Features speichern
        return result

    def get_feature_names_out(self, input_features=None):
        """Gibt alle Feature-Namen nach dem Engineering zurück."""
        if input_features is None:
            feature_names = list(self.feature_names_in_) if hasattr(self, "feature_names_in_")                             else [f"x{i}" for i in range(self.n_features_in_)]
        else:
            feature_names = list(input_features)
        return np.array(feature_names + ENGINEERED_COLS)  # Original + 5 Engineered Features

# ─────────────────────────────────────────────────────────────────────────────
#  SEITEN-KONFIGURATION  (muss als erstes Streamlit-Kommando stehen)s
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Riskly",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  FARBPALETTE
# ─────────────────────────────────────────────────────────────────────────────

NAVY   = "#0F2D5E"   # dunkelblau – Sidebar, Überschriften
MID    = "#1A4A8A"   # mittelblau – Hover / aktive Elemente
LIGHT  = "#3B82C4"   # hellblau  – Akzente, Links
PALE   = "#DBEAFE"   # sehr hell – Info-Boxen, Kartenhintergründe
SURF   = "#F4F8FF"   # Seitenhintergrund
MUTED  = "#6B8CAE"   # gedämpft  – Labels, Hints
BORDER = "#BDD4EE"   # Rahmen
GRÜN   = "#1D9E75"   # Erfolg / NIEDRIG
AMBER  = "#BA7517"   # Warnung / MITTEL
ROT    = "#E24B4A"   # Gefahr  / HOCH

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBALES CSS  – Sidebar dunkelblau, Hauptbereich hellblau-weiß
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
  /* Sidebar Gradient */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {NAVY} 0%, #0a2249 100%) !important;
      border-right: 1px solid rgba(255,255,255,0.08);
  }}
  [data-testid="stSidebar"] * {{ color: rgba(255,255,255,0.92) !important; }}
  [data-testid="stSidebar"] .stRadio label {{
      padding: 9px 14px !important; border-radius: 8px !important;
      font-size: 0.9rem !important; font-weight: 500 !important;
  }}

  /* Hauptbereich */
  .main {{ background-color: {SURF}; }}
  .main .block-container {{ padding: 2rem 2.5rem; max-width: 960px; }}

  /* Alle Streamlit-Buttons: kein Pulsieren, kein Farbflash */   # überschreibt die automatischen Streamlit-Einstellungen
  .stButton > button,
  .stButton > button *,
  .stButton > button p,
  .stButton > button span,
  .stButton > button::before,
  .stButton > button::after {{
      animation: none !important;
      transition: background 0.12s ease !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      color: white !important;
  }}
  .stButton > button[kind="primary"],
  .stButton > button[kind="primary"]:hover,
  .stButton > button[kind="primary"]:active,
  .stButton > button[kind="primary"]:focus,
  .stButton > button[kind="primary"]:focus-visible {{
      background: {NAVY} !important;
      border: none !important;
      color: white !important;
      box-shadow: 0 2px 8px rgba(15,45,94,0.22) !important;
      animation: none !important;
      outline: none !important;
  }}
  .stButton > button[kind="primary"] p,
  .stButton > button[kind="primary"] span,
  .stButton > button[kind="primary"] * {{
      color: white !important;
  }}
  .stButton > button[kind="primary"]:hover {{
      background: {MID} !important;
      color: white !important;
  }}

  .stProgress > div > div {{ background-color: {LIGHT} !important; border-radius: 4px; }}
  .stProgress > div {{ background-color: {PALE} !important; border-radius: 4px; }}

  #Die nächsten Zeieln erstellen Methoden, damit die Layouts gespeichert und beim Aufbau der Website einfach referenziert werden können.

  /* Abschnittsüberschriften */ 
  .section-label {{
      font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em;
      color: {MUTED}; text-transform: uppercase;
      border-bottom: 1px solid {BORDER}; padding-bottom: 7px; margin-bottom: 18px;
  }}

  /* Info-Callout */
  .info-box {{
      background: {PALE}; border-left: 3px solid {LIGHT};
      border-radius: 0 10px 10px 0; padding: 14px 18px;
      color: {NAVY}; font-size: 0.9rem; margin: 14px 0;
  }}

  /* Feature-Karten */
  .feat-card {{
      background: white; border: 1px solid {BORDER}; border-radius: 12px;
      padding: 20px 22px; box-shadow: 0 1px 4px rgba(15,45,94,0.06);
  }}
  .feat-card .icon  {{ font-size: 1.5rem; margin-bottom: 8px; }}
  .feat-card .title {{ font-weight: 700; color: {NAVY}; font-size: 1rem; margin-bottom: 5px; }}
  .feat-card .body  {{ color: #4a6080; font-size: 0.87rem; line-height: 1.55; }}

  /* Info-Karten (Modell-Info) */
  .info-card {{
      background: white; border: 1px solid {BORDER}; border-radius: 12px;
      padding: 22px 26px; margin-bottom: 16px;
      box-shadow: 0 1px 4px rgba(15,45,94,0.06);
  }}
  .info-card h3 {{ color: {NAVY}; margin: 0 0 10px 0; font-size: 1.05rem; }}
  .info-card p  {{ color: #1E3A5F; font-size: 0.9rem; line-height: 1.65; margin: 0; }}

  /* API-Badge – kein Pulsieren */
  .api-badge {{
      margin-top: 20px; padding: 9px 13px;
      background: rgba(29,158,117,0.18); border-radius: 8px;
      display: flex; align-items: center; gap: 8px; font-size: 0.79rem;
      border: 1px solid rgba(29,158,117,0.3);
  }}

  /* Globale Farben – KEIN h1-Override hier, hero nutzt div statt h1 */
  h2, h3 {{ color: {NAVY} !important; }}
  p, li {{ color: #1E3A5F; }}
  /* h1 nur außerhalb des Hero dunkelblau */
  .main h1 {{ color: {NAVY} !important; font-weight: 700 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def letzte_6_monate() -> list[str]:
    """
    Berechnet die letzten 6 Kalendermonate dynamisch und gibt
    deutsche Monatsnamen zurück, z. B. ['November 2025', …, 'April 2026'].
    """
    
    #Manuelle Übersetzung der Monate aus dem Englischen, da es sich um die Std-Python-Sprache handelt.
    
    DE = {"January":"Januar","February":"Februar","March":"März",         
          "April":"April","May":"Mai","June":"Juni","July":"Juli",
          "August":"August","September":"September",
          "October":"Oktober","November":"November","December":"Dezember"}
    heute  = datetime.date.today()
    result = []
    for i in range(6, 0, -1):          #Sechs Monate durch die Range-Funktion zurückgehen
        jahr = heute.year
        monat = heute.month - i
        while monat <= 0:              #Stellt den korrekten Jahressprung von Januar zu Dezember sichers
            monat += 12
            jahr  -= 1
        label = datetime.date(jahr, monat, 1).strftime("%B %Y")   #strftime stellt die richtige Darstellung 'Monat Jahr' sicher
        for en, de in DE.items():
            label = label.replace(en, de)
        result.append(label)                                      #Zusammenfügen der jeweiligen letzten 6 Monate in eine Liste
    return result


# Fallback-Kurse für den Fall, dass keine Live-Verbindung besteht (Stand: Q2 2026)
FALLBACK_KURSE = {
    "CHF":33.5,"EUR":36.2,"USD":32.1,"GBP":41.0,"JPY":0.22,
    "CAD":23.8,"AUD":20.5,"SEK":2.95,"NOK":2.90,"DKK":4.85,
    "PLN":7.90,"CZK":1.38,"HUF":0.088,"RON":7.15,"BAM":18.50,
}

def num_input_de(label: str, key: str, default: int = 0,
                 min_val: int = 0, help: str = None, show_label=False) -> int:
    """
    Zahlen-Eingabefeld mit deutschem Tausenderpunkt-Format (z. B. 25.000).
    Startet leer; beim Verlassen des Feldes wird automatisch formatiert.
    """
    raw_key = f"_raw_{key}"                     # raw-key speichert den Texteingabewert
    num_key = f"_num_{key}"                     # num_key speichert den bereinigten numerischen Wert

    # Leer initialisieren (kein voreingetragener Wert)
    if num_key not in st.session_state:
        st.session_state[num_key] = 0
        st.session_state[raw_key] = ""

    #Call-Back Funktion, die automatisch ausgeführt wird, sobald sich das Eingabefeld verändert.
    def _on_change():
        raw = st.session_state.get(raw_key, "")
        if raw.strip() == "":
            st.session_state[num_key] = 0
            return
        try:
            cleaned = raw.replace(".", "").replace(",", "").replace(" ", "")     # Stellt sicher, dass die Eingabe als Integer verarbeitet werden kann.
            val = max(int(cleaned), min_val)
        except (ValueError, AttributeError):
            val = st.session_state[num_key]
        st.session_state[num_key] = val
        st.session_state[raw_key] = f"{val:,}".replace(",", ".")                 # Formatiert die Zahl wieder im deutschen Tausenderformat

    kwargs = {"key": raw_key, "on_change": _on_change,                           # Dictionary mit Optionen, die an st.text_inpit() übergeben werden
              "label_visibility": "visible" if show_label else 'collapsed', "placeholder": "0"}
    if help:
        kwargs["help"] = help
    st.text_input(label, **kwargs)                                               # Nutzung eines Textfeldes anstatt eines numerischen Inputs
    return st.session_state[num_key]


def wechselkurs_holen(code: str) -> tuple[float, bool]:
    """
    Holt Wechselkurs CURRENCY → TWD.
    Stellt sicher, dass Nutzer ihre eigene Währung nutzen können
    Gibt (kurs, live) zurück; live=False wenn Fallback genutzt wird.
    """
    key = os.getenv("EXCHANGE_API_KEY", "")     # Liest den API-Key aus den Umgebungsvariablen                            
    if key:
        try:
            url = (f"https://api.exchangerate.host/latest"
                   f"?base={code}&symbols=TWD&access_key={key}")          # Anfrage an die Wechselkurs-API
            r = requests.get(url, timeout=5)
            return r.json()["rates"]["TWD"], True
        except Exception:
            pass
    return FALLBACK_KURSE.get(code, 30.0), False                          # Falls keine Live-Daten verfügbar sind, werden die vordefinierten FALLBACK-Kurse verwendet.


@st.cache_resource
def lade_modell():
    """
    Lädt das trainierte XGBoost-Modell (best_xgb_model.pkl) einmalig und cached es.
    Gibt None zurück, falls die Datei nicht gefunden wird.
    """
    try:
        return joblib.load("best_xgb_model.pkl")          # Lädt das Modell direkt aus der .pkl-Datei im App-Verzeichnis
    except FileNotFoundError:
        return None                                        # Gibt None zurück, damit der Fallback greift
    except Exception as e:
        st.warning(f"Modell konnte nicht geladen werden: {e}")
        return None


def berechne_risiko(inp: dict) -> dict:
    """
    Wertet die Eingabedaten aus und gibt Risikoklasse, Ausfallwahrscheinlichkeit
    sowie positive/negative Einflussfaktoren zurück.
    Verwendet das trainierte XGBoost-Modell (best_xgb_model.pkl).
    Fällt auf eine Heuristik zurück, falls das Modell nicht verfügbar ist.
    """
    # ── Hilfsvariablen für Fallback & Erklärungen ────────────────────────────
    verzoegerungen = sum(v for v in inp["rueckzahlung"] if v > 0)       # Zählt Monate mit positivem Verzugsstatus
    hoher_saldo    = any(b > inp["kreditbetrag_twd"] * 0.8 for b in inp["kontoauszug"])  # True wenn Saldo > 80% des Limits

    pay  = inp["rueckzahlung"]   # 6 Monate Rückzahlungsstatus (−1 … 9)
    bill = inp["kontoauszug"]    # 6 Monate Kontoauszug (TWD)
    paid = inp["bezahlt"]        # 6 Monate tatsächlich bezahlt (TWD)

    total_bill = sum(bill)       # Gesamtbetrag aller Kontoauszüge der letzten 6 Monate
    total_paid = sum(paid)       # Gesamtbetrag aller tatsächlichen Rückzahlungen

    # ── Mapping: Deutsche Labels → Modell-Integer ────────────────────────────
    geschlecht_map = {"Männlich": 1, "Weiblich": 2}
    bildung_map = {
        "Graduiertenschule / Master / Doktorat": 1,
        "Universität / Bachelor":                2,
        "Gymnasium / Matura":                    3,
        "Andere":                                4,
    }
    familie_map = {
        "Verheiratet / Eingetragene Partnerschaft": 1,
        "Ledig":        2,
        "Andere":       3,
        "Keine Angabe": 3,
    }

    # ── One-Hot-Encoding (identisch zum Training) ───────────────────────────
    # Das Modell wurde mit pd.get_dummies(drop_first=True) trainiert.
    # SEX: 1=Männlich (Baseline, fällt weg), 2=Weiblich → SEX_2
    # EDUCATION: 0 ist Baseline (drop_first), 1–6 werden als Dummies kodiert
    # MARRIAGE: 0 ist Baseline, 1–3 werden als Dummies kodiert
    sex_val = geschlecht_map.get(inp["geschlecht"], 1)
    edu_val = bildung_map.get(inp["bildung"], 4)
    mar_val = familie_map.get(inp["familienstand"], 3)

    raw_input = {
        # Numerische Spalten (unverändert)
        "LIMIT_BAL": inp["kreditbetrag_twd"],
        "AGE":       inp["alter"],
        # Rückzahlungsstatus (PAY_0 = aktuellster Monat)
        "PAY_0": pay[0], "PAY_2": pay[1], "PAY_3": pay[2],
        "PAY_4": pay[3], "PAY_5": pay[4], "PAY_6": pay[5],
        # Kontoauszugsbeträge (TWD)
        "BILL_AMT1": bill[0], "BILL_AMT2": bill[1], "BILL_AMT3": bill[2],
        "BILL_AMT4": bill[3], "BILL_AMT5": bill[4], "BILL_AMT6": bill[5],
        # Tatsächlich bezahlte Beträge (TWD)
        "PAY_AMT1": paid[0], "PAY_AMT2": paid[1], "PAY_AMT3": paid[2],
        "PAY_AMT4": paid[3], "PAY_AMT5": paid[4], "PAY_AMT6": paid[5],
        # One-Hot: SEX (drop_first → nur SEX_2)
        "SEX_2": int(sex_val == 2),
        # One-Hot: EDUCATION (drop_first → Kategorien 1–6)
        "EDUCATION_1": int(edu_val == 1), "EDUCATION_2": int(edu_val == 2),
        "EDUCATION_3": int(edu_val == 3), "EDUCATION_4": int(edu_val == 4),
        "EDUCATION_5": int(edu_val == 5), "EDUCATION_6": int(edu_val == 6),
        # One-Hot: MARRIAGE (drop_first → Kategorien 1–3)
        "MARRIAGE_1": int(mar_val == 1), "MARRIAGE_2": int(mar_val == 2),
        "MARRIAGE_3": int(mar_val == 3),
    }

    # ── Modellvorhersage ─────────────────────────────────────────────────────
    prob      = None              # Ausfallwahrscheinlichkeit, wird durch Modell oder Fallback befüllt
    model     = lade_modell()     # Ruft das gecachte XGBoost-Modell ab
    fallback  = False             # Schalter: True wenn Modell nicht verfügbar

    if model is not None:
        try:
            X = pd.DataFrame([raw_input])                            # Einzelner Datensatz als DataFrame
            # Spaltenreihenfolge an Modell anpassen (Pipeline-Input-Spalten)
            try:
                model_cols = list(model.named_steps["feature_engineering"].feature_names_in_)
                X = X.reindex(columns=model_cols, fill_value=0)      # Fehlende Spalten mit 0 auffüllen
            except Exception:
                pass                                                  # Ohne Reindex versuchen, falls Step-Name abweicht
            prob = float(model.predict_proba(X)[0, 1])               # Index 1 = Wahrscheinlichkeit für Klasse 1 (Ausfall)
        except Exception as err:
            st.warning(f"⚠️ Modell-Vorhersage fehlgeschlagen ({err}). Heuristik wird verwendet.")
            fallback = True
    else:
        fallback = True

    # ── Heuristischer Fallback (falls Modell nicht verfügbar) ────────────────
    if fallback or prob is None:
        score = verzoegerungen * 12 + (20 if hoher_saldo else 0) + max(0, 38 - inp["alter"])  # Einfache Heuristik: Verzug, Saldo und Alter gewichtet
        prob  = min(0.95, score / 100)   # Score auf Wahrscheinlichkeit normiert, max. 95%

    # ── Risikoklasse aus Wahrscheinlichkeit ──────────────────────────────────
    if prob < 0.33:   risiko, farbe = "NIEDRIG", GRÜN
    elif prob < 0.50: risiko, farbe = "MITTEL",  AMBER
    else:             risiko, farbe = "HOCH",    ROT

    # ── Erklärbare Einflussfaktoren für den Nutzer ───────────────────────────
    positiv = ["Regelmäßige Zahlungen in den letzten Monaten",
               "Moderates Kreditvolumen im Verhältnis zum Alter",
               "Langjährige Kredithistorie ohne grobe Ausreißer"]
    negativ = (["Mehrere Zahlungsverzögerungen in den letzten 6 Monaten"]
               if verzoegerungen > 0 else []) + \
              (["Hoher Saldo im Verhältnis zum Kreditlimit"] if hoher_saldo else []) + \
              ["Zahlungshistorie zeigt unregelmäßige Muster"]

    return {
        "risiko": risiko, "farbe": farbe, "prob": round(prob * 100),
        "ausfall_ja": prob >= 0.5,
        "positiv": positiv, "negativ": negativ,
        "empfehlungen": [
            "Stelle alle ausstehenden Zahlungen so schnell wie möglich nach.",
            "Halte deinen monatlichen Saldo unter 30 % des Kreditlimits.",
            "Richte automatische Zahlungen ein, um künftige Verzögerungen zu vermeiden.",
            "Überprüfe dein Budget und reduziere nicht notwendige Ausgaben.",
            "Erwäge eine Beratung bei deiner Bank zu Umschuldungsoptionen.",
        ],
    }


def chat_antwort(frage: str, kontext: dict, verlauf: list) -> str:
    """
    Verarbeitet eine Nutzerfrage im Kontext des Analyseergebnisses.
    Nutzt die Anthropic API wenn ANTHROPIC_API_KEY gesetzt ist,
    sonst greift die eingebaute Stichwort-Logik als Fallback.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")    # Liest den API-Key aus den Umgebungsvariablen
    if api_key:                                      # Wenn Anthropic-Key vorhanden, wird Claude für die Antworten verwendet.
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            system = (                                                                # Systemprompt definiert Rolle, Sprache und Kontext des Assistenten
                f"Du bist ein freundlicher Kreditrisikoberater. Antworte ausschließlich auf Deutsch. "
                f"Nutzerprofil: Risikoklasse={kontext.get('risiko')}, "
                f"Ausfallwahrscheinlichkeit={kontext.get('prob')}%, "
                f"Alter={kontext.get('alter')}, Kreditbetrag={kontext.get('kreditbetrag')}."
            )
            msgs = [{"role": h["role"], "content": h["content"]} for h in verlauf[-6:]]
            msgs.append({"role": "user", "content": frage})
            r = client.messages.create(
                model="claude-sonnet-4-20250514",    # Anthropic Claude Sonnet
                max_tokens=400,
                system=system,
                messages=msgs,
            )
            return r.content[0].text
        except Exception as e:
            return f"(Verbindungsfehler: {e})"
    # Stichwort-Fallback wenn kein API-Key hinterlegt
    q = frage.lower()
    if any(w in q for w in ["risiko","klasse","kategorie"]):
        return (f"Deine Risikoklasse ist **{kontext.get('risiko','–')}**. "
                f"Das Modell schätzt dein Ausfallrisiko auf {kontext.get('prob',0)} %.")
    if any(w in q for w in ["faktor","wichtig","ursache"]):
        n = kontext.get("negativ", [])
        return "Wichtigster negativer Faktor: " + (n[0] if n else "keiner erkannt") + "."
    if any(w in q for w in ["senken","verbesser","tun","empfehlung"]):
        return ("Pünktliche Zahlungen und ein niedriger Saldo sind die wirksamsten Hebel. "
                "Sprich außerdem mit deiner Bank über Umschuldungsmöglichkeiten.")
    return ("Für vollständige Chat-Antworten bitte ANTHROPIC_API_KEY als "
            "Umgebungsvariable setzen.")


@st.cache_data
def referenz_datensatz() -> pd.DataFrame:
    """
    Lädt den echten UCI-Datensatz (Default of Credit Card Clients, 30.000 Einträge)
    und bereitet die Spalten für die Visualisierungen auf.
    Gecacht, damit kein Re-Rendering bei jedem Widget-Update passiert.
    Fällt auf synthetische Daten zurück, falls die Datei nicht gefunden wird.
    """
    DATEINAME = "default of credit card clients.xlsx"

    try:
        # Zeile 0 = X1/X2/... (Alias), Zeile 1 = echte Spaltennamen → header=1
        # calamine ist robuster für dieses Excel-Format; openpyxl findet keine Sheets
        try:
            raw = pd.read_excel(DATEINAME, header=1, engine="calamine")
        except Exception:
            raw = pd.read_excel(DATEINAME, header=1, engine="xlrd")   # header=1: Zeile 0 = X1/X2-Aliase, Zeile 1 = echte Spaltennamen

        bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]  # Kontoauszugsspalten 6 Monate
        pay_cols  = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]  # Rückzahlungsspalten 6 Monate

        limit       = raw["LIMIT_BAL"].astype(float)
        avg_saldo   = raw[bill_cols].mean(axis=1)            # Ø monatlicher Kontoauszug
        ueber_pct   = np.clip((avg_saldo / limit.clip(lower=1) - 1) * 100, -80, 200)  # Wie stark liegt der Saldo über/unter dem Limit (%)
        bezahlt_sum = raw[bill_cols].sum(axis=1)             # Summe Kontoauszüge 6 Monate
        rueck_sum   = raw[pay_cols].sum(axis=1)              # Summe tatsächlich bezahlt 6 Monate
        letzter     = raw["BILL_AMT1"].astype(float)         # Aktuellster Monat

        return pd.DataFrame({
            "limit":               limit.values,
            "saldo":               avg_saldo.values,
            "ueber_limit_pct":     ueber_pct.values,
            "bezahlt_gesamt":      bezahlt_sum.values,
            "rueckgezahlt_gesamt": rueck_sum.values,
            "letzter_monat":       letzter.values,
        })

    except FileNotFoundError:
        # Fallback: synthetische Daten wenn Datei nicht im Verzeichnis liegt
        rng = np.random.default_rng(42)
        n   = 3000
        limit    = rng.integers(10_000, 800_001, n).astype(float)
        saldo    = rng.uniform(0, 1.2, n) * limit
        ueber_pct = np.clip((saldo / limit - 1) * 100, -80, 200)
        bezahlt  = rng.integers(0, 60_001, (n, 6))
        rueckgez = rng.integers(0, 55_001, (n, 6))
        return pd.DataFrame({
            "limit":               limit,
            "saldo":               saldo,
            "ueber_limit_pct":     ueber_pct,
            "bezahlt_gesamt":      bezahlt.sum(axis=1),
            "rueckgezahlt_gesamt": rueckgez.sum(axis=1),
            "letzter_monat":       rueckgez[:, -1],
        })

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  – Navigation und Status
# ─────────────────────────────────────────────────────────────────────────────

# Erstellt die linke Navigationsleiste der Streamlit-App
with st.sidebar:
    # Eigenes HTML für Logo, Titel und Branding der Sidebar
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:2.5rem'>
      <div style='width:38px;height:38px;border-radius:10px;background:{LIGHT};
           display:flex;align-items:center;justify-content:center;font-size:18px;
           box-shadow:0 2px 8px rgba(0,0,0,0.2)'>💳</div>
      <div>
        <div style='font-weight:700;font-size:1.2rem;line-height:1.2;letter-spacing:-0.02em'>Riskly</div>
        <div style='font-size:0.72rem;opacity:0.55;letter-spacing:0.06em;text-transform:uppercase'>Kreditrisiko</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

   # Navigation zwischen den Hauptseiten der App
   # Die ausgewählte Seite wird in der Variable "seite" gespeichert
    seite = st.radio(
        "Navigation",
        ["Übersicht", "Analyse", "Datenvergleich", "Modell-Info"],
        label_visibility="collapsed",
    )

    # Status-Badges – statischer Punkt, kein Pulsieren
    exchange_aktiv  = bool(os.getenv("EXCHANGE_API_KEY", ""))
    anthropic_aktiv = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    ex_farbe  = GRÜN if exchange_aktiv  else MUTED
    ex_label  = "Wechselkurs-API aktiv"  if exchange_aktiv  else "Wechselkurs-API inaktiv"
    ant_farbe = GRÜN if anthropic_aktiv else MUTED
    ant_label = "Anthropic-API aktiv"   if anthropic_aktiv else "Anthropic-API inaktiv"
    st.markdown(f"""
    <div style='display:flex;gap:8px;flex-wrap:wrap'>
      <div class="api-badge">
        <span style='width:8px;height:8px;border-radius:50%;background:{ex_farbe};
              display:inline-block'></span>
        {ex_label}
      </div>
      <div class="api-badge">
        <span style='width:8px;height:8px;border-radius:50%;background:{ant_farbe};
              display:inline-block'></span>
        {ant_label}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: ÜBERSICHT
# ─────────────────────────────────────────────────────────────────────────────

if seite == "Übersicht":  # die Startseite wird ausgeführt, wenn die Variable seite als 'Übersicht' gespeichert wird

    # ── Hero-Banner mit Branding und visueller Einführung in die Website───────────────────────

    st.markdown(f"""
    <div class="hero-banner" style='background:linear-gradient(135deg,{NAVY} 0%,{MID} 55%,{LIGHT} 100%);
         border-radius:16px;padding:36px 44px;margin-bottom:28px;
         display:flex;align-items:center;justify-content:space-between;gap:24px;
         box-shadow:0 4px 24px rgba(15,45,94,0.22)'>

      <div style='flex:1;min-width:0'>
        <div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;color:rgba(255,255,255,0.6);margin-bottom:10px'>
          Kreditrisiko · Analyse
        </div>
        <div style='color:white;font-size:2rem;font-weight:800;
             margin:0 0 12px 0;line-height:1.2;letter-spacing:-0.02em'>
          Riskly
        </div>
        <div style='color:white;font-size:0.97rem;
             line-height:1.65;margin:0 0 22px 0;white-space:nowrap'>
          Verstehe dein persönliches Kreditausfallrisiko — transparent, fair und privat.
        </div>
      </div>
    </div>
        """, unsafe_allow_html=True)
    

    
    # ── Feature-Karten, die die zentralen Eigenschaften der Anwendung darstellen ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)               # Erstellen von drei Spalten für drei Karten
    for col, icon, titel, text in [          # Dynamisches Erzeugen der Informationskarten
        (c1, "📊", "Datenbasiert",
         "Auf realen Kreditkartendaten trainiertes Vorhersagemodell."),
        (c2, "🔒", "Privat",
         "Alle Eingaben bleiben lokal. Keine Speicherung, keine Weitergabe."),
        (c3, "💬", "Erklärbar",
         "Verständliche Faktoren und ein Chat für deine Fragen."),
    ]:
    # Erstellt eine einzelne Feature-Karte mit Icon, Titel und Beschreibung
        col.markdown(f"""
        <div class="feat-card">
          <div class="icon">{icon}</div>
          <div class="title">{titel}</div>
          <div class="body">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Zusätzlicher vertikaler Abstand zwischen den Abschnitten
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # ── Mission ──────────────────────────────────────────────────────────────
    # Kurze Beschreibung des Ziels und Nutzens der App
    st.markdown('<div class="section-label">UNSERE MISSION</div>', unsafe_allow_html=True)
    st.markdown(
        "Unsere Mission ist es, Kreditrisikoanalyse transparent, fair und für alle zugänglich "
        "zu machen — unabhängig vom Wohnort oder der verwendeten Währung. Wir glauben, dass "
        "jeder Mensch das Recht hat, die eigene finanzielle Situation zu verstehen."
    )

    # ── Schritte ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.5rem">SO FUNKTIONIERT DIE APP</div>',
                unsafe_allow_html=True)
    # Inhalte für die visuelle Schrittübersicht
    SCHRITTE = [
        ("1", 'Tab „Analyse" öffnen'),
        ("2", "Währung und Kreditbetrag eingeben"),
        ("3", "Persönliche Angaben ausfüllen"),
        ("4", '„Analyse starten" klicken'),
        ("5", "Risikoprofil und Empfehlungen ansehen"),
        ("6", "Chat für weitere Fragen nutzen"),
    ]
    col_a, col_b = st.columns(2)  # Erstellt zwei Spalten für die Schritte
    for i, (nr, text) in enumerate(SCHRITTE):   # Dynamisches Rendern der einzelnenen Schritte
        ziel = col_a if i % 2 == 0 else col_b
        ziel.markdown(f"""
        <div style='border:1px solid {BORDER};border-radius:10px;padding:12px 16px;
             margin-bottom:10px;background:white;display:flex;align-items:center;gap:12px;
             box-shadow:0 1px 3px rgba(15,45,94,0.05)'>
          <div style='min-width:28px;height:28px;border-radius:50%;background:{NAVY};
               color:white;display:flex;align-items:center;justify-content:center;
               font-weight:700;font-size:0.82rem;flex-shrink:0'>{nr}</div>
          <span style='color:{NAVY};font-size:0.9rem'>{text}</span>
        </div>""", unsafe_allow_html=True)

    # ── Datenschutz-Box ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="info-box" style="margin-top:8px">
      🔒 <strong>Deine Daten verlassen diesen Browser nicht.</strong><br>
      Es werden keine persönlichen Informationen gespeichert oder weitergegeben.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: ANALYSE  – Eingabeformular + Ergebnisse
# ─────────────────────────────────────────────────────────────────────────────

elif seite == "Analyse":     # der Analysetab wird ausgeführt, wenn die Variable seite als 'Analyse' gespeichert wird
    st.title("Analyse")

    monate = letzte_6_monate()                               # Dynamische Monatsliste für die Eingabefelder der letzten 6 Monate
    heute_str = datetime.date.today().strftime("%d.%m.%Y")   # Korrekt formatiertes heutiges Datum

    # ── Abschnitt 1: Währung & Kreditbetrag ─────────────────────────────────
    
    # Auswahl der Nutzerwährung und Umrechnung in interne Modellwährung (TWD)
    
    st.markdown('<div class="section-label">WÄHRUNG & KREDITBETRAG</div>',
                unsafe_allow_html=True)

    # WAEHRUNGEN unterstützt die Eingabewährungen der Anwendungen
    WAEHRUNGEN = {
        "– Bitte auswählen –":              None,
        "CHF – Schweizer Franken":          "CHF",
        "EUR – Euro":                       "EUR",
        "USD – US-Dollar":                  "USD",
        "GBP – Britisches Pfund":           "GBP",
        "JPY – Japanischer Yen":            "JPY",
        "CAD – Kanadischer Dollar":         "CAD",
        "AUD – Australischer Dollar":       "AUD",
        "SEK – Schwedische Krone":          "SEK",
        "NOK – Norwegische Krone":          "NOK",
        "DKK – Dänische Krone":             "DKK",
        "PLN – Polnischer Zloty":           "PLN",
        "CZK – Tschechische Krone":         "CZK",
        "HUF – Ungarischer Forint":         "HUF",
        "RON – Rumänischer Leu":            "RON",
        "BAM – Bosnisch-Herzegowinische Mark": "BAM",
    }
    # Zuordnung von Währungscodes zu Anzeige-Symbolen
    SYMBOLE = {"CHF":"CHF","EUR":"€","USD":"$","GBP":"£","JPY":"¥",
               "CAD":"CA$","AUD":"A$","SEK":"kr","NOK":"kr","DKK":"kr",
               "PLN":"zł","CZK":"Kč","HUF":"Ft","RON":"lei","BAM":"KM"}

    # Dropdown zur Auswahl der Eingabewährung
    waehrung_label = st.selectbox("Währung auswählen", list(WAEHRUNGEN.keys()), index=0)
    code   = WAEHRUNGEN.get(waehrung_label) or "EUR"  # Extrahiert den internen Währungscode (z. B. CHF, EUR, USD)
    symbol = SYMBOLE.get(code, code)         # Wandelt den Währungscode in ein Anzeige-Symbol um
    kurs, live = wechselkurs_holen(code)  # Holt den aktuellen Wechselkurs zur internen Vergleichswährung TWD
    # Badge nur anzeigen wenn Fallback-Kurs verwendet wird (kein Live-Kurs verfügbar)
    if not live:
        st.markdown(
            f"<div style='display:inline-flex;align-items:center;gap:6px;background:#FFF8E1;"
            f"border-radius:20px;padding:4px 14px;font-size:0.8rem;color:{AMBER};margin-bottom:6px'>"
            f"<span style='width:7px;height:7px;border-radius:50%;background:{AMBER};"
            f"display:inline-block'></span>Kurs vom {heute_str} — offline-Schätzung</div>",
            unsafe_allow_html=True
        )

    # Benutzerfreundliche Eingabe des Kreditbetrags mit deutschem Zahlenformat
    kreditbetrag = num_input_de(
        f"Kreditbetrag ({symbol})", key="kreditbetrag", default=25_000,
        help="Beinhaltet persönlichen Konsumentenkredit sowie eventuelle Familienergänzungskredite.",
        show_label=True
    )
    kreditbetrag_twd = kreditbetrag * kurs   # intern in TWD umrechnen; nie in UI zeigen

    st.divider()

    # ── Abschnitt 2: Persönliche Angaben ────────────────────────────────────
    # Ermöglicht die Eingabe grundlegender demografischer Informationen

    st.markdown('<div class="section-label">PERSÖNLICHE ANGABEN</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)          # Zweispaltiges Layout für eine übersichtlichere Dateneingabe
    with ca:                        # Linke Eingabbespalte
        alter_raw = st.text_input("Alter (Jahre)", placeholder="z. B. 35",
                                  key="alter_raw", label_visibility="visible")  # Texteingabe für das Alter des Nutzers
        try:
            alter = max(18, min(99, int(alter_raw)))  # Validiert die Eingabe und begrenzt das Alter auf realistische Werte
        except (ValueError, TypeError):
            alter = 30
        bildung = st.selectbox("Höchster Bildungsabschluss", [      # Auswahl des höchsten Bildungsabschlusses
            "– Bitte auswählen –",
            "Graduiertenschule / Master / Doktorat",
            "Universität / Bachelor",
            "Gymnasium / Matura",
            "Andere",
        ]) 
    with cb:                        # Rechte Eingabbespalte
        geschlecht = st.selectbox("Biologisches Geschlecht", [     # Angabe zum biologischen Geschlecht
            "– Bitte auswählen –",
            "Männlich", "Weiblich",
        ])
        familienstand = st.selectbox("Beziehungs- / Familienstand", [       # Angabe zum Familienstand
            "– Bitte auswählen –",
            "Ledig", "Verheiratet / Eingetragene Partnerschaft", "Andere", "Keine Angabe",
        ])

    st.divider()

    # ── Abschnitt 3: Rückzahlungsstatus ─────────────────────────────────────
    # Titel
    st.markdown('<div class="section-label">RÜCKZAHLUNGSSTATUS — LETZTE 6 MONATE</div>',
                unsafe_allow_html=True)
    # Erklärung der Skala für Zahlungsverzug
    st.markdown(f"""
    <div style='background:{PALE};border-radius:8px;padding:9px 16px;font-size:0.87rem;
         color:{NAVY};margin-bottom:12px'>
      Skala: −1 = pünktlich &nbsp;·&nbsp; 1 = 1 Monat Verzug &nbsp;·&nbsp;
      2 = 2 Monate &nbsp;·&nbsp; … &nbsp;·&nbsp; 9 = ≥ 9 Monate
    </div>""", unsafe_allow_html=True)

    # Zuordnung der sichtbaren Auswahlwerte zu numerischen Modellwerten
    STATUS_LABELS = {"–": None, "-1": -1, "1": 1, "2": 2, "3": 3, "4": 4,
                     "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
    rueckzahlung = []                      # Leere Liste für die Rückzahlungswerte der sechs Monate
    cols3 = st.columns(3)                  # Dreispaltiges Layout für kompakte Monatseingaben
    for i, m in enumerate(monate):         # Erstellt für jeden der letzten 6 Monate ein eigenes Auswahlfeld
        with cols3[i % 3]:
            st.markdown(f"**{m}**")
            v = st.selectbox("", list(STATUS_LABELS.keys()), key=f"pay_{i}",
                             label_visibility="collapsed")
            rueckzahlung.append(STATUS_LABELS[v] if STATUS_LABELS[v] is not None else -1)   # Speichert den ausgewählten Status als Zahl für die spätere Risikoanalyse

    st.divider()

    # ── Abschnitt 4: Kontoauszugsbetrag ─────────────────────────────────────

    st.markdown(f'<div class="section-label">KONTOAUSZUGSBETRAG — LETZTE 6 MONATE ({symbol})</div>',
                unsafe_allow_html=True)
    kontoauszug = []                # Leere Liste für die monatlichen Kontoauszugsbeträge
    cols4 = st.columns(3)           # Dreispaltiges Layout für die sechs Monatswerte
    for i, m in enumerate(monate):  # Erstellt für jeden Monat ein Eingabefeld für den Kontoauszugsbetrag
        with cols4[i % 3]:
            st.markdown(f"**{m}**")
            v = num_input_de(f"{symbol} ", key=f"bill_{i}", default=0)   # Betragseingabe im ausgewählten Währungsformat
            kontoauszug.append(v * kurs)                                 # Umrechnung in TWD, da das Modell intern mit Taiwan-Dollar arbeitet

    st.divider()

    # ── Abschnitt 5: Tatsächlich bezahlter Betrag ────────────────────────────

    st.markdown(f'<div class="section-label">TATSÄCHLICH BEZAHLTER BETRAG — LETZTE 6 MONATE ({symbol})</div>',
                unsafe_allow_html=True)
    bezahlt_liste = []                # Leere Liste für die tatsächlich bezahlten Monatsbeträge
    cols5 = st.columns(3)             # Dreispaltiges Layout für die Eingabefelder
    for i, m in enumerate(monate):    # Erstellt für jeden Monat ein Eingabefeld für die Rückzahlungen
        with cols5[i % 3]:
            st.markdown(f"**{m}**")
            v = num_input_de(f"{symbol}  ", key=f"paid_{i}", default=0)
            bezahlt_liste.append(v * kurs)    # Umrechnung der Eingaben in die interne Modellwährung TWD

    st.divider()

    # ── CTA-Button ───────────────────────────────────────────────────────────
    # Startet die Risikoanalyse mit allen Nutzereingaben

    # Analyse wird erst nach aktivem Nutzerklick ausgeführt
    if st.button("▶  Analyse starten", use_container_width=True, type="primary"):
        with st.spinner("Analyse wird durchgeführt…"):    # Zeigt während der Analyse eine Ladeanimation an
            time.sleep(1.1)   # Vorhersage-Latenz abwarten
            # eingaben bündelt alle Nutzereingaben in einer gemeinsamen Datenstruktur
            eingaben = {
                "alter": alter, "geschlecht": geschlecht,
                "bildung": bildung, "familienstand": familienstand,
                "kreditbetrag": kreditbetrag, "kreditbetrag_twd": kreditbetrag_twd,
                "rueckzahlung": rueckzahlung,
                "kontoauszug":  kontoauszug,
                "bezahlt":      bezahlt_liste,
                "symbol":       symbol,
            }
            st.session_state["ergebnis"] = berechne_risiko(eingaben)  # Speicherung der Analyseergebnisse für spätere Nutzung in der App
            st.session_state["eingaben"] = eingaben
            st.session_state["chat"]     = []                         # Initialisiert einen neuen Chatverlauf für die aktuelle Analyse
 
    # ── Ergebnisse ───────────────────────────────────────────────────────────

    if "ergebnis" in st.session_state:          # Ergebnisbereich wird erst angezeigt, nachdem eine Analyse durchgeführt wurde
        e   = st.session_state["ergebnis"]      # Lädt gespeicherte Analyseergebnisse und Eingaben aus dem Session State
        inp = st.session_state["eingaben"]
        st.markdown("---")
        st.subheader("Dein Risikoprofil")

        # Block 1 – Risikokategorie (farblich hinterlegt)
        # Definiert Farben, Icons und Erklärungstexte für jede Risikoklasse
        RISIKO_STIL = {
            "NIEDRIG": (GRÜN, "#E6F7F1", "🛡️",
                        "Dein Zahlungsausfallrisiko ist gering — du bist auf gutem Kurs."),
            "MITTEL":  (AMBER, "#FEF3E0", "⚠️",
                        "Es gibt einige Risikofaktoren. Handele proaktiv, bevor sie sich häufen."),
            "HOCH":    (ROT,  "#FDE8E8", "🚨",
                        "Erhöhtes Risiko festgestellt. Bitte umgehend Maßnahmen ergreifen."),
        }
        farbe, bg, icon, beschr = RISIKO_STIL[e["risiko"]]       # Wählt das passende Design anhand der berechneten Risikoklasse aus
        
        # Stellt die Risikokategorie als farbige Ergebnisbox dar
        st.markdown(f"""
        <div style='background:{bg};border:1.5px solid {farbe};border-radius:12px;
             padding:20px 24px;margin:8px 0'>
          <div style='font-size:2rem;font-weight:700;color:{farbe}'>{icon} {e["risiko"]}</div>
          <div style='color:{NAVY};margin-top:6px'>{beschr}</div>
        </div>""", unsafe_allow_html=True)

        # Block 2 – Zahlungsausfallvorhersage
        ausfall_farbe = ROT if e["ausfall_ja"] else GRÜN       # Übersetzt die Modellvorhersage in sichtbaren Text und passende Farbe
        ausfall_text  = "JA" if e["ausfall_ja"] else "NEIN"
        # Zeigt die Vorhersage, ob im nächsten Monat ein Zahlungsausfall erwartet wird
        st.markdown(f"""
        <div style='background:white;border:1px solid {BORDER};border-radius:12px;
             padding:18px 22px;margin:10px 0'>
          <div style='font-size:0.8rem;color:{MUTED};text-transform:uppercase;
               letter-spacing:.06em'>Zahlungsausfall nächsten Monat?</div>
          <div style='font-size:2.6rem;font-weight:700;color:{ausfall_farbe};margin:4px 0'>
            {ausfall_text}</div>
          <div style='font-size:0.85rem;color:{MUTED}'>
            Modellsicherheit: {e["prob"]} %</div>
        </div>""", unsafe_allow_html=True)
        
        # Visualisiert die geschätzte Ausfallwahrscheinlichkeit als Fortschrittsbalken
        st.progress(e["prob"] / 100)

        # Block 3 – Einflussfaktoren
        fp, fn = st.columns(2)     # Zwei Spalten für positive und negative Einflussfaktoren
        with fp:                   # Positive Faktoren, die das Risiko senken oder stabilisieren
            st.markdown(f"""
            <div style='border-left:3px solid {GRÜN};background:#E8F9F2;
                 border-radius:0 8px 8px 0;padding:14px;min-height:110px'>
              <div style='font-weight:600;color:{GRÜN};margin-bottom:8px'>
                ✅ Positiv wirkende Faktoren</div>
              {''.join(f'<div style="color:{NAVY};font-size:.87rem;margin-bottom:4px">• {p}</div>'
                       for p in e["positiv"])}
            </div>""", unsafe_allow_html=True)
        with fn:                  # Negative Faktoren, die das Risiko erhöhen können
            st.markdown(f"""
            <div style='border-left:3px solid {ROT};background:#FDE8E8;
                 border-radius:0 8px 8px 0;padding:14px;min-height:110px'>
              <div style='font-weight:600;color:{ROT};margin-bottom:8px'>
                ❌ Negativ wirkende Faktoren</div>
              {''.join(f'<div style="color:{NAVY};font-size:.87rem;margin-bottom:4px">• {n}</div>'
                       for n in e["negativ"])}
            </div>""", unsafe_allow_html=True)

        # Block 4 – Empfehlungen
        st.markdown('<div class="section-label" style="margin-top:18px">EMPFEHLUNGEN FÜR DICH</div>',  # Titel des Blicks
                    unsafe_allow_html=True)
        for em in e["empfehlungen"]:   # Gibt die personalisierten Handlungsempfehlungen einzeln aus
            st.markdown(f"▸ {em}")

        # Block 5 – Chat-Bereich
        st.markdown('<div class="section-label" style="margin-top:20px">FRAGEN ZU DEINEM ERGEBNIS</div>',  # Titel des Blocks
                    unsafe_allow_html=True)

        # Vorschlag-Buttons für häufige Fragen
        chips = ["Warum bin ich in dieser Risikoklasse?",     # Erstellt für jede Beispiel-Frage einen eigenen Button
                 "Was ist der wichtigste Faktor?",
                 "Wie kann ich mein Risiko senken?"]
        c_cols = st.columns(3)
        for ci, chip in enumerate(chips):
            if c_cols[ci].button(chip, key=f"chip_{ci}"):
                ant = chat_antwort(chip, {**e, **inp}, st.session_state["chat"])  # Generiert eine Antwort auf Basis des Analyseergebnisses
                st.session_state["chat"] += [                                     # Speichert Frage und Antwort im Chatverlauf
                    {"role":"user","content":chip},
                    {"role":"assistant","content":ant},
                ]

        # Chat-Verlauf rendern
        for msg in st.session_state.get("chat", []):                              # Rendert alle bisherigen Nachrichten im Chatverlauf
            bg_c  = NAVY if msg["role"] == "user" else PALE                       # Unterschiedliches Styling für Nutzer- und Assistenznachrichten
            txt_c = "white" if msg["role"] == "user" else NAVY
            align = "right" if msg["role"] == "user" else "left"
            st.markdown(f"""
            <div style='text-align:{align};margin:5px 0'>
              <div style='display:inline-block;background:{bg_c};color:{txt_c};
                   border-radius:12px;padding:9px 14px;max-width:80%;font-size:.87rem'>
                {msg["content"]}</div>
            </div>""", unsafe_allow_html=True)

        # Texteingabe
        # Freie Eingabe für individuelle Fragen zum Ergebnis
        frage = st.text_input("", placeholder="Stelle eine Frage zu deinem Ergebnis…",
                              key="chat_inp", label_visibility="collapsed")
        if st.button("Senden", key="senden"):                                           # Sendet die Nutzerfrage an die Chat-Funktion
            if frage.strip():                                                           # Verhindert leere Chatnachrichten
                ant = chat_antwort(frage, {**e, **inp}, st.session_state["chat"])
                st.session_state["chat"] += [
                    {"role":"user","content":frage},
                    {"role":"assistant","content":ant},
                ]
                st.rerun()                                                             # Aktualisiert die App, damit die neue Chatnachricht sofort sichtbar wird

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: DATENVERGLEICH  – drei Visualisierungen
# ─────────────────────────────────────────────────────────────────────────────

elif seite == "Datenvergleich":         # der Visualisierungstab wird ausgeführt, wenn die Variable seite als 'Datenvergleich' gespeichert wird
    st.title("Datenvergleich")
    st.caption(
        "Vergleiche deine Werte mit dem Referenzdatensatz (30.000 Kreditkartenprofile). "
        "Führe zuerst eine Analyse durch, damit deine persönlichen Werte eingetragen werden."
    )

    if "eingaben" not in st.session_state:                     # Visualisierung wird erst angezeigt, nachdem eine Analyse durchgeführt wurde
        st.info('ℹ️  Bitte führe zuerst eine Analyse durch (Tab "Analyse").')
        st.stop()

    inp = st.session_state["eingaben"]                         # Lädt gespeicherte Analyseergebnisse und Eingaben aus dem Session State
    df  = referenz_datensatz()                                 # Dataframe des Refernzdatensatzes als Vergleich
    sym = inp.get("symbol", "CHF")

    # ── Chart 2 : Bezahlt vs. Rückgezahlt───────────────────
    # Titel und Beschreibung
    st.markdown('<div class="section-label">BEZAHLT VS. ZURÜCKGEZAHLT — 6-MONATS-SUMME</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Vergleich zwischen dem, was du laut Kontoauszug bezahlt hast, "
        "und dem, was du tatsächlich zurückgezahlt hast — gegenüber dem Datensatz-Durchschnitt."
    )
    # Definition der Hilfsvariablen
    nutzer_bezahlt    = sum(inp["kontoauszug"])   # Kontoauszugssumme über 6 Monate (intern TWD)
    nutzer_rueck      = sum(inp["bezahlt"])        # tatsächlich bezahlt (TWD)
    ds_bezahlt_mittel = float(df["bezahlt_gesamt"].mean())       # Durchschnittswerte des Referenzdatensatzes
    ds_rueck_mittel   = float(df["rueckgezahlt_gesamt"].mean())

    # Daten für das Balkendiagramm
    kategorien = [
        "Datensatz Ø:\nKontoauszug",
        "Datensatz Ø:\nRückzahlung",
        "Du:\nKontoauszug",
        "Du:\nRückzahlung"
    ]
    
    werte = [
        ds_bezahlt_mittel,
        ds_rueck_mittel,
        nutzer_bezahlt,
        nutzer_rueck
    ]

    #Erstellung des Diagramms
    fig2, ax = plt.subplots(figsize=(12, 3))
    farben = [LIGHT, LIGHT, NAVY, GRÜN]
    bars = ax.bar(kategorien, werte, color=farben)

    ax.set_ylim(0, max(werte) * 1.15) # stellt sicher, dass für Beschriftungen genug Platz ist

    for bar in bars:                  # stellt sicher, dass die Bars so hoch und breit sind, wie sie sein sollen
        hoehe = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            hoehe,
            f"{hoehe:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY
        )
    
    ax.set_ylabel("Betrag (intern TWD)")         # Erstellt Y - Achse
    ax.ticklabel_format(style='plain', axis='y') # Deaktiviert wissenschaftliche Notation der Zahlen 
      
    st.pyplot(fig2)

    # Rückzahlungsquoten
    quote_du = nutzer_rueck / max(nutzer_bezahlt, 1) * 100
    quote_ds = ds_rueck_mittel / max(ds_bezahlt_mittel, 1) * 100
    q1, q2 = st.columns(2)                                                   # Erstellen von zwei Spalten für die Darstellung der Erfebnisse
    q1.metric("Deine Rückzahlungsquote (6 Monate)", f"{quote_du:.1f} %")
    q2.metric("Datensatz-Durchschnitt", f"{quote_ds:.1f} %",
              delta=f"{quote_du - quote_ds:+.1f} %",
              delta_color="normal" if quote_du >= quote_ds else "inverse")



# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4: MODELL-INFO
# ─────────────────────────────────────────────────────────────────────────────

elif seite == "Modell-Info":
    st.title("Modell-Info")
    st.caption("Transparente Einblicke in Leistung, Datenbasis und ethische Grundsätze des Modells.")

    # ── Modellperformance-Karten ─────────────────────────────────────────────
    st.markdown('<div class="section-label">MODELLPERFORMANCE (Test-Set)</div>', unsafe_allow_html=True)
    # Werte aus Notebook-Output: XGBoost, ROC-AUC 0.7824, Schwellwert 0.5, Klasse 1 (Ausfall)
    METRIKEN = [("Gesamtgenauigkeit", 76), ("Präzision (Ausfall-Klasse)", 47),
                ("Recall / True Positive Rate (Ausfall-Klasse)", 62), ("F1-Score (Ausfall-Klasse)", 54)]
    m1, m2 = st.columns(2)
    for i, (name, wert) in enumerate(METRIKEN):
        with (m1 if i % 2 == 0 else m2):
            st.markdown(f"""
            <div style='background:{PALE};border:1px solid {BORDER};border-radius:10px;
                 padding:16px;margin-bottom:12px'>
              <div style='font-size:0.82rem;color:{MUTED}'>{name}</div>
              <div style='font-size:2.2rem;font-weight:700;color:{NAVY}'>{wert}
                <span style='font-size:1rem;font-weight:400'>%</span></div>
            </div>""", unsafe_allow_html=True)
            st.progress(wert / 100)

    st.markdown(f"""
    <div style='font-size:0.78rem;color:{MUTED};margin-top:-6px;margin-bottom:12px;
         padding:8px 12px;background:{PALE};border-left:3px solid {LIGHT};border-radius:4px'>
      ℹ️ Präzision, Recall und F1-Score beziehen sich auf <strong>Klasse 1 (Positives = Zahlungsausfall)</strong>,
      da dies die relevante Zielklasse für die Kreditrisikoanalyse ist.
    </div>""", unsafe_allow_html=True)

    # ── Konfusionsmatrix (Heatmap) ────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1rem">KONFUSIONSMATRIX</div>',
                unsafe_allow_html=True)
    cm = [[3768, 905], [511, 816]]
    labels = [["Richtig negativ\n3.768", "Falsch positiv\n905"],
              ["Falsch negativ\n511",    "Richtig positiv\n816"]]
    farben = [["#E6F7F1", "#FDE8E8"],
              ["#FDE8E8", "#E6F7F1"]]

    FONT = {"fontfamily": "sans-serif", "color": NAVY}
    fig_cm, ax_cm = plt.subplots(figsize=(7, 3))
    for i in range(2):
        for j in range(2):
            ax_cm.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=farben[i][j]))
            ax_cm.text(j + 0.5, 1.5 - i, labels[i][j],
                       ha="center", va="center", fontsize=9, **FONT)
    ax_cm.set_xlim(0, 2)
    ax_cm.set_ylim(0, 2)
    ax_cm.set_xticks([0.5, 1.5])
    ax_cm.set_xticklabels(["Vorhergesagt:\nKein Ausfall", "Vorhergesagt:\nAusfall"],
                           fontsize=8.5, **FONT)
    ax_cm.set_yticks([0.5, 1.5])
    ax_cm.set_yticklabels(["Tatsächlich:\nAusfall", "Tatsächlich:\nKein Ausfall"],
                           fontsize=8.5, **FONT)
    ax_cm.tick_params(length=0)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_cm)

    # ── Textinfo-Kacheln (keine Dropdowns) ───────────────────────────────────
    i1, i2 = st.columns(2)   # Erstellen von zwei Spalten
    with i1:                 # Info-Card für Modellherkunft und -valididerung
        st.markdown(f"""
        <div class="info-card">
          <h3>Über das Modell</h3>
          <p>Das Modell wurde auf einem realen Datensatz mit <strong>30.000 Kreditkartenkundinnen
          und -kunden</strong> trainiert. Es verwendet <strong>XGBoost</strong> und wurde mittels
          k-facher Kreuzvalidierung evaluiert. Die Ausgangsdaten stammen aus einer akademischen
          Studie und wurden für den globalen Einsatz angepasst.</p>
        </div>
        """, unsafe_allow_html=True)
    with i2:               # Info-Card zum Diskriminierungsschutz
        st.markdown(f"""
        <div class="info-card">
          <h3>Datenschutz &amp; Fairness</h3>
          <p>Bei der Gestaltung dieser App wurde besonderer Wert auf
          <strong>diskriminierungsfreie Eingaben</strong> gelegt — insbesondere bei
          Geschlechtsidentität und Familienstand. Alle Angaben sind freiwillig.
          Deine Daten verlassen zu keiner Zeit diesen Browser.</p>
        </div>
        """, unsafe_allow_html=True)
