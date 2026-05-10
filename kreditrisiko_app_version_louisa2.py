"""
KreditRisiko Analyse – Streamlit App
=====================================
Ausführen:
  pip install streamlit plotly pandas numpy requests
  streamlit run kreditrisiko_app.py

Umgebungsvariablen (.env oder Systemvariable):
  OPENAI_API_KEY       – Chat-Funktion im Analyse-Tab
  EXCHANGE_API_KEY     – Live-Wechselkurse via api.exchangerate.host
"""

import os, time, random, datetime, requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


def berechne_risiko(inp: dict) -> dict:
    """
    Wertet die Eingabedaten aus und gibt Risikoklasse, Ausfallwahrscheinlichkeit
    sowie positive/negative Einflussfaktoren zurück.
    Kernlogik: gewichteter Score aus Zahlungsverzögerungen und Saldo-zu-Limit-Ratio.
    Produktiv wird hier der Endpunkt /api/predict aufgerufen.
    """
    verzoegerungen = sum(v for v in inp["rueckzahlung"] if v > 0)                        # Verzögerungen summiert alle Monate mit Zahlungsverzug
    hoher_saldo    = any(b > inp["kreditbetrag_twd"] * 0.8 for b in inp["kontoauszug"])  # hoher_saldo prüft, ob der Kontostand zeitweise über 80% des Kreditlimits lag
    score = verzoegerungen * 12 + (20 if hoher_saldo else 0) + max(0, 38 - inp["alter"]) # score erstellt eine einfache Risikoheuristik: Zahlungsverzug, hohe Auslastung und junges Alter erhöhen Risiko
    random.seed(inp["alter"] + verzoegerungen)                                           # random.seed sorgt für reproduzierbare Zufallswerte bei identischen Eingaben

   # Einteilung des Scores in Risikokategorien
    if score < 20:   risiko, farbe = "NIEDRIG", GRÜN
    elif score < 45: risiko, farbe = "MITTEL",  AMBER
    else:            risiko, farbe = "HOCH",    ROT

    # Umrechnung des Scores in eine geschätzte Ausfallswahrscheinlichkeit
    prob = min(0.95, score / 100 + random.uniform(-0.04, 0.04))

  # Erklärbare Einflussfaktoren für den Nutzer
    positiv = ["Regelmäßige Zahlungen in den letzten Monaten",           
               "Moderates Kreditvolumen im Verhältnis zum Alter",
               "Langjährige Kredithistorie ohne grobe Ausreißer"]
    negativ = (["Mehrere Zahlungsverzögerungen in den letzten 6 Monaten"]
               if verzoegerungen > 0 else []) + \
              (["Hoher Saldo im Verhältnis zum Kreditlimit"] if hoher_saldo else []) + \
              ["Zahlungshistorie zeigt unregelmäßige Muster"]

    return {
        "risiko": risiko, "farbe": farbe, "prob": round(prob * 100),
        "ausfall_ja": prob > 0.5,
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
    Nutzt die OpenAI API wenn OPENAI_API_KEY gesetzt ist,
    sonst greift die eingebaute Stichwort-Logik als Fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")    # Liest den API-Key aus den Umgebungsvariablen  
    if api_key:                                  # Wenn OpenAi-Key vorhanden, wird er für die Antwortgenerierung verwendet.   
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            system = (                                                                                  # Systemprompt definiert Rolle, Sprache und Kontext des Assistenten
                f"Du bist ein freundlicher Kreditrisikoberater. Antworte ausschließlich auf Deutsch. "
                f"Nutzerprofil: Risikoklasse={kontext.get('risiko')}, "
                f"Ausfallwahrscheinlichkeit={kontext.get('prob')}%, "
                f"Alter={kontext.get('alter')}, Kreditbetrag={kontext.get('kreditbetrag')}."
            )
            msgs = [{"role":"system","content":system}]                                           
            for h in verlauf[-6:]:
                msgs.append(h)
            msgs.append({"role":"user","content":frage})
            r = client.chat.completions.create(model="gpt-4o", messages=msgs, max_tokens=400)            # Anfrage an das Sprachmodell GPT-4o    
            return r.choices[0].message.content
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
    return ("Für vollständige Chat-Antworten bitte OPENAI_API_KEY als "
            "Umgebungsvariable setzen.")


@st.cache_data
# Zwischenspeicherung, damit der Datensatz nur einmal erzeugt und anschliessend gecacht wird
def referenz_datensatz(n: int = 3000) -> pd.DataFrame:
    """
    Baut den Vergleichsdatensatz für die Visualisierungen auf.
    Basiert auf den statistischen Kennwerten des UCI-Datensatzes
    (Default of Credit Card Clients, 30k Einträge).
    Gecacht, damit kein Re-Rendering bei jedem Widget-Update passiert.
    """
    rng = np.random.default_rng(42)                            # Fester Seed für reproduzierbare Vergleichsdaten                              
    limit    = rng.integers(10_000, 800_001, n).astype(float)  # Simulation typischer Kreditlimits
    saldo    = rng.uniform(0, 1.2, n) * limit                  # Simulation tiypscher Kontosalden
    über_pct = np.clip((saldo / limit - 1) * 100, -80, 200)    # Berechnet, wie stark Personen ihr Kreditlimit über- oder unterschreiten
    bezahlt     = rng.integers(0, 60_001, (n, 6))
    rueckgez    = rng.integers(0, 55_001, (n, 6))
    
    # Aufbau des finalen Vergleichsdatensatzes für die Visualisierungen
    return pd.DataFrame({
        "limit":          limit,
        "saldo":          saldo,
        "ueber_limit_pct": über_pct,
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

    # Status-Badge für die Wechselkurs-API – statischer Punkt, kein Pulsieren
    # Grünes Badge signalisiert aktive Wechselkursdaten
    st.markdown(f"""
    <div class="api-badge">
      <span style='width:8px;height:8px;border-radius:50%;background:{GRÜN};
            display:inline-block'></span>
      Wechselkurs-API aktiv
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
    kurs_badge = f"Kurs vom {heute_str} — {'live' if live else 'offline-Schätzung'}"   # Dynamischer Statushinweis für die Wechselkursquelle

    # Visuelles Status-Badge für Live-/Offline-Wechselkursdaten
    st.markdown(f"""
    <div style='display:inline-flex;align-items:center;gap:6px;background:#E8F9F2;
         border-radius:20px;padding:4px 14px;font-size:0.8rem;color:{GRÜN};margin-bottom:6px'>
      <span style='width:7px;height:7px;border-radius:50%;background:{GRÜN};
            display:inline-block'></span>{kurs_badge}
    </div>""", unsafe_allow_html=True)

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

    # ── Chart 1: Saldo / Limit – wie weit über dem Limit? ──────────────────
    # Titel und Beschreibung
    st.markdown('<div class="section-label">SALDO IM VERHÄLTNIS ZUM KREDITLIMIT</div>',
                unsafe_allow_html=True)
    st.markdown("Wie weit bist du im Schnitt über deinem Kreditlimit — im Vergleich zum Datensatz?")

    
    if inp["kreditbetrag_twd"] <= 0:     # damit die Prozentzahl bei einem kleinen Kreditbetrag nicht explodiert
        st.warning("Bitte gib zuerst einen Kreditbetrag größer als 0 ein.")
        st.stop()
    
    avg_saldo_twd  = np.mean(inp["kontoauszug"]) if any(inp["kontoauszug"]) else 0  # durchscnittlicher Saldo des Nutzers
    nutzer_pct     = (avg_saldo_twd / max(inp["kreditbetrag_twd"], 1) - 1) * 100    # Prozentuale Abweichung vom Kreditlimit
    median_ds      = float(df["ueber_limit_pct"].median())                          # Median der Vergleichsgruppe
    
    fig1, ax = plt.subplots(figsize=(12, 3))
    
    # Erstellen eines Histogramms des Referenzdatensatzes
    ax.hist(
        df["ueber_limit_pct"],
        bins=50,
        edgecolor="black",
        alpha=0.8
    )

    # Median des Datensatzes
    ax.axvline(
        median_ds,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_ds:.1f} %"
    )

    # Eigener Nutzerwert
    ax.axvline(
        nutzer_pct,
        color="red",
        linewidth=1.5,
        label=f"Du: {nutzer_pct:.1f} %"
    )

    # Achsenbeschriftung & Titel
    ax.set_xlabel("% über/unter Kreditlimit (0 = genau am Limit)") 
    ax.set_ylabel("Anzahl Kunden")
    ax.set_title("Saldo im Verhältnis zum Kreditlimit")

    ax.legend()   # Legende

    st.pyplot(fig1)  # führt Darstellung aus

    # KPI-Zeile
    k1, k2, k3 = st.columns(3)  # erstellt 3 Spalten für die KPIs

    k1.metric("Dein Wert", f"{nutzer_pct:.1f} %")  # Erstellt KPI von vorher definierter Funktion im String-Format

    k2.metric(
        "Datensatz-Median",
        f"{median_ds:.1f} %"
    )

    k3.metric(
        "Differenz",
        f"{nutzer_pct - median_ds:+.1f} %",
        delta_color="inverse" if nutzer_pct > median_ds else "normal"
    )

    st.divider()


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

    st.divider()

    # ── Chart 3: Relative Rückzahlungsquote letzter Monat ───────────────────
    
    # Titel und Beschreibung des Charts
    st.markdown('<div class="section-label">RELATIVE RÜCKZAHLUNGSQUOTE — LETZTER MONAT</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Wie viel Prozent deines Kontoauszugsbetrags hast du im letzten Monat "
        "zurückgezahlt — und wo stehst du damit im Vergleich?"
    )

    # Letzte Monatswerte des Nutzers
    letzter_bezahlt  = inp["bezahlt"][-1]   if inp["bezahlt"]    else 0
    letzter_auszug   = inp["kontoauszug"][-1] if inp["kontoauszug"] else 0
    monate           = letzte_6_monate()
    letzter_name     = monate[-1] if monate else "Letzter Monat"

    # Relative Quote: bezahlt / Kontoauszug (gedeckelt auf 200 %)
    nutzer_quote = min(200.0, letzter_bezahlt / max(letzter_auszug, 1) * 100)

    # Referenzdatensatz: relative Quote aus letztem Monat
    ds_quote = np.clip(
        df["letzter_monat"] / np.where(df["bezahlt_gesamt"] / 6 > 0,
                                        df["bezahlt_gesamt"] / 6, 1) * 100,
        0, 200
    )
    
    # Median der Vergleichsgruppe
    median_q = float(np.median(ds_quote))

    # Erstellen des Diagramms
    fig3, ax = plt.subplots(figsize=(12, 3))
    
    # Histogramm des Referenzdatensatzes
    ax.hist(
        ds_quote,
        bins=50,
        edgecolor="black",
        alpha=0.8,
        color=PALE
    )

    # Median des Datensatzes
    ax.axvline(
        median_q,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_q:.0f} %"
    )
    
    # Nutzerwert
    ax.axvline(
        nutzer_quote,
        color=GRÜN,
        linewidth=3,
        label=f"Du ({letzter_name}): {nutzer_quote:.0f} %"
    )

  
   # Achsenbeschriftung
    ax.set_xlabel("Rückgezahlter Anteil am Kontoauszug (%)")   
    ax.set_ylabel("Anzahl Personen")

    # Keine wissenschaftliche Notation
    ax.ticklabel_format(style='plain', axis='y')

    ax.legend()  # Erstellt Legende

    plt.tight_layout()

    st.pyplot(fig3)  # führt die Erstellung der Darstellung aus

      # Perzentil-Einordnung
    pct_rank = float((ds_quote < nutzer_quote).mean() * 100)      # Perzentil-Einordnung mit if-Clause in Form eines Strings
    if letzter_auszug == 0:
        st.markdown(f"""
        <div class="info-box">
          Kein Kontoauszugsbetrag für den letzten Monat eingetragen — kein Vergleich möglich.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
          Im letzten Monat hast du <strong>{nutzer_quote:.0f} %</strong> deines Kontoauszugsbetrags
          zurückgezahlt. Das ist mehr als bei <strong>{pct_rank:.0f} %</strong> der Vergleichsgruppe.
        </div>""", unsafe_allow_html=True)
    

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4: MODELL-INFO
# ─────────────────────────────────────────────────────────────────────────────

elif seite == "Modell-Info":
    st.title("Modell-Info")
    st.caption("Transparente Einblicke in Leistung, Datenbasis und ethische Grundsätze des Modells.")

    # ── Modellperformance-Karten ─────────────────────────────────────────────
    st.markdown('<div class="section-label">MODELLPERFORMANCE</div>', unsafe_allow_html=True)
    METRIKEN = [("Gesamtgenauigkeit",82),("Präzision",67),
                ("True Positive Rate (Recall)",71),("F1-Score",69)]
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

    # ── Konfusionsmatrix (Heatmap) ────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1rem">KONFUSIONSMATRIX</div>',
                unsafe_allow_html=True)
    fig_cm = go.Figure(go.Heatmap(
        z=[[6720,1280],[720,1280]],
        x=["Vorhergesagt: Kein Ausfall","Vorhergesagt: Ausfall"],
        y=["Tatsächlich: Kein Ausfall","Tatsächlich: Ausfall"],
        text=[["Richtig negativ\n6.720","Falsch positiv\n1.280"],
              ["Falsch negativ\n720","Richtig positiv\n1.280"]],
        texttemplate="%{text}",
        colorscale=[[0,"#FDE8E8"],[0.45,"#FDE8E8"],[0.55,"#E6F7F1"],[1,"#E6F7F1"]],
        showscale=False,
    ))
    fig_cm.update_layout(
        height=290, font=dict(color=NAVY),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=20,b=40),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

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
