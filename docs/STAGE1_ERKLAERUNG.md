#s Stage 1 erklärt – Schritt für Schritt, mit einem echten Beispiel

> Diese Erklärung gehört zum Notebook `notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb`.
> Die Abschnittsnamen in *kursiv* entsprechen den Überschriften im Notebook, damit du die Stellen dort wiederfindest.
> Die Code-Ausschnitte sind **vereinfacht**: Tabellen-Formatierung, Ausgaben und Plots sind weggelassen, damit man den Kern sieht. Der vollständige Code steht im Notebook.

---

## 0. Worum geht es in Stage 1?

**In einem Satz:** Stage 1 verwandelt 47'751 rohe Nachrichtenartikel über KI in (a) einen sauberen Datensatz, (b) eine Auswahl der Artikel zum Thema *AI Governance* und (c) einen **Wissensgraphen**, der zeigt, welche Akteure (Firmen, Länder, Gesetze, Behörden …) wie miteinander zusammenhängen.

Warum? Weil Stage 3 ein Frage-Antwort-System (RAG) baut, das Fragen wie *„Welche Behörden haben 2025 gegen OpenAI ermittelt?“* beantworten soll. Dafür braucht es saubere Texte **und** strukturiertes Wissen. Stage 1 liefert beides.

Die Pipeline im Überblick:

```
 Rohdaten (CSV, 47'751 Artikel)
   │
   ├─ 1. Laden & prüfen
   ├─ 2. Bereinigen          → kaputte Zeichen, Formatierung, Duplikate
   │                           = sauberer Gesamtkorpus (47'679 Artikel)      → Stage 2
   ├─ 3. Explorative Analyse → Governance-Artikel über Tags auswählen (12'946)
   │
   │   nur noch Governance-Artikel:
   ├─ 4. Entitäten erkennen (NER)      → „wer/was kommt vor?“
   ├─ 5. Co-Occurrence                 → „wer wird zusammen erwähnt?“
   ├─ 6. Knowledge Graph               → Netzwerk + Kennzahlen               → Stage 3
   ├─ 7. Relationsextraktion mit LLM   → „wie hängen sie zusammen?“          → Stage 3
   └─ 8. Topic Modeling                → „über welche Themen wird berichtet, und wie ändert sich das?“
```

### Unser Beispielartikel

Damit es greifbar wird, verfolgen wir **einen echten Artikel** aus dem Datensatz durch alle Schritte:

| Feld | Wert |
|---|---|
| `record_id` | 100583 |
| Titel | *Artificial Intelligence Disclosures Are Key to Customer Trust* |
| Datum | 24.09.2024 |
| Quelle (`domain`) | sloanreview.mit (MIT Sloan Management Review) |
| Tags | ResponsibleAI, AIAct, Dataiku, LargeLanguageModel, Accountability, Fairness, AutomationAnywhere, Locus, Robotics, Metaverse, EthicalAI |
| Länge | ca. 13'400 Zeichen |

Inhaltlich: Ein Expertenpanel diskutiert, ob Firmen offenlegen sollen, wann sie KI einsetzen – unter anderem mit Bezug auf den EU AI Act.

---

## 1. Laden und erste Prüfung
*Notebook: „1. Data Loading and Initial Inspection“, „2. Data Quality Assessment“*

Die CSV-Datei hat 7 Spalten: eine ID (`Unnamed: 0`), `title`, `date`, `content`, `domain`, `url`, `tags`.

Geprüft wird, ob die Daten brauchbar sind:

| Prüfung | Ergebnis |
|---|---|
| Fehlende Werte | keine |
| Doppelte IDs / URLs | keine |
| Identischer Text bei mehreren Artikeln | 66 Fälle → wird in Schritt 2 behandelt |

**Hauptcode:**

```python
# Projektordner finden (der Ordner, der .git enthält) und CSV laden
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai_media_dataset_20260911.csv"

df = pd.read_csv(DATA_PATH)

# Qualitätsprüfung
df.isna().sum().sum()                    # fehlende Werte
df["Unnamed: 0"].duplicated().sum()      # doppelte IDs
df["url"].duplicated().sum()             # doppelte URLs
df["content"].duplicated().sum()         # identischer Text
```

**Am Beispiel:** Die Spalte `content` ist kein normaler Text, sondern ein *Text, der wie eine Python-Liste aussieht*:

```
"['MIT SMR’ s fall 2024 issue highlights …', 'MIT SMR’ s fall 2024 issue highlights …', 'In collaboration with', 'For the third year in a row, … ( BCG) have assembled …']"
```

Man sieht schon zwei Probleme: `SMR’ s` (Leerzeichen nach dem Apostroph) und `( BCG)` (Leerzeichen in der Klammer).

> **Fazit Schritt 1:** Der Datensatz ist vollständig (keine fehlenden Werte, keine doppelten IDs oder URLs), aber **nicht direkt verwendbar**: Der Text ist als Listen-Text gespeichert, enthält Formatierungsfehler und 66 Artikel haben exakt denselben Text wie ein anderer. Das bestimmt die Aufgaben von Schritt 2.

---

## 2. Bereinigung
*Notebook: „Structured Text Fields“, „Systematic Encoding Inspection“, „Text Normalization“, „Removal of Duplicate Content“*

### 2a. Struktur reparieren
- `Unnamed: 0` wird zu `record_id` umbenannt.
- Das Datum wird von Text (`"2024-09-24"`) zu einem echten Datum, damit man nach Monaten gruppieren kann.
- `content` und `tags` werden von „Text, der wie eine Liste aussieht“ in **echte Listen** umgewandelt.
- Aus den Textabschnitten wird ein durchgehender Text `content_text` gebaut.

```python
df_clean = df.rename(columns={"Unnamed: 0": "record_id"})
df_clean["date"] = pd.to_datetime(df_clean["date"], format="%Y-%m-%d", errors="coerce")

def ensure_list(value):
    # "['a', 'b']" (Text) -> ['a', 'b'] (echte Liste)
    return ast.literal_eval(value) if isinstance(value, str) else value

df_clean["tags"] = df_clean["tags"].apply(ensure_list)
df_clean["content"] = df_clean["content"].apply(ensure_list)

# Textabschnitte zu einem durchgehenden Text verbinden
df_clean["content_text"] = df_clean["content"].apply(
    lambda parts: " ".join(str(part).strip() for part in parts)
)
```

### 2b. Kaputte Zeichen reparieren (Encoding / „Mojibake“)
Wenn Text in UTF-8 gespeichert, aber falsch eingelesen wurde, entstehen Zeichensalate. Beispiel aus dem Datensatz (Artikel 128500):

```
vorher:  US approves sale of Nvidia chips to UAEâ€™s G42
nachher: US approves sale of Nvidia chips to UAE’s G42
```

Das Notebook sucht solche Muster systematisch und repariert sie in drei Durchgängen: **49'305 kaputte Stellen → 21 übrig**. Unser Beispielartikel war davon nicht betroffen.

Der Trick: Man macht den Fehler rückgängig, indem man den Text wieder so in Bytes umwandelt, wie er falsch gelesen wurde (`latin1`/`cp1252`), und diese Bytes korrekt als UTF-8 liest.

```python
# Typische Mojibake-Muster: â + 1-2 Steuerzeichen, Ã + Zeichen, Â + Zeichen, ð + Zeichen (Emojis)
mojibake_pattern = re.compile(r"(?:â[\u0080-¿]{1,2}|Ã[\u0080-¿]|Â(?:[\u0080-¿]|\s)|ð[\u0080-¿]{2,3})")

def propose_encoding_repair(sequence):
    for encoding in ["latin1", "cp1252"]:
        try:
            repaired = sequence.encode(encoding).decode("utf-8")   # "â€™" -> "’"
            if repaired != sequence:
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return None

# Alle gefundenen kaputten Sequenzen einzeln ersetzen (längste zuerst)
automatic_repair_mapping = {s: propose_encoding_repair(s) for s in sequence_counts
                            if propose_encoding_repair(s) is not None}

def repair_sequences(text):
    for broken, repaired in sorted(automatic_repair_mapping.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(broken, repaired)
    return text
```

### 2c. Formatierung normalisieren
Nur eindeutige Fehler werden korrigiert – nichts, was die Bedeutung ändern könnte:

```
vorher:  MIT SMR’ s fall 2024 issue …   … Boston Consulting Group ( BCG) have …
nachher: MIT SMR’s fall 2024 issue …    … Boston Consulting Group (BCG) have …
```

```python
def normalize_text(text):
    text = re.sub(r"[ \t]{2,}", " ", text)      # mehrfache Leerzeichen -> eines
    text = re.sub(r"\(\s+", "(", text)          # "( BCG"  -> "(BCG"
    text = re.sub(r"\s+\)", ")", text)          # "BCG )"  -> "BCG)"
    return text

def normalize_apostrophes(text):
    # "SMR’ s" -> "SMR’s", "don’ t" -> "don’t" (nur bei typischen englischen Endungen)
    text = re.sub(r"(?<=[A-Za-z])[’']\s+(?=(?:s|t|re|ve|ll|d|m)\b)", lambda m: m.group(0)[0], text)
    # französische Elisionen: "l’ ère" -> "l’ère"
    text = re.sub(r"\b([cdjlmnst]|qu)([’'])\s+(?=[A-Za-zÀ-ÖØ-öø-ÿ])", r"\1\2", text, flags=re.IGNORECASE)
    return text
```

### 2d. Duplikate entfernen
Nach der Bereinigung haben **72 Artikel exakt denselben Text** wie ein früherer Artikel – meist dieselbe Quelle, die den Artikel doppelt erfasst oder später neu veröffentlicht hat. Beispiel:

| record_id | Datum | Quelle | Titel |
|---|---|---|---|
| 64238 | 05.11.2025 | deeplearning | OpenAI Reorgs For Profit, MiniMax-M2 Leads Open Coding, Universal Music Group Embraces AI, and more… |
| 64242 | 05.11.2025 | deeplearning | OpenAI Reorgs For Profit, MiniMax-M2 Leads Open Coding, Universal Music Group Emb *(abgeschnitten)* |

Würde man beide behalten, würde derselbe Artikel doppelt gezählt (z. B. im Wissensgraphen). Behalten wird jeweils **die früheste Veröffentlichung**.

```python
# Nach Datum sortieren, damit beim Duplikat-Check die früheste Version "gewinnt"
first_publication_order = df_clean.sort_values(["date", "record_id"]).index

is_duplicate_content = (
    df_clean.loc[first_publication_order, "content_text"]
    .duplicated(keep="first")          # alle späteren Kopien = True
)

duplicate_records = df_clean.loc[is_duplicate_content[is_duplicate_content].index]
df_clean = df_clean.drop(index=duplicate_records.index).reset_index(drop=True)

# Bereinigten Korpus speichern (Grundlage für Stage 2)
df_clean.drop(columns=["content"]).to_pickle(CORPUS_DIR / "ai_media_clean.pkl.gz", compression="gzip")
```

**Ergebnis von Schritt 2:** 47'679 saubere, eindeutige Artikel → gespeichert als `data/processed/corpus/ai_media_clean.pkl.gz` (Grundlage für Stage 2).

> **Fazit Schritt 2:**
> - Das grösste Qualitätsproblem war die **Zeichenkodierung**: fast 50'000 kaputte Stellen, vor allem Apostrophe und Anführungszeichen (`â€™` → `’`). Nach der Reparatur bleiben nur 21 Stellen in 14 Artikeln übrig – sie wurden bewusst nicht „geraten“, sondern belassen.
> - **Warum 72 statt 66 Duplikate?** Vor der Bereinigung unterscheiden sich manche Kopien nur durch kaputte Zeichen oder Leerzeichen. Erst nach der Reparatur sind sie wirklich identisch. Deshalb wird der Duplikat-Check bewusst **nach** der Bereinigung gemacht.
> - Die Duplikate sind fast immer dieselbe Quelle (65 von 67 Gruppen) – also echte Doppelerfassungen, keine verschiedenen Artikel. Das Entfernen ist damit gut begründet.

---

## 3. Explorative Analyse und Auswahl der Governance-Artikel
*Notebook: „3. Exploratory Data Analysis“, „AI Governance Tag Exploration“, „Final Governance Theme Definition“*

### 3a. Überblick
Zuerst wird der ganze Datensatz beschrieben: Artikel pro Monat, häufigste Quellen, häufigste Tags. Das zeigt z. B., dass September 2024 und September 2026 nur Teilmonate sind und vorsichtig interpretiert werden müssen.

### 3b. Was ist ein „Governance-Artikel“?
Jeder Artikel hat im Schnitt 5 Tags (insgesamt 1'311 verschiedene). Alle 1'311 Tags wurden von Hand durchgesehen und **30 davon** einem der vier Governance-Themen zugeordnet:

| Thema | Beispiel-Tags |
|---|---|
| Security | AISafety, Deepfake, RedTeaming, Surveillance, Disinformation |
| Accountability | Accountability, ResponsibleAI, EthicalAI, ExplainableAI, HumanInTheLoop |
| Bias / Fairness | Fairness, AlgorithmicBias |
| Regulation | AIAct, AIPolicy, Legal, SocialScoring |

**Regel:** Ein Artikel ist Governance-relevant, wenn er **mindestens einen** dieser Tags hat.

**Am Beispiel:** Unser Artikel hat die Tags `ResponsibleAI`, `Accountability`, `EthicalAI` (→ Accountability), `Fairness` (→ Bias/Fairness) und `AIAct` (→ Regulation). Er gehört also zu **3 von 4 Themen**:

| security | accountability | bias_fairness | regulation | governance_related |
|---|---|---|---|---|
| False | **True** | **True** | **True** | **True** |

Tags wie `Robotics` oder `Metaverse` zählen nicht – sie sind nicht spezifisch für Governance.

```python
governance_tag_mapping = {
    "Security":        ["Surveillance", "Disinformation", "Deepfake", "AISafety", "RedTeaming", ...],
    "Accountability":  ["Accountability", "EthicalAI", "ExplainableAI", "ResponsibleAI", ...],
    "Bias / Fairness": ["Fairness", "AlgorithmicBias"],
    "Regulation":      ["AIAct", "AIPolicy", "Legal", "SocialScoring"]
}

# Pro Thema eine True/False-Spalte: hat der Artikel mindestens einen Tag des Themas?
for theme, theme_tags in governance_tag_mapping.items():
    column = theme.lower().replace(" / ", "_").replace(" ", "_")     # z. B. "bias_fairness"
    df_clean[column] = df_clean["tags"].apply(lambda tags: any(tag in theme_tags for tag in tags))

governance_columns = ["security", "accountability", "bias_fairness", "regulation"]
df_clean["governance_related"] = df_clean[governance_columns].any(axis=1)     # mind. 1 Thema
df_clean["governance_theme_count"] = df_clean[governance_columns].sum(axis=1)  # wie viele Themen
```

**Ergebnis:** 12'946 von 47'679 Artikeln (27 %) sind Governance-relevant. Ab Schritt 4 wird nur noch mit diesen gearbeitet.

### 3c. Entwicklung über die Zeit
Das Notebook zählt pro Monat, wie viele Artikel zu jedem Thema erscheinen – **absolut und als Anteil**. Der Anteil ist wichtig: Wenn im Mai einfach doppelt so viele Artikel gesammelt wurden, steigen alle absoluten Zahlen, ohne dass ein Thema wirklich wichtiger wurde. Erst der Anteil zeigt echte Trends – z. B. steigt der Anteil von Accountability von ca. 10 % (Ende 2024) auf bis zu ca. 25 % (Juni 2026).

```python
# Anzahl Artikel pro Thema und Monat ("ME" = Monatsende)
monthly_governance = df_clean.set_index("date")[governance_columns].resample("ME").sum()

# ... geteilt durch alle Artikel des Monats = Anteil in %
monthly_total_articles = df_clean.set_index("date").resample("ME").size()
monthly_governance_share = monthly_governance.div(monthly_total_articles, axis=0) * 100
```

> **Fazit Schritt 3:**
> - **Gut ein Viertel** des Korpus (12'946 Artikel, 27 %) ist Governance-relevant.
> - **Accountability dominiert** klar (7'592 Artikel), dann Security (4'455), Bias/Fairness (2'662) und Regulation (1'370). Überraschend: Das explizite Thema *Regulation* ist am seltensten – über KI-Gesetze wird weniger berichtet als über Verantwortung und Sicherheit.
> - 80 % der Governance-Artikel betreffen **nur ein** Thema; etwa jeder fünfte verbindet mehrere (wie unser Beispielartikel mit drei).
> - **Accountability wird wichtiger:** Der Anteil steigt von ca. 10–13 % (Ende 2024) auf bis zu 24.6 % (Juni 2026). Das ist nicht nur ein Effekt von mehr Artikeln insgesamt – der Anteil steigt auch relativ.
> - Die Quellen sind breit verteilt (392 Quellen, die 15 grössten liefern nur 31 % der Artikel) – die Ergebnisse hängen also nicht an einzelnen Medien.

---

## 4. Entitäten erkennen (Named Entity Recognition, NER)
*Notebook: „Entity Extraction“, „Governance-Specific Entity Patterns“, „Entity Normalization“*

### 4a. Was ist NER?
Ein Sprachmodell (hier **spaCy**, Modell `en_core_web_sm`) liest den Text und markiert Namen mit einem Typ:

```
… the European Union’s AI Act …  Mark Surman of the Mozilla Foundation …
    └────── ORG ───────┘ └LAW─┘   └─PERSON──┘        └──── ORG ─────┘
```

Behalten werden die Typen ORG (Organisation), PERSON, GPE (Land/Stadt), LAW (Gesetz), PRODUCT – plus zwei eigene Typen (siehe 4b). Pro Artikel wird jede Entität nur **einmal** gezählt.

**Am Beispiel** (Auszug der 65 erkannten Entitäten):

| erkannt | Typ von spaCy | Bewertung |
|---|---|---|
| MIT Sloan Management Review | ORG | ✅ korrekt |
| Boston Consulting Group | ORG | ✅ korrekt |
| Mark Surman | PERSON | ✅ korrekt |
| AI Act | LAW | ✅ korrekt |
| AI | **GPE** (Land!) | ❌ falsch |
| Artificial Intelligence Disclosures Are Key | **ORG** | ❌ falsch – das ist der Titel |
| Dataiku’s | **PERSON** | ❌ falsch – Dataiku ist eine Firma |

Das zeigt: Ein allgemeines Modell macht bei Fachbegriffen viele Fehler. Deshalb gibt es zwei Korrekturschritte.

```python
nlp = spacy.load("en_core_web_sm")
ENTITY_TYPES = {"ORG", "PERSON", "GPE", "LAW", "PRODUCT", "STANDARD", "REGULATOR"}

def texts():
    for row in governance_df.itertuples():
        yield f"{row.title}. {row.content_text}", row.record_id

entity_results = {}
# nlp.pipe verarbeitet viele Texte effizient; Parser/Tagger werden für NER nicht gebraucht
for doc, record_id in nlp.pipe(texts(), as_tuples=True, batch_size=4,
                               disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]):
    seen = set()
    for ent in doc.ents:
        entity = (" ".join(ent.text.split()), ent.label_)
        if ent.label_ in ENTITY_TYPES and entity not in seen:   # jede Entität 1x pro Artikel
            seen.add(entity)
    entity_results[record_id] = list(seen)
```

Weil das ca. 50 Minuten dauert, speichert das Notebook alle 250 Artikel einen Zwischenstand und verwendet beim nächsten Lauf das Ergebnis wieder („Cache“) – aber nur, wenn die Konfiguration (Modell, Typen, Regeln) gleich geblieben ist.

### 4b. Feste Regeln für Governance-Begriffe (EntityRuler)
Bevor spaCy rät, erkennt ein **Regelwerk** 25 wichtige Begriffe immer gleich:

| Typ | Begriffe (Auswahl) |
|---|---|
| LAW | EU AI Act, AI Act, GDPR, Digital Services Act, CCPA, Colorado AI Act |
| STANDARD *(neu)* | NIST AI RMF, ISO/IEC 42001, OECD AI Principles |
| REGULATOR *(neu)* | FTC, NIST, AI Office, European Commission |

Ohne diese Regeln wurde „AI Act“ im Datensatz mal als Organisation, mal als Person, mal als Produkt erkannt.

```python
GOVERNANCE_ENTITY_PATTERNS = {
    "LAW":       ["EU AI Act", "AI Act", "GDPR", "Digital Services Act", "CCPA", "Colorado AI Act", ...],
    "STANDARD":  ["NIST AI RMF", "AI RMF", "ISO/IEC 42001", "ISO 42001", "OECD AI Principles"],
    "REGULATOR": ["FTC", "Federal Trade Commission", "NIST", "AI Office", "European Commission", ...]
}

# Die Regeln laufen VOR dem statistischen NER; spaCy übernimmt diese Treffer und erkennt den Rest wie bisher
entity_ruler = nlp.add_pipe("entity_ruler", before="ner")
entity_ruler.add_patterns([
    {"label": label, "pattern": pattern}
    for label, patterns in GOVERNANCE_ENTITY_PATTERNS.items()
    for pattern in patterns
])
```

### 4c. Normalisierung
Gleiche Dinge mit verschiedenen Namen werden zusammengeführt, falsche Typen korrigiert:

| vorher | nachher |
|---|---|
| `AI` (GPE) | `Artificial Intelligence` (TECHNOLOGY) |
| `AI Act` (LAW) | `EU AI Act` (LAW) |
| `US`, `U.S.`, `the United States` | `United States` (GPE) |
| `OpenAI` (manchmal PERSON oder GPE) | `OpenAI` (ORG) |

Die Rohdaten bleiben trotzdem gespeichert, damit man jede Korrektur nachvollziehen kann.

```python
ENTITY_ALIASES = {"AI": "Artificial Intelligence", "US": "United States", "U.S.": "United States",
                  "AI Act": "EU AI Act", "ISO 42001": "ISO/IEC 42001", ...}
ENTITY_TYPE_OVERRIDES = {"OpenAI": "ORG", "Artificial Intelligence": "TECHNOLOGY",
                         "Claude": "MODEL", "EU AI Act": "LAW", "FTC": "REGULATOR", ...}

def normalize_entity_list(entity_list):
    normalized, seen = [], set()
    for text, entity_type in entity_list:
        text = ENTITY_ALIASES.get(text, text).strip()              # Schreibweise vereinheitlichen
        entity_type = ENTITY_TYPE_OVERRIDES.get(text, entity_type)  # bekannte Typ-Fehler korrigieren
        if (text, entity_type) not in seen:                          # nach dem Zusammenführen erneut deduplizieren
            normalized.append((text, entity_type))
            seen.add((text, entity_type))
    return normalized

governance_df["entities_normalized"] = governance_df["entities"].apply(normalize_entity_list)
```

> **Ehrliche Grenze:** Nicht alle Fehler werden korrigiert (z. B. bleibt „Dataiku’s“ als PERSON). Korrigiert werden nur häufige, eindeutige Fälle. Seltene Fehler fallen in Schritt 5 meist durch die Mindesthäufigkeit heraus.

> **Fazit Schritt 4:**
> - spaCy findet in den 12'946 Artikeln knapp **182'000 verschiedene Entitäten** – eine riesige Menge, die meisten davon selten.
> - Das allgemeine Modell ist bei KI-Begriffen unzuverlässig: „AI“ wurde über 10'000-mal als *Land* (GPE) erkannt, „OpenAI“ mal als Organisation, Person oder Land. Ohne Normalisierung wäre der Graph voller Duplikate und falscher Typen.
> - Die festen Regeln (EntityRuler) machen Governance-Begriffe erst sichtbar: Der **EU AI Act** taucht jetzt einheitlich als ein Gesetz in **790 Artikeln** auf (vorher verteilt auf verschiedene Schreibweisen und Typen); dazu kommen 8 Behörden und 6 Standards als eigene Typen.
> - Eine Entdeckung am Rand: Selbst nach der Korrektur bleiben mehrdeutige Namen wie *Washington* (Stadt, Regierung oder Person?) – solche Fälle hängen vom Kontext ab und werden bewusst nicht pauschal umgeschrieben.

---

## 5. Co-Occurrence – wer wird zusammen erwähnt?
*Notebook: „Relation Extraction and Co-occurrence Analysis“*

### Die Idee
Wenn zwei Entitäten **im selben Artikel** vorkommen, gibt es zwischen ihnen eine Verbindung. Je öfter das passiert, desto stärker die Verbindung.

**Mini-Beispiel:** Ein Artikel enthält die Entitäten *A, B, C*. Daraus entstehen die Paare *A–B*, *A–C*, *B–C*, und jedes Paar bekommt +1.

**Am Beispiel:** Unser Artikel enthält u. a. *Artificial Intelligence*, *EU AI Act* und *Boston Consulting Group*. Also bekommt z. B. das Paar *Artificial Intelligence – EU AI Act* +1. Über alle 12'946 Artikel summiert, kommt dieses Paar auf **777 gemeinsame Artikel**.

### Zwei Filter gegen „Rauschen“
1. **Entität muss in mind. 20 Artikeln vorkommen** – sonst bleiben Tausende Einzelerwähnungen (inkl. NER-Fehlern wie „Artificial Intelligence Disclosures Are Key“) übrig. Von über 180'000 Entitäten bleiben **2'317**.
2. **Paar muss in mind. 50 Artikeln gemeinsam vorkommen** – sonst ist das Netz so dicht, dass man nichts erkennt (über 600'000 Paare!). Übrig bleiben **4'282 Verbindungen zwischen 720 Entitäten**.

```python
from itertools import combinations
from collections import Counter

MIN_ENTITY_FREQUENCY = 20
graph_entities = set(
    normalized_entity_frequency.loc[normalized_entity_frequency["Article Frequency"] >= MIN_ENTITY_FREQUENCY,
                                    ["Entity", "Type"]].itertuples(index=False, name=None)
)

cooccurrence_counter = Counter()
for entities in governance_df["entities_normalized"]:
    retained = sorted(set(entities) & graph_entities)     # nur häufige Entitäten
    for entity_a, entity_b in combinations(retained, 2):  # alle Paare im Artikel
        cooccurrence_counter[(entity_a, entity_b)] += 1

MIN_COOCCURRENCE = 50
graph_edges = cooccurrence_df.loc[cooccurrence_df["Cooccurrence_Count"] >= MIN_COOCCURRENCE]
```

> **Wichtig:** Co-Occurrence sagt nur „wird zusammen erwähnt“ – nicht *warum*. Dass *China* und *United States* oft zusammen vorkommen, sagt nicht, ob es um Handel, Konflikt oder Kooperation geht. Genau dafür gibt es Schritt 7.

> **Fazit Schritt 5:**
> - Die stärksten Paare sind *Artificial Intelligence – United States* (3'775 gemeinsame Artikel), *– Google* (2'058) und *– OpenAI* (1'780); dazu politische Paare wie *China – United States* (1'450) und *Trump – United States* (1'355).
> - Die Governance-Diskussion verbindet also **Technologie, grosse US-Konzerne und Geopolitik** sehr eng miteinander.
> - Die Filter sind entscheidend: Ohne sie gäbe es über 600'000 Paare, mit ihnen bleibt ein lesbares Netz von 4'282 Verbindungen. Die vollständigen Daten bleiben gespeichert, falls man später andere Schwellen testen will.

---

## 6. Der Knowledge Graph
*Notebook: „Knowledge Graph Schema and Data Dictionary“, „Knowledge Graph Construction and Analysis“*

### 6a. Erst das Schema, dann der Graph
Bevor der Graph gebaut wird, legt ein **Schema** fest, welche Arten von Knoten und Kanten es geben darf (Knotentypen ORG, PERSON, LAW, STANDARD, REGULATOR …; Kantentypen `CO_MENTIONED_WITH`, `ASSOCIATED_WITH_THEME`, `REGULATES`, `FINES` …). Das Schema wird mit dem Graphen gespeichert, damit Stage 3 weiss, wie der Graph zu lesen ist.

### 6b. Knoten und Kanten
- **Knoten** = eine Entität, eindeutig benannt als `TYP::Name`, z. B. `LAW::EU AI Act`, `ORG::OpenAI`.
- **Kante** = eine Co-Occurrence-Verbindung `CO_MENTIONED_WITH` mit Gewicht, z. B.
  `TECHNOLOGY::Artificial Intelligence ── 777 ── LAW::EU AI Act`

### 6c. Kennzahlen – wer ist wichtig im Netz?

| Kennzahl | Intuition | Beispiel |
|---|---|---|
| **Degree** | Mit wie vielen anderen ist ein Knoten verbunden? | *Artificial Intelligence* ist mit 633 von 719 anderen verbunden |
| **PageRank** | Wichtig ist, wer mit *wichtigen* Knoten verbunden ist (wie Google-Suche) | Platz 1: Artificial Intelligence, Platz 2: United States, dann Trump, Google, China |
| **Clustering** | Sind die Nachbarn eines Knotens auch untereinander verbunden? | Sehr zentrale Knoten haben tiefe Werte – sie verbinden verschiedene „Welten“ |

```python
import networkx as nx

G = nx.Graph()
for row in graph_edges.itertuples(index=False):
    node_1, node_2 = f"{row.Type_1}::{row.Entity_1}", f"{row.Type_2}::{row.Entity_2}"
    G.add_node(node_1, name=row.Entity_1, entity_type=row.Type_1)
    G.add_node(node_2, name=row.Entity_2, entity_type=row.Type_2)
    G.add_edge(node_1, node_2, relation="CO_MENTIONED_WITH", weight=int(row.Cooccurrence_Count))

degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G, weight="weight")
clustering = nx.clustering(G, weight="weight")

nx.write_graphml(G, KG_DIR / "ai_governance_knowledge_graph.graphml")   # für Stage 3
```

### 6d. Anreicherung mit den Governance-Themen
Zusätzlich kommen die 4 Themen als eigene Knoten in den Graphen. Eine Kante `ASSOCIATED_WITH_THEME` zählt, in wie vielen Artikeln eines Themas eine Entität vorkommt, z. B.
`LAW::EU AI Act ── ASSOCIATED_WITH_THEME (787) ──▶ THEME::Regulation`.
Unser Beispielartikel trägt dazu je +1 für Accountability, Bias/Fairness und Regulation bei.

> **Fazit Schritt 6:**
> - Der Graph hat **720 Knoten und 4'282 Kanten** (angereichert: 724 Knoten, 7'139 Kanten).
> - *Artificial Intelligence* ist mit 633 von 719 anderen Knoten verbunden – erwartbar, weil es ein Oberbegriff ist. Interessanter sind die nächsten Plätze: **United States, Trump, Google, China, OpenAI, Washington, Microsoft**. AI Governance wird in den Medien also stark als **US-amerikanisches und geopolitisches** Thema verhandelt.
> - Die tiefen Clustering-Werte der zentralen Knoten zeigen, dass sie **Brücken** zwischen verschiedenen Bereichen sind (z. B. Technik ↔ Politik), nicht Teil einer abgeschlossenen Gruppe.
> - Bei den Themen-Kanten zeigt sich ein Muster: Technologie-Begriffe und Firmen hängen vor allem an **Accountability**, Länder und Politiker vor allem an **Security**.

---

## 7. Relationsextraktion mit einem Sprachmodell (LLM)
*Notebook: „LLM-Based Relation Extraction“*

### Warum?
Co-Occurrence sagt *dass*, nicht *wie*. Ein LLM kann Sätze verstehen und daraus **gerichtete, benannte Beziehungen** ableiten: *wer* tut *was* mit *wem*.

### Wie?
- Werkzeug: **LangExtract** (von Google, im Auftrag explizit genannt) mit dem Modell **Gemini 3.5 Flash-Lite**.
- Das Modell bekommt eine Anleitung, die erlaubten Beziehungstypen aus dem Schema (`REGULATES`, `INVESTIGATES`, `FINES`, `SUES`, `COMPLIES_WITH`, `ADOPTS`, `PROPOSES`, `DEVELOPS`, `PARTNERS_WITH`) und zwei Beispieltexte mit korrekten Lösungen („Few-Shot“).
- Wichtigste Regel: Das Modell muss die **genaue Textstelle** angeben, die die Beziehung belegt. LangExtract prüft, wo diese Stelle im Text steht („Grounding“).
- Aus Kosten- und Zeitgründen nicht alle 12'946 Artikel, sondern eine **Stichprobe von ca. 400 Artikeln**, die ein Gesetz, eine Behörde oder einen Standard erwähnen – gleichmässig verteilt über alle 24 Monate. Nur Titel + erste 3'000 Zeichen.
- Alle Antworten werden zwischengespeichert: Ein erneuter Lauf ruft die API nicht noch einmal auf.

```python
import langextract as lx

RELATION_PROMPT = """Extract explicit AI-governance relations between named entities from the article.
Only extract a relation if the text states it directly. Use one of these predicates: REGULATES, INVESTIGATES, ...
For extraction_text, copy the exact words from the article that state the relation (no paraphrasing). ..."""

# Few-Shot-Beispiel: Text + korrekte Lösung
RELATION_EXAMPLES = [lx.data.ExampleData(
    text="The European Commission fined Meta 200 million euros under the Digital Markets Act. ...",
    extractions=[lx.data.Extraction(
        extraction_class="relation",
        extraction_text="The European Commission fined Meta",
        attributes={"subject": "European Commission", "subject_type": "REGULATOR",
                    "predicate": "FINES", "object": "Meta", "object_type": "ORG"}), ...])]

result = lx.extract(
    text_or_documents=f"{row.title}. {row.content_text}"[:3000],   # Titel + Artikelanfang
    prompt_description=RELATION_PROMPT,
    examples=RELATION_EXAMPLES,
    model_id="gemini-3.5-flash-lite",
    api_key=GEMINI_API_KEY,                                       # aus der lokalen .env-Datei
)

for extraction in result.extractions:
    extraction.attributes      # subject, predicate, object, Typen
    extraction.char_interval   # wo im Text steht der Beleg? (Grounding)
```

Danach werden Subjekt und Objekt mit denselben Regeln wie in Schritt 4c normalisiert, alles, was nicht dem Schema entspricht, wird verworfen, und die Relationen kommen in einen gerichteten Graphen (`nx.MultiDiGraph`) mit `record_id`, `date` und `evidence` an jeder Kante.

### Am Beispiel
Das Modell liefert für unseren Artikel:

```json
{
  "subject": "European Union", "subject_type": "ORG",
  "predicate": "ADOPTS",
  "object": "EU AI Act", "object_type": "LAW",
  "evidence": "the European Union’s AI Act",
  "position": Zeichen 830–857,  "record_id": 100583,  "date": "2024-09-24"
}
```

Im Graphen: `ORG::European Union ──ADOPTS──▶ LAW::EU AI Act` – mit Quelle und Datum. Damit kann Stage 3 später antworten *und belegen*: „laut Artikel 100583 vom 24.09.2024 …“.

> **Ehrliche Grenze – gut an diesem Beispiel zu sehen:** Der Beleg „the European Union’s AI Act“ sagt eigentlich nur, dass es das Gesetz der EU ist, nicht ausdrücklich, dass die EU es *verabschiedet* hat. Die Relation ist inhaltlich richtig, aber vom Modell **abgeleitet** und nicht wörtlich im Text belegt. LLM-Extraktionen sind deshalb nützlich, aber nicht fehlerfrei – das wird im Notebook bei den Limitations festgehalten und sollte in Stage 3 bei der Evaluation berücksichtigt werden.

### Ergebnis über alle 408 Artikel

| Kennzahl | Wert |
|---|---|
| Artikel verarbeitet | 408 (17 pro Monat) |
| vom Sicherheitsfilter blockiert | 1 |
| Artikel mit mind. einer Relation | 104 (25 %) |
| Relationen, die dem Schema entsprechen | 162 |
| davon an einer Textstelle verankert | 100 % |

Häufigste Typen: `DEVELOPS` 79, `REGULATES` 33, `PARTNERS_WITH` 16, `PROPOSES` 14 – aber nur 3× `INVESTIGATES` und 1× `FINES`.

**Qualitätskontrolle** – 40 zufällig gezogene Relationen von Hand geprüft:

| Bewertung | Anzahl | Beispiel |
|---|---|---|
| ✅ korrekt und belegt | 29 (73 %) | `FTC –INVESTIGATES→ companies` („An FTC investigation found that companies track…“) |
| 🟡 richtig, aber schwach belegt | 6 (15 %) | `European Union –ADOPTS→ EU AI Act` aus „including the EU AI Act“ |
| ❌ falsch | 5 (12 %) | `California –REGULATES→ AI Transparency Act` (richtig wäre ADOPTS) |

### Was in der Praxis passiert ist (und was man daraus lernt)
- **Sicherheitsfilter:** Gemini hat den Artikel *„Grok's Deepfake Porn Has Put X in Deep Trouble“* abgelehnt (`PROHIBITED_CONTENT`). Die erste Version des Codes hat das wie einen vorübergehenden Fehler behandelt, fünfmal wiederholt und dann die ganze Extraktion gestoppt. Lösung: Blockierte Artikel werden jetzt als „blocked“ markiert und übersprungen; abgebrochen wird nur, wenn mehrere Artikel hintereinander scheitern. **Lehre:** Bei LLM-APIs muss man zwischen *dauerhaften* und *vorübergehenden* Fehlern unterscheiden.
- **Kosten und Reproduzierbarkeit:** Weil alle Antworten im Cache liegen, kostet ein erneuter Lauf nichts und liefert exakt dieselben Relationen – auch für Teammitglieder ohne API-Key.

> **Fazit Schritt 7:**
> - Das LLM liefert, was Co-Occurrence nicht kann: **gerichtete, benannte und datierte Fakten mit Beleg** – ideal für Stage 3.
> - Die Medien berichten in Artikelanfängen vor allem über **neue Modelle und Produkte** (`DEVELOPS`) und **neue Regeln** (`REGULATES`, `PROPOSES`) – konkrete Durchsetzung (Bussen, Ermittlungen, Klagen) kommt selten vor.
> - Nur 23 % der Entitäten aus den Relationen kommen auch im Co-Occurrence-Graphen vor: Die Relationen betreffen oft **spezifische** Dinge (z. B. *Sora*, *RAISE Act*), die zu selten für die Häufigkeitsfilter sind. Die beiden Graphen **ergänzen** sich also.
> - Mit ca. 73 % klar korrekten Relationen ist die Qualität brauchbar, aber nicht perfekt – in Stage 3 sollte man Graph-Fakten immer zusammen mit dem Beleg-Text verwenden.

---

## 8. Topic Modeling – über welche Themen wird berichtet?
*Notebook: „Topic Modeling“, „Text Preparation for Topic Modeling“*

Die 4 Governance-Themen aus Schritt 3 kommen von den **Tags**. Topic Modeling findet dagegen Themen **direkt aus den Wörtern** der Texte – ohne Vorgabe.

### 8a. Vorverarbeitung: Lemmatisierung
Jedes Wort wird auf seine Grundform gebracht, Füllwörter (*the, for, are …*) und Zahlen werden entfernt:

```
Original:   Artificial Intelligence Disclosures Are Key to Customer Trust. MIT SMR’s fall 2024 issue
            highlights the need for personal and organizational resilience … have assembled an
            international panel of AI experts …
Lemmatisiert: artificial intelligence disclosures key customer trust mit smr fall issue highlight need
            personal organizational resilience … assemble international panel expert …
```

*highlights → highlight*, *assembled → assemble*, *experts → expert*. So zählen „regulation“ und „regulations“ als dasselbe Wort.
(Kleines Detail: „Disclosures“ im Titel bleibt stehen, weil spaCy es wegen der Grossschreibung als Eigenname behandelt.)

Für NER (Schritt 4) wird **nicht** lemmatisiert – dort ist die genaue Schreibweise von Namen wichtig.

**Eine Stolperfalle:** spaCy macht aus *data* die lateinische Einzahl *datum*. Weil *data* in diesem Korpus immer „Daten“ meint, wird das nach der Lemmatisierung gezielt zurückgesetzt. (Beim ersten Versuch hat diese Korrektur nicht gegriffen, weil beim Einfügen in das Notebook aus der Regex-Wortgrenze `\b` ein unsichtbares Steuerzeichen wurde – ein gutes Beispiel, warum man Ergebnisse immer kontrollieren sollte, statt dem Code blind zu vertrauen.)

```python
nlp_lemma = spacy.load("en_core_web_sm", disable=["parser", "ner"])   # Tagger + Lemmatizer reichen

def lemmatize(doc):
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2   # nur Wörter, keine Füllwörter
    )

texts = governance_df["title"] + ". " + governance_df["content_text"]
governance_df["text_lemmatized"] = [lemmatize(doc) for doc in nlp_lemma.pipe(texts, batch_size=16)]
```

### 8b. TF-IDF – welche Wörter sind typisch für einen Artikel?

**Das Problem:** Ein Computer kann nicht „lesen“. Für das Topic-Modell muss jeder Artikel in **Zahlen** umgewandelt werden. Die naheliegende Idee – einfach zählen, wie oft jedes Wort vorkommt – hat einen Haken:

- Ein Artikel über den **EU AI Act** und einer über **KI-Agenten** enthalten beide sehr oft Wörter wie *data*, *company* oder *model*.
- Nach reinem Zählen sähen sie deshalb ähnlich aus. Die Wörter, die sie wirklich **unterscheiden** (*regulation*, *fine* vs. *agent*, *workflow*), gehen in der Menge unter.

**Die Lösung:** TF-IDF gibt jedem Wort in jedem Artikel ein Gewicht aus zwei Teilen:

| Teil | Frage | Wirkung |
|---|---|---|
| **TF** (term frequency) | Wie oft kommt das Wort **in diesem Artikel** vor? | oft → wichtig für diesen Artikel |
| **IDF** (inverse document frequency) | In wie vielen **aller** Artikel kommt das Wort vor? | in fast allen → unwichtig, weil es nichts unterscheidet |

**Gewicht = TF × IDF.** Ein Wort bekommt also nur dann ein hohes Gewicht, wenn es *in diesem Artikel häufig* **und** *im Gesamtkorpus selten* ist.

#### Beispiel zum Nachrechnen
Angenommen, der Korpus hätte nur **4 Artikel**, und wir betrachten Artikel 1 über den EU AI Act:

| Wort | Vorkommen in Artikel 1 (TF) | in wie vielen der 4 Artikel? | IDF = log(4 / Anzahl) | **TF × IDF** |
|---|---|---|---|---|
| *data* | 8 | 4 von 4 | log(4/4) = log(1) = **0** | 8 × 0 = **0** |
| *regulation* | 6 | 2 von 4 | log(4/2) = log(2) ≈ **0.69** | 6 × 0.69 ≈ **4.2** |
| *fine* | 3 | 1 von 4 | log(4/1) = log(4) ≈ **1.39** | 3 × 1.39 ≈ **4.2** |

Was man daran sieht:
- *data* kommt im Artikel am häufigsten vor (8×), bekommt aber das Gewicht **0** – es steht in *jedem* Artikel und sagt deshalb nichts über diesen Artikel aus.
- *fine* kommt nur 3× vor, ist aber genauso wichtig wie *regulation*, weil es im Korpus **selten** ist: Wenn ein Artikel über Bussen spricht, ist das ein starkes Merkmal.
- Die Rangfolge nach TF-IDF (*regulation*, *fine* vor *data*) beschreibt den Artikel viel besser als die Rangfolge nach reiner Häufigkeit (*data* vor allem anderen).

> scikit-learn rechnet die Formel leicht anders (geglättet und normalisiert, damit lange Artikel nicht automatisch höhere Werte bekommen) – das Prinzip ist aber genau dieses.

#### Was bringt das?
Am Ende ist jeder Artikel eine **Zeile in einer grossen Tabelle**: 12'946 Artikel × 15'000 Wörter, in den Zellen stehen die TF-IDF-Gewichte.

```
                 regulation   fine   agent   workflow   data   ...
Artikel 1 (AI Act)    4.2      4.2     0        0        0
Artikel 2 (Agents)    0        0       5.1      3.8      0
...
```

Genau diese Tabelle braucht **NMF** im nächsten Schritt: NMF sucht Gruppen von Wörtern, die **in denselben Artikeln gemeinsam hohe Gewichte** haben. Haben viele Artikel hohe Werte bei *agent*, *mcp*, *workflow*, *tool*, wird daraus ein Thema („AI Agents, MCP & Enterprise Tooling“). Ohne IDF würden Allerweltswörter wie *data* oder *company* jedes Thema dominieren, und alle Themen sähen gleich aus.

**Kurz:** TF-IDF filtert das „Rauschen“ aus den Wörtern heraus. Erst dadurch findet das Topic-Modell sinnvolle, unterscheidbare Themen.

**Bezug zu Stage 3:** Dasselbe Prinzip steckt in der *sparse retrieval* (z. B. BM25, ein Verwandter von TF-IDF). Wenn das RAG-System passende Artikel zu einer Frage sucht, zählen seltene, spezifische Suchwörter („FTC“, „fine“) mehr als häufige („AI“, „data“).

```python
tfidf_vectorizer = TfidfVectorizer(
    stop_words=list(custom_stop_words),   # zusätzlich z. B. "ai", "say", Textbausteine einzelner Quellen
    ngram_range=(1, 2),                   # Einzelwörter und Wortpaare ("foreign policy")
    min_df=10,                            # Wort muss in mind. 10 Artikeln vorkommen
    max_df=0.85,                          # ... und darf nicht in über 85 % vorkommen
    max_features=15000
)
tfidf_matrix = tfidf_vectorizer.fit_transform(governance_df["text_lemmatized"])   # Artikel x Wörter
```

#### Am echten Beispiel
Für unseren Beispielartikel (100583) berechnet der gespeicherte TF-IDF-Vektorisierer diese Gewichte:

| Rang | Wort | TF-IDF-Gewicht | Warum hoch? |
|---|---|---|---|
| 1 | *disclosure* | **0.685** | kommt oft vor *und* ist im Korpus selten (IDF 3.79) |
| 2 | *rai* | 0.290 | Abkürzung für „Responsible AI“, typisch für diesen Artikel |
| 3 | *disclose* | 0.193 | |
| 4 | *mit* | 0.152 | Quelle (MIT Sloan) |
| 5 | *transparency* | 0.143 | |
| … | *company* | 0.119 | häufig, aber auch sonst verbreitet (IDF 1.38) |
| … | *data* | 0.054 | steht in fast jedem Artikel (IDF 1.20) → tiefes Gewicht |

Genau wie im Rechenbeispiel: Das Thema des Artikels (Offenlegung, Transparenz, Responsible AI) steht ganz oben, Allerweltswörter wie *data* ganz unten.

### 8c. NMF – Themen finden
NMF (Non-negative Matrix Factorization) zerlegt alle Artikel in eine Mischung aus **8 Themen**. Jedes Thema ist eine Liste typischer Wörter:

| Nr. | Thema | typische Wörter | Artikel |
|---|---|---|---|
| 1 | Business, Markets & Investment | business, company, customer, market, financial | 2'730 (21 %) |
| 2 | AI Agents, MCP & Enterprise Tooling | agent, agentic, mcp, tool, workflow | 1'226 (9 %) |
| 3 | Software Engineering & DevSecOps | software, devsecops, orchestration *(Stellenanzeigen einer Quelle)* | 281 (2 %) |
| 4 | Geopolitics & Public Policy | trump, china, government, president, policy | 2'466 (19 %) |
| 5 | LLMs, Model Training & Safety | model, llm, language, training, openai | 2'102 (16 %) |
| 6 | Cybersecurity & Cloud Security | security, cloud, threat, attack, vulnerability | 1'153 (9 %) |
| 7 | Search, Digital Media & Marketing | search, content, seo, brand, marketing | 1'342 (10 %) |
| 8 | Data Governance, Privacy & Compliance | data, healthcare, patient, compliance, privacy | 1'646 (13 %) |

**Am Beispiel:** Unser Artikel ist eine **Mischung** – am stärksten *Business, Markets & Investment* (Gewicht 0.031), knapp gefolgt von *Data Governance, Privacy & Compliance* (0.027). Als „dominant topic“ zählt Thema 1. Das zeigt: Die Zuordnung zu *einem* Thema ist eine Vereinfachung – viele Artikel liegen zwischen Themen.

Jeder Artikel bekommt das Thema, das am stärksten auf ihn zutrifft („dominant topic“). Die Namen der Themen vergibt nicht der Algorithmus, sondern der Mensch anhand der Wörter – im Notebook automatisch über eine Liste typischer Begriffe, damit die Namen nicht vertauscht werden, wenn sich die Reihenfolge der Themen ändert.

```python
nmf_model = NMF(n_components=8, init="nndsvda", random_state=42, max_iter=400)
document_topic_matrix = nmf_model.fit_transform(tfidf_matrix)   # Artikel x 8 Themen (Mischungsanteile)
topic_term_matrix = nmf_model.components_                       # 8 Themen x Wörter

# Namen über typische Begriffe zuordnen (statt fest über die Nummer)
TOPIC_SIGNATURES = {"Geopolitics & Public Policy": {"trump", "china", "president", "government", ...}, ...}
term_overlap = [[len(top_terms & TOPIC_SIGNATURES[label]) for label in TOPIC_SIGNATURES]
                for top_terms in topic_top_terms]
topic_indices, label_indices = linear_sum_assignment(-np.array(term_overlap))   # beste 1:1-Zuordnung

governance_df["dominant_topic"] = document_topic_matrix.argmax(axis=1) + 1
governance_df["dominant_topic_label"] = governance_df["dominant_topic"].map(topic_labels)
```

### 8d. Entwicklung über die Zeit
Pro Monat wird gezählt, welcher Anteil der Governance-Artikel zu welchem Thema gehört. Verglichen werden die ersten drei mit den letzten drei Monaten:

| Thema | erste 3 Monate | letzte 3 Monate | Veränderung |
|---|---|---|---|
| AI Agents, MCP & Enterprise Tooling | 1.5 % | 16.9 % | **+15.4** |
| Software Engineering & DevSecOps | 0.0 % | 7.7 % | +7.7 |
| Cybersecurity & Cloud Security | 8.2 % | 11.4 % | +3.3 |
| Search, Digital Media & Marketing | 7.5 % | 10.7 % | +3.2 |
| LLMs, Model Training & Safety | 19.6 % | 18.0 % | −1.5 |
| Geopolitics & Public Policy | 21.8 % | 14.9 % | −6.9 |
| Business, Markets & Investment | 21.6 % | 14.0 % | −7.7 |
| Data Governance, Privacy & Compliance | 19.9 % | 6.4 % | **−13.6** |

```python
topic_time_df["month"] = topic_time_df["date"].dt.to_period("M").dt.to_timestamp()
monthly_topic_counts = topic_time_df.groupby(["month", "dominant_topic_label"]).size().unstack(fill_value=0)
monthly_topic_share = monthly_topic_counts.div(monthly_topic_counts.sum(axis=1), axis=0) * 100
```

> **Fazit Schritt 8:**
> - Der deutlichste Trend der ganzen Stage 1: **KI-Agenten** (inkl. Model Context Protocol) wachsen von fast nichts auf ein Sechstel der Governance-Berichterstattung. Gleichzeitig verliert **Datenschutz/Compliance** stark an Anteil. Die Diskussion verschiebt sich also von „Wie schützen wir Daten?“ hin zu „Wie kontrollieren wir autonome KI-Systeme im Einsatz?“ – passend dazu wächst auch **Cybersecurity**.
> - Das passt zu Schritt 3 (Accountability wird wichtiger): Bei Agenten stellt sich die Frage, wer für ihre Handlungen verantwortlich ist.
> - **Die Lemmatisierung hat die Themen spürbar verändert:** Ohne sie gab es ein Thema „Science, Society & Human Impact“, mit ihr entsteht ein eigenes **Cybersecurity**-Thema (weil *threat/threats*, *attack/attacked* jetzt als ein Wort zählen). Topics sind also keine „Wahrheit“, sondern hängen von der Vorverarbeitung ab.
> - Deshalb werden die Namen im Notebook über **typische Begriffe** statt über die Nummer vergeben – und das Notebook bricht ab, wenn ein Name nicht mehr passt. Genau das ist während der Arbeit passiert und hat verhindert, dass Themen falsch beschriftet wurden.
> - Thema 3 (Software/DevSecOps) besteht grösstenteils aus Stellenanzeigen-Text einer einzelnen Quelle und sollte nicht inhaltlich interpretiert werden.

---

## 9. Was am Ende herauskommt – und wofür

| Ergebnis | Ordner | Wofür |
|---|---|---|
| Sauberer Gesamtkorpus | `data/processed/corpus/` | **Stage 2**: Embeddings und Fine-Tuning trainieren |
| Lemmatisierte Governance-Texte | `data/processed/corpus/` | Stage 2 (Word2Vec/FastText) |
| Entitäten pro Artikel, Co-Occurrence, LLM-Antworten | `data/processed/entities/` | Nachvollziehbarkeit, Wiederverwendung |
| Topics pro Artikel und pro Monat | `data/processed/topics/` | Trendanalyse, Stage 2 (z. B. Klassifikation) |
| **Knowledge Graph + Schema + extrahierte Relationen** | `outputs/knowledge_graph/` | **Stage 3**: graph-gestütztes RAG |
| Topic-Modell | `outputs/models/stage1/` | Wiederverwendung ohne Neutraining |
| Alle Abbildungen | `outputs/figures/stage1/` | Präsentation, Bericht |

---

## 10. Gesamtfazit – was Stage 1 zeigt

1. **Datenqualität zuerst:** Ohne Bereinigung wären fast 50'000 kaputte Zeichen und 72 doppelte Artikel in alle Auswertungen eingeflossen.
2. **AI Governance ist ein grosses Thema:** Gut ein Viertel aller KI-Artikel betrifft Governance; *Accountability* dominiert und wird über die Zeit wichtiger.
3. **Wer prägt die Diskussion?** Im Zentrum stehen die USA, Trump, China und die grossen US-Tech-Firmen (Google, OpenAI, Microsoft). AI Governance wird stark geopolitisch und US-zentriert diskutiert.
4. **Wie verändert sich die Diskussion?** Weg von Datenschutz, Business und Politik, hin zu **KI-Agenten, ihrer Sicherheit und Kontrolle** – besonders 2026.
5. **Was Stage 3 bekommt:** einen Wissensgraphen mit drei Ebenen – wer wird zusammen erwähnt (Co-Occurrence), zu welchem Governance-Thema gehört etwas (Theme-Kanten), und wer tut was mit wem, mit Quelle und Datum (LLM-Relationen).
6. **Methodische Lehren:** Allgemeine NLP-Modelle brauchen fachliche Korrekturen (EntityRuler, Normalisierung); LLM-Extraktion ist nützlich, aber zu etwa einem Viertel ungenau; Topics hängen von der Vorverarbeitung ab; und Prüfungen im Code (z. B. bei den Topic-Namen) fangen Fehler ab, die man sonst übersehen würde.

---

## Glossar

| Begriff | Bedeutung |
|---|---|
| **Korpus** | Die Sammlung aller Texte, mit der man arbeitet |
| **Mojibake** | Zeichensalat durch falsch eingelesene Kodierung (`â€™` statt `’`) |
| **Tag** | Schlagwort, das im Datensatz schon an jedem Artikel hängt |
| **NER** | Named Entity Recognition – automatisches Erkennen von Namen und deren Typ |
| **Entität** | Ein erkannter „Akteur“ oder Gegenstand: Firma, Person, Land, Gesetz, … |
| **EntityRuler** | Feste Regeln in spaCy, die bestimmte Begriffe immer gleich erkennen |
| **Normalisierung** | Vereinheitlichen von Schreibweisen (`US` → `United States`) |
| **Co-Occurrence** | Gemeinsames Vorkommen zweier Entitäten im selben Artikel |
| **Knowledge Graph** | Netzwerk aus Knoten (Entitäten) und Kanten (Beziehungen) |
| **Schema** | Festlegung, welche Knoten- und Kantentypen erlaubt sind |
| **PageRank** | Mass für die Wichtigkeit eines Knotens im Netz |
| **LLM** | Large Language Model, z. B. Gemini |
| **Few-Shot** | Dem LLM werden einige gelöste Beispiele gezeigt |
| **Grounding** | Jede Extraktion ist an eine konkrete Textstelle gebunden |
| **Lemmatisierung** | Wörter auf ihre Grundform bringen (*regulations* → *regulation*) |
| **TF-IDF** | Gewichtung, wie typisch ein Wort für einen Text ist |
| **NMF** | Verfahren, das Texte in eine Mischung von Themen zerlegt |
| **RAG** | Retrieval-Augmented Generation – ein LLM, das vor dem Antworten passende Texte (und hier Graph-Wissen) nachschlägt (Stage 3) |
