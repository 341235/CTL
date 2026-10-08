w# Stage 1 – AI Governance: Offene Punkte

Stand: 08.10.2026 · Grundlage: `projectOrder/CLT_AI_Trend_Project.pdf` und Review von `Stage1_AI_Governance.ipynb`

Legende: ✅ erledigt · [~] teilweise · 🔴 muss (bricht sonst / Auftrag nicht erfüllt) · 🟠 sollte (Risiko bei Bewertung) · 🟢 optional

---

## A. Lauffähigkeit herstellen

- [x] ✅ Datensatz nach `data/raw/ai_media_dataset_20260911.csv` legen (Kaggle: jannalipenkova/ai-media-dataset)
- [x] ✅ `openpyxl` und spaCy-Modell `en_core_web_sm` installieren
- [x] ✅ Python-Version geklärt (Notebook erstellt mit 3.11, lokal 3.13 / pandas 3)
- [x] ✅ **Zellreihenfolge fixen:** Zelle „COMPLETE TAG REVIEW“ (Excel-Export) nutzt `tag_counts`, das erst in der *nächsten* Zelle definiert wird → bei „Run All“ `NameError`. Die `tag_counts`/Top-20-Zelle (und den Plot) **vor** den Export verschieben.
- [x] ✅ `requirements.txt` ergänzen, damit alle dieselbe Umgebung haben:
  ```
  openpyxl
  joblib
  ipykernel
  en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
  ```
- [x] ✅ **Kernel neu starten → „Run All“** (08.10.2026, fehlerfrei, ~67 min; Ergebnisse identisch mit Originallauf) und prüfen, dass alles ohne Fehler durchläuft (NER dauert lange; bricht er ab, setzt er beim Checkpoint fort).
- [x] ✅ Zahlen in allen Markdown-Texten mit dem finalen Lauf abgeglichen (automatische Prüfung: keine veralteten Werte mehr).
- [x] ✅ Hart codierte Werte: `66` und `0` durch berechnete Variablen ersetzt; `49305`/`21` in den Encoding-Tabellen bleiben dokumentierte Ergebnisse der Zellen darüber.

## B. Daten für das ganze Team in `data/processed/`

- [x] ✅ `.gitignore` angepasst: `data/processed/` wird jetzt versioniert (ausser `*_checkpoint.pkl`, `*.tmp`, Office-Lockfiles).
- [x] ✅ **Bereinigten Gesamtkorpus speichern** (`ai_media_clean.pkl.gz`, 164 MB) – fehlt aktuell komplett! `df_clean` (alle 47'751 Artikel, bereinigt) wird nirgends gespeichert, Stage 2 braucht aber den ganzen Korpus. Vorschlag am Ende von Abschnitt 2:
  ```python
  df_clean.to_pickle(PROJECT_ROOT / "data" / "processed" / "ai_media_clean.pkl.gz", compression="gzip")
  ```
  (Pickle behält die Listen in `tags`/`content`. Alternative Parquet bräuchte `pyarrow`.)
- [x] ✅ Nach dem Lauf **Dateigrössen prüfen** → Entscheid: Variante B (OneDrive für `ai_media_clean.pkl.gz` und `governance_entities_fulltext.pkl`, Rest via Git). GitHub lehnt Dateien > 100 MB ab (Warnung ab 50 MB). Kandidaten für zu gross: bereinigter Korpus, `governance_entities_fulltext.pkl` (enthält Volltexte), `governance_cooccurrence_full.pkl`.
  - Entscheid im Team: grosse Dateien per **Git LFS** (`git lfs track "data/processed/*.pkl*"`, ist lokal installiert; Achtung: Gratis-Kontingent 1 GB Speicher/Traffic pro Monat) **oder** über OneDrive/Teams teilen und nur kleine Dateien (GraphML, CSV, JSON, joblib) in Git.
  - 🟢 Vorschlag: `governance_entities_fulltext.pkl` ohne `content_text` speichern (Text steckt schon im Gesamtkorpus) → Datei würde vermutlich < 20 MB und könnte normal über Git statt OneDrive geteilt werden. Der NER-Cache im Notebook braucht nur `record_id` + `entities`.
- [x] ✅ Excel-Export `AI_Governance_All_Tags_Review.xlsx` steht jetzt nach der finalen Tag-Zuordnung und ist ausgefüllt (30 × „Yes“ mit Thema, Review-Notizen für die diskutierten Ausschlüsse; Prüfung, dass alle zugeordneten Tags im Datensatz existieren).
- [x] ✅ Knotenzahl einheitlich: Schwellenwert-Tabelle und Graph-Zelle zählen jetzt Name + Typ (überall 723 statt teils 703).
- [x] ✅ Neue Ordnerstruktur: Graph, Schema und Tabellen in `outputs/knowledge_graph/`, Modelle in `outputs/models/stage1/`, Abbildungen in `outputs/figures/stage1/`, Daten in `data/processed/{corpus,entities,topics}/`.
- [ ] 🟠 `README.md` ergänzen (**wartet auf OneDrive-Freigabelink**): Setup (venv, `pip install -r requirements.txt`, Datensatz-Download, Notebook-Reihenfolge) und Liste der Dateien in `data/processed/`.

## C. Inhaltliche Lücken gegenüber dem Auftrag

- [x] ✅ **Strukturierte Relationsextraktion** mit LangExtract + Gemini (`gemini-3.5-flash-lite`) auf 408 Artikeln: 162 Relationen, 100 % an Textstellen verankert, manuelle Prüfung von 40 Relationen (73 % korrekt, 15 % schwach belegt, 12 % falsch); Ergebnisse gecacht, eigener Relationsgraph für Stage 3.
  - Relationsschema definieren, z. B. `ORG –DEVELOPS→ MODEL`, `GOV/ORG –REGULATES→ ORG/TECH`, `LAW –APPLIES_TO→ ORG/TECH`, `ORG –COMPLIES_WITH→ LAW/STANDARD`, `ORG –FINED_BY/SUED_BY→ ORG/GPE`, `PERSON –LEADS→ ORG`
  - Mit LangExtract (braucht API-Key, z. B. Gemini) oder einem LLM auf einer **Stichprobe** (z. B. 300–1'000 Governance-Artikel, nicht alle 12'972 → Kosten/Zeit) extrahieren
  - Extrahierte Relationen als zusätzliche Kantentypen in den Graph, mit `record_id` + `date` als Quelle (wichtig für Stage 3: Nachvollziehbarkeit und zeitbezogene Fragen)
  - *Geprüft 08.10.:* bestätigt – keine Zelle extrahiert Relationen aus dem Text. Aufwand mittel–hoch; Entscheid nötig: LLM-Anbieter/API-Key und Stichprobengrösse.
- [x] ✅ **Schema vor dem Graphbau** mit Data Dictionary (Knoten- und Relationstypen) am Anfang des Knowledge-Graph-Kapitels; eine gemeinsame Schema-Datei für alle Graphen.
- [x] ✅ **Governance-spezifische Entitätstypen** per EntityRuler (LAW, STANDARD, REGULATOR; 25 Muster) + Cache-Validierung der NER; EU AI Act jetzt einheitlich in 790 Artikeln.
  - *Geprüft 08.10.:* Governance-Artikel erwähnen u. a. EU AI Act (742), GDPR (665), Executive Order (389), NIST (247), FTC (235), OECD (111), DSA/DMA (107), NIST AI RMF (85), ISO 42001 (57), EU AI Office (48). spaCy labelt diese inkonsistent: „AI Act“ als ORG/PERSON/PRODUCT, „ISO 42001“ sogar als LAW/ORG/PERSON/PRODUCT; NIST/FTC/OECD nur als generische ORG (kein Typ REGULATOR/STANDARD). → Lücke bestätigt, Fix gut machbar (Ruler + Aliase, NER-Neulauf ~60 min).
- [x] ✅ **Duplikate:** 72 Artikel mit identischem Text (nach Bereinigung) entfernt, früheste Veröffentlichung behalten → 47'679 Artikel.
  - *Geprüft 08.10.:* Nach der Textbereinigung sind es sogar **72** Duplikate (vorher 66, weil die Normalisierung weitere Texte angleicht), davon **26 im Governance-Teil**. Stichprobe: echte Doppelerfassungen (gleicher Artikel, gleiche Quelle, gleiches Datum, nur Titel abgeschnitten, z. B. record_id 64238/64242). → Behalten ist hier schwer zu begründen; Entfernen (erstes behalten) empfohlen. Wirkung auf Graph klein (26 von 12'972), aber Markdown-Texte/Zahlen müssen angepasst werden.
- [x] ✅ **Vorverarbeitung:** Lemmatisierung mit spaCy für das Topic Modeling (gecacht), inkl. Korrektur *datum* → *data*; Topic-Namen werden über typische Begriffe vergeben (mit Prüfung).
  - *Geprüft 08.10.:* Lemmatisierung fehlt komplett (im Notebook begründet weggelassen); Stopwörter/Tokenisierung nur implizit im TF-IDF.
- [ ] 🟢 **Sentiment/Stance:** Aktuell nur begründet weggelassen. Sicherer: kleine Stance-Analyse auf Regulation-Artikeln über die Zeit (z. B. Hugging-Face-Modell oder Zero-Shot), mit Einordnung der Grenzen.
- [ ] 🟢 Topic Modeling: optional BERTopic bzw. dynamische Topics als Vergleich zu NMF.

## D. Abgabe-Formalitäten

- [x] ✅ **KI-Tool-Offenlegung mit Toolnamen** auf Gruppenebene: Claude Code (Claude Opus 5.5), Google Gemini via LangExtract (Teil der Analyse), ChatGPT und Claude.
- [x] ✅ **Contributions:** Dana Srdic (Aufbau Stage 1), Michael Nguyen und Henrik Szent-Ivany (Review und Ergänzungen).
- [ ] 🟠 Notebook **mit Ausgaben** committen (Notebook ist fertig ausgeführt; Commit macht Michael) und als reproduzierbares Tutorial gegenlesen.
- [x] ✅ Leeren Ordner `notebooks/stage1_cleaning_eda/` entfernt (nur Platzhalter-README, nirgends referenziert).
- [ ] 🔴 Abgabe über ILIAS gemäss Frist.
