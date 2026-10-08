# CTL Project – Tracking AI Trends: AI Governance

Group project for *Computational Language Technologies* (HSLU, MSc Applied Data Science and AI).
Topic track: **AI Governance** (security, accountability, bias and fairness, regulation), based on the
[AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset) (September 2024 – August 2026).
The project brief is in [`docs/CLT_AI_Trend_Project.pdf`](docs/CLT_AI_Trend_Project.pdf).

| Stage | Notebook | Status |
|---|---|---|
| 1 – Cleaning, EDA, entities, Knowledge Graph, topics | [`notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb`](notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb) | in review, open items in [`docs/STAGE1_TODO.md`](docs/STAGE1_TODO.md) |
| 2 – Embeddings and fine-tuning | `notebooks/stage2_embeddings/` | not started |
| 3 – RAG question answering | `notebooks/stage3_rag/` | not started |

## Folder structure

```
ProjectCTL/
├── data/
│   ├── raw/                     Kaggle CSV (not in Git, see Setup)
│   └── processed/
│       ├── corpus/              cleaned corpus, lemmatized Governance texts, tag review
│       ├── entities/            NER results, co-occurrences, cached LLM relation extraction
│       └── topics/              topic keywords, topic per article, monthly topic shares
├── outputs/
│   ├── knowledge_graph/         Knowledge Graph (GraphML), schema (JSON), edges and metrics (CSV)  → Stage 3
│   ├── models/stage1/           NMF topic model and TF-IDF vectorizer
│   └── figures/stage1/          all Stage 1 figures as PNG
├── notebooks/                   one folder per stage
├── docs/                        project brief and Stage 1 to-do list
├── src/                         shared code for later stages
└── requirements.txt
```

## Setup

1. **Python environment** (Python 3.11+):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
   `requirements.txt` also installs the spaCy model `en_core_web_sm`.
2. **Dataset:** download the CSV from Kaggle and save it as
   `data/raw/ai_media_dataset_20260911.csv`.
3. **Large Stage 1 files:** two files exceed GitHub's 100 MB limit and are shared via the team OneDrive folder
   **[ONEDRIVE LINK – to be added]**. Copy them into your clone:

   | File | Target folder | Needed for |
   |---|---|---|
   | `ai_media_clean.pkl.gz` | `data/processed/corpus/` | Stage 2 (cleaned full corpus) |
   | `governance_entities_fulltext.pkl` | `data/processed/entities/` | Re-running Stage 1 without repeating NER (~50 min) |

   Without these files, the Stage 1 notebook recreates them (about two hours in total).
4. **Optional – Gemini API key:** the LLM relation extraction in Stage 1 uses Google Gemini via LangExtract.
   Its results are cached in `data/processed/entities/governance_relations_langextract.jsonl`, so the notebook
   runs without a key. To extract new relations, create a file `.env` in the project root:
   ```
   GEMINI_API_KEY=your-key
   ```
   `.env` is excluded from Git – never commit API keys.

## Using the Stage 1 outputs

```python
import pandas as pd
import networkx as nx

corpus = pd.read_pickle("data/processed/corpus/ai_media_clean.pkl.gz")        # 1 row per article
kg = nx.read_graphml("outputs/knowledge_graph/ai_governance_knowledge_graph_enriched.graphml")
relations = pd.read_csv("outputs/knowledge_graph/ai_governance_extracted_relations.csv")  # with record_id, date, evidence
```

The schema of all graph files (node types, relation types, attributes) is documented in
`outputs/knowledge_graph/ai_governance_knowledge_graph_schema.json` and in the Stage 1 notebook
(section *Knowledge Graph Schema and Data Dictionary*).

## Use of AI tools

AI coding tools are used in this project and are disclosed per team member in each notebook,
as required by the project brief.
