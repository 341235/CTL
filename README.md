# CTL Project – Tracking AI Trends: AI Governance

Group project for *Computational Language Technologies* (HSLU, MSc Applied Data Science and AI).
Topic track: **AI Governance** (security, accountability, bias and fairness, regulation), based on the
[AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset) (September 2024 – August 2026).
The project brief is in [`docs/CLT_AI_Trend_Project.pdf`](docs/CLT_AI_Trend_Project.pdf).

| Stage | Notebook | Status |
|---|---|---|
| 1 – Cleaning, EDA, entities, Knowledge Graph, topics | [`notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb`](notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb) | completed |
| 2 – Embeddings (Word2Vec, FastText, Doc2Vec) and QLoRA fine-tuning (Qwen3-1.7B, Qwen3-4B) | [`notebooks/stage2_embeddings/Stage2_Embeddings.ipynb`](notebooks/stage2_embeddings/Stage2_Embeddings.ipynb) | completed |
| 3 – RAG question answering | `notebooks/stage3_rag/` | not started |

## Folder structure

```
ProjectCTL/
├── data/
│   ├── raw/                     Kaggle CSV (not in Git, see Setup)
│   └── processed/
│       ├── corpus/              cleaned corpus, lemmatized Governance texts, tag review
│       ├── entities/            NER results, co-occurrences, cached LLM relation extraction
│       ├── topics/              topic keywords, topic per article, monthly topic shares
│       └── stage2/              train/validation/test splits, document vectors (token cache not in Git)
├── outputs/
│   ├── knowledge_graph/         Knowledge Graph (GraphML), schema (JSON), edges and metrics (CSV)  → Stage 3
│   ├── models/stage1/           NMF topic model and TF-IDF vectorizer
│   ├── models/stage2/           embedding models and QLoRA adapters (not in Git, see Setup)
│   ├── results/stage2/          metric tables, cached QLoRA predictions, relevance scores
│   ├── figures/stage1/          all Stage 1 figures as PNG
│   └── figures/stage2/          all Stage 2 figures as PNG
├── notebooks/                   one folder per stage
├── docs/                        project brief
├── src/                         shared code for later stages
└── requirements.txt
```

## Setup

1. **Python environment** (Python 3.11):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
   `requirements.txt` also installs the spaCy model `en_core_web_sm` and PyTorch with CUDA 12.8.
   For a CPU-only installation, remove the `--extra-index-url` line at the top of the file.
2. **Dataset:** download the CSV from Kaggle and save it as
   `data/raw/ai_media_dataset_20260911.csv`.
3. **Large files (team OneDrive):** files that exceed GitHub's 100 MB limit are shared in the OneDrive folder
   [`CTL_large_files`](https://hsluzern-my.sharepoint.com/:f:/g/personal/michael_nguyen_stud_hslu_ch/IgBuiZZKUtrMRYd4A2ZJ2Fi7AeBmhY87SHB3Av0PP0lKgM4?e=PktggR).
   It mirrors the repository structure: copy its **contents** (`data/` and `outputs/`) into the project root,
   and every file lands in the right place.

   | File | Target folder | Needed for |
   |---|---|---|
   | `ai_media_clean.pkl.gz` | `data/processed/corpus/` | Stage 2 (cleaned full corpus) |
   | `governance_entities_fulltext.pkl` | `data/processed/entities/` | Re-running Stage 1 without repeating NER (~50 min) |
   | `stage2/` (about 2.7 GB) | `outputs/models/` | Original model caches (optional); the revised protocol uses separate matching caches, see step 4 |

   Without the Stage 1 files, the Stage 1 notebook recreates them (about two hours in total).
4. **Stage 2 caches and evaluation:** the saved QLoRA predictions in `outputs/results/stage2/` are earlier
   GPU runs. Their reference split and reviewed input snapshot are checked before reuse; evaluating these
   results does not repeat fine-tuning. The revised classical protocol excludes both validation and test
   articles from phrase/embedding training. Its models are stored separately in
   `outputs/models/stage2/train_only_<fingerprint>/`; the older models directly under `stage2/` belong to
   the previous protocol and are not reused. Matching input/configuration caches skip CPU retraining;
   actual training times are reported in the notebook. Model selection uses validation Macro-F1 across
   all families, and the selected model is used for the Governance insights.
5. **Optional – Qwen chunking:** Section 4 measures input coverage with the real tokenizer. The saved
   Qwen scores use the 256-token prefix. `RUN_CHUNKING_EXPERIMENT = False` keeps a separate four-window
   inference experiment disabled; enabling it requires CUDA and the saved LoRA adapters. Its pooling and
   thresholds are selected on validation, and no chunking F1 is claimed before that experiment runs.
   The local macOS run uses Python 3.12.4 and a Gensim 4.4.0 build with the
   [dot-wrapper correction](https://github.com/piskvorky/gensim/pull/3668); its wheel and build notes are
   retained locally in `.venv/ctl-fixes/`.
6. **Optional – Gemini API key:** the LLM relation extraction in Stage 1 uses Google Gemini via LangExtract.
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
