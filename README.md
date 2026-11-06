# CTL Project – Tracking AI Trends: AI Governance

Group project for *Computational Language Technologies* (HSLU, MSc Applied Data Science and AI).
Topic track: **AI Governance** (security, accountability, bias and fairness, regulation), based on the
[AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset) (September 2024 – August 2026).
The project brief is in [`docs/CLT_AI_Trend_Project.pdf`](docs/CLT_AI_Trend_Project.pdf).

| Stage | Notebook | Status |
|---|---|---|
| 1 – Cleaning, EDA, entities, Knowledge Graph, topics | [`notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb`](notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb) | completed |
| 1 – Feedback: article relevance, edge evidence and graph linkage | [`notebooks/stage1_ai_governance/Stage1_AI_Governance_Validation.ipynb`](notebooks/stage1_ai_governance/Stage1_AI_Governance_Validation.ipynb) | source review and evidence audit; review method disclosed per row |
| 2 – Embeddings (Word2Vec, FastText, Doc2Vec) and QLoRA fine-tuning (Qwen3-1.7B, Qwen3-4B) | [`notebooks/stage2_embeddings/Stage2_Embeddings.ipynb`](notebooks/stage2_embeddings/Stage2_Embeddings.ipynb) | completed |
| 3 – RAG question answering | [`notebooks/stage3_rag/Stage3_RAG.ipynb`](notebooks/stage3_rag/Stage3_RAG.ipynb) | implemented, locally evaluated and Q&A reviewed |

## Folder structure

```
ProjectCTL/
├── data/
│   ├── raw/                     Kaggle CSV (not in Git, see Setup)
│   └── processed/
│       ├── corpus/              cleaned corpus, lemmatized Governance texts, tag review
│       ├── entities/            NER results, co-occurrences, cached LLM relation extraction
│       ├── topics/              topic keywords, topic per article, monthly topic shares
│       ├── stage2/              token lengths, document vectors (token cache not in Git)
│       ├── stage3/              120 passages, 240 Q&A drafts, source splits and preserved manual review
│       └── validation/stage1/   preserved article and relation review CSVs
├── outputs/
│   ├── knowledge_graph/         Knowledge Graph (GraphML), schema (JSON), edges and metrics (CSV)  → Stage 3
│   ├── models/stage1/           NMF topic model and TF-IDF vectorizer
│   ├── models/stage2/           embedding models and QLoRA adapters (not in Git, see Setup)
│   ├── results/stage2/          metric tables, cached QLoRA predictions, relevance scores
│   ├── results/stage3/          retrieval/RAG comparisons, local outputs, AI judges and source review packet
│   ├── validation/stage1/       evidence audit, source reading packets, entity/article index, handoff graph
│   ├── figures/stage1/          all Stage 1 figures as PNG
│   ├── figures/stage2/          all Stage 2 figures as PNG
│   └── figures/stage3/          dataset, retrieval, answer and graph figures as PNG
├── notebooks/                   one folder per stage
├── docs/                        project brief
├── src/                         shared code for later stages
└── requirements.txt
```

See the [CSV artifact map](docs/CSV_ARTIFACTS.md) for retained files and removed duplicates.

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

Before Stage 3, run the supplementary [validation notebook](notebooks/stage1_ai_governance/Stage1_AI_Governance_Validation.ipynb).
Its scope is the instructor's requested article/edge sample validation, documented limitations and
reliable linkage of the two graph layers to source articles. The notebook's *Response to the Stage 1
feedback* section maps each request to the implemented evidence. The measured weaknesses of the tag
filter are reported as findings; filter redesign and further model training are separate tasks.
It uses the local Stage 1/2 exports without model training or API calls. It selects 120 originally tagged
Governance articles and 40 separate Stage 2 candidates, audits all extracted relation evidence, and prepares
40 important/random relation reviews. The HTML source packets are in `outputs/validation/stage1/`;
The user-requested Codex source assessments are recorded in the two CSVs under
`data/processed/validation/stage1/` as `review_method=ai_assisted`, with exact source quotes and rationales.
They are AI-assisted judgments, not human annotations. Human reviewers can replace individual decisions
using `review_method=human` and their own identity/date. Original AI decisions are archived as JSONL.
Reviewer fields are preserved; changed reviewed sources stop execution.

The handoff files are `outputs/validation/stage1/stage3_handoff.json`,
`audited_retrieval_graph.graphml` and `entity_articles.csv.gz`. The manifest remains
`pending_review` while decisions are missing; AI-assisted completion is explicitly reported as
`ai_reviewed` or `ai_reviewed_with_unresolved_cases`. The separate `human_review_complete` flag remains
false until actual human assessment. Review completion does not validate every article: inspect the
AI-assessed weighted relevance rate/bounds and topic failures before
selecting Stage 3 passages. Unreviewed/uncertain articles keep their weak-label status; reviewed irrelevant
articles and out-of-period articles are excluded. Only source-verified, supported semantic relations
are exported as assertions with their review method attached. The remaining co-occurrence/theme edges represent associations with retrievable
article support. Relation-only entities retain direct provenance on the accepted semantic edges.

The original exports below remain available for reproducing Stage 1; they include unreviewed semantic
relations and are not the reviewed Stage 3 handoff.

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

Validation helpers live in `src/stage1_validation/`. Run their focused regression checks with:

```bash
python -m unittest discover -s tests -p 'test_stage1_validation*.py' -v
```

## Open the UI of stage 3

If it is already running, open [http://127.0.0.1:8501](http://127.0.0.1:8501).
Otherwise, from the project root:

1. Start the Ollama application. The existing `llama3.2:latest` model is required.
2. Start the UI with the project environment (including `streamlit`):

   ```bash
   .venv/bin/python -m streamlit run src/stage3_rag/ui.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
   ```

3. Open the link above, enter a question, choose a retrieval method and click **Ask question**.

The first question loads the cached search indexes. Keep the terminal open; **Ctrl+C** stops the UI.
The UI reuses the prepared Stage 3 data, cached models and retrieval configuration.

## Use of AI tools

AI coding tools are used in this project and are disclosed per team member in each notebook,
as required by the project brief.
