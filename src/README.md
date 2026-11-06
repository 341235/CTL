# Source Code

This folder contains reusable Python functions and modules used across the project.

Examples:
- Data cleaning utilities
- AI Governance filtering logic
- Entity and relation extraction
- Knowledge graph construction
- Topic modeling helpers

`stage1_validation/` implements the supplementary Stage 1 feedback workflow:

- `articles.py`: reproducible stratified review, coverage and weighted relevance diagnostics
- `graphs.py`: exact evidence checks, preserved relation review, conservative entity linkage and article provenance
- `common.py`: source fingerprints, safe review persistence and offline source reading packets

Run the workflow through `notebooks/stage1_ai_governance/Stage1_AI_Governance_Validation.ipynb`.
The focused checks are in `tests/test_stage1_validation_*.py`.

`stage3_rag/` contains the local Stage 3 workflow:

- `data.py`: exact tokenizer-aligned source chunks, source-held-out Q&A data and preserved manual reviews
- `qa_review.py`: source-bound AI Q&A corrections, explicitly separate from human approval
- `retrieval.py`: sparse/dense/hybrid ranking, source-validated graph expansion and cached cross-encoder re-ranking
- `generation.py`: local Ollama generation with source citations, structured graph evidence and complete caches
- `evaluation.py`: retrieval metrics, explicitly limited answer proxies and actual local AI judge calls
- `snapshot.py`: checks source/question/retrieval snapshots before answer generation
- `pipeline.py`: the shared notebook/CLI experiment orchestration
- `ui.py`: local Streamlit interface for questions and source inspection

The tutorial is `notebooks/stage3_rag/Stage3_RAG.ipynb`; focused checks use
`python -m unittest discover -s tests -p 'test_stage3_*.py' -v`.
