"""Offline checks binding answer experiments to the frozen retrieval snapshot.

This module reads small experiment artifacts and hashes the source corpus. It
never loads the retrieval index, unpickles the corpus, or contacts a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import fingerprint, read_jsonl, sha256_file
from .data import CATEGORIES, answer_evaluation_sample, validate_qas


VARIANTS = {"text_only", "graph", "graph_rerank"}


def _fail(reason: str) -> None:
    raise ValueError(f"Answer snapshot validation failed: {reason}. "
                     "Rerun QA preparation and retrieval before answer generation.")


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"cannot read {path.name}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path.name} must contain a JSON object")
    return value


def _frame(path: Path, required: set[str]) -> pd.DataFrame:
    try:
        frame = pd.DataFrame(read_jsonl(path))
    except (OSError, ValueError, TypeError) as exc:
        _fail(f"cannot read {path.name}: {exc}")
    if frame.empty or not required.issubset(frame.columns):
        _fail(f"{path.name} is empty or lacks {sorted(required - set(frame.columns))}")
    return frame


def _unique(frame: pd.DataFrame, key: str, label: str) -> None:
    if frame[key].isna().any() or not frame[key].is_unique:
        _fail(f"{label} contains missing or duplicate {key}")


def _disjoint_articles(frame: pd.DataFrame, label: str) -> None:
    if not set(frame["split"]).issubset({"development", "test"}):
        _fail(f"{label} contains an unknown source split")
    development = set(frame.loc[frame.split.eq("development"), "record_id"].astype(str))
    test = set(frame.loc[frame.split.eq("test"), "record_id"].astype(str))
    if not development or not test:
        _fail(f"{label} must include development and test articles")
    if development & test:
        _fail(f"source article leakage between development and test in {label}")


def _balanced_sample(frame: pd.DataFrame, *, count: int, per_category: int, label: str) -> None:
    _unique(frame, "qa_id", label)
    if (len(frame) != count or set(frame.category) != set(CATEGORIES)
            or not frame.category.value_counts().eq(per_category).all()):
        _fail(f"{label} must contain exactly {count} questions ({per_category} per category)")
    if not frame.split.eq("test").all():
        _fail(f"{label} must contain held-out test questions only")


def _check_sample_records(sample: pd.DataFrame, qas: pd.DataFrame, label: str) -> None:
    lookup = {str(row["qa_id"]): row for row in qas.to_dict("records")}
    for row in sample.to_dict("records"):
        current = lookup.get(str(row["qa_id"]))
        if current is None or fingerprint(row) != fingerprint(current):
            _fail(f"{label} question {row['qa_id']} differs from the current effective QA record")


def preflight_answers(project_root: Path | str) -> tuple[pd.DataFrame, dict, dict]:
    """Verify frozen 60/15 paired held-out samples before any local-model call.

    Hashes bind questions, manual reviews, selected source passages, retrieval
    results/code, Stage 1 provenance and graph. Semantic checks additionally
    reject source leakage, altered golds/questions, unpaired variant runs and
    resampling after outcomes are available. Every failure is explicit.
    """
    root = Path(project_root)
    data = root / "data/processed/stage3"
    output = root / "outputs/results/stage3"
    retrieval_manifest = _object(output / "retrieval_manifest.json")
    qa_manifest = _object(data / "qa_dataset_manifest.json")
    handoff_path = root / "outputs/validation/stage1/stage3_handoff.json"
    handoff = _object(handoff_path)
    chunk_manifest = _object(data / "chunk_manifest.json")
    graph_relative = handoff.get("artifacts", {}).get("graph")
    if not isinstance(graph_relative, str) or not graph_relative:
        _fail("Stage 1 handoff has no graph artifact")

    hashes: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in hashes:
            try:
                hashes[path] = sha256_file(path)
            except OSError as exc:
                _fail(f"cannot hash {path.name}: {exc}")
        return hashes[path]

    paths = {
        "qa_dataset_sha256": data / "qa_dataset.jsonl",
        "qa_dataset_manifest_sha256": data / "qa_dataset_manifest.json",
        "qa_review_sha256": data / "qa_manual_review.csv",
        "selected_passages_sha256": data / "selected_passages.jsonl",
        "test_retrieval_runs_sha256": output / "test_retrieval_runs.json",
        "answer_sample_sha256": data / "answer_evaluation_sample.jsonl",
        "judge_sample_sha256": data / "judge_evaluation_sample.jsonl",
        "stage1_handoff_sha256": handoff_path,
        "graph_sha256": root / graph_relative,
        "retrieval_code_sha256": root / "src/stage3_rag/retrieval.py",
    }
    for key, path in paths.items():
        expected = retrieval_manifest.get(key)
        if not isinstance(expected, str) or not expected or digest(path) != expected:
            _fail(f"stale or missing {key}: {path.name} changed since retrieval")

    corpus = root / "data/processed/corpus/ai_media_clean.pkl.gz"
    expected_corpus = handoff.get("inputs_sha256", {}).get("corpus")
    if not expected_corpus or digest(corpus) != expected_corpus:
        _fail("source corpus changed since the Stage 1 handoff")
    for name, expected in handoff.get("review_sha256", {}).items():
        if digest(root / "data/processed/validation/stage1" / name) != expected:
            _fail(f"Stage 1 source review {name} changed since its handoff")
    if (not retrieval_manifest.get("chunk_cache_id")
            or retrieval_manifest["chunk_cache_id"] != chunk_manifest.get("chunk_cache_id")):
        _fail("chunk_cache_id differs from the prepared source snapshot")
    for key in ("selected_passages_sha256", "stage1_handoff_sha256", "graph_sha256"):
        if chunk_manifest.get(key) != digest(paths[key]):
            _fail(f"chunk preparation has stale {key}")
    if qa_manifest.get("review_sha256") != digest(paths["qa_review_sha256"]):
        _fail("effective QA manifest has a stale manual-review snapshot")
    if qa_manifest.get("draft_sha256") != digest(data / "qa_drafts_codex.jsonl"):
        _fail("QA draft changed before effective QA preparation")
    ai_review = data / "qa_ai_review.jsonl"
    if ai_review.exists() or qa_manifest.get("ai_review_sha256"):
        if not qa_manifest.get("ai_review_sha256") or digest(ai_review) != qa_manifest["ai_review_sha256"]:
            _fail("AI Q&A review changed before effective QA preparation")

    qa_fields = {"qa_id", "passage_id", "record_id", "split", "category", "question",
                 "reference_answer", "gold_chunk_ids", "gold_evidence", "qa_review_verdict"}
    qas = _frame(paths["qa_dataset_sha256"], qa_fields)
    _unique(qas, "qa_id", "effective QA dataset")
    qa_fingerprint = fingerprint(qas.to_dict("records"))
    if qa_manifest.get("effective_qa_fingerprint") != qa_fingerprint:
        _fail("QA dataset manifest fingerprint does not match the effective dataset")
    if retrieval_manifest.get("qa_fingerprint") != qa_fingerprint:
        _fail("retrieval QA fingerprint differs from the current effective dataset")
    _disjoint_articles(qas, "effective QA dataset")
    passages = _frame(paths["selected_passages_sha256"],
                      {"passage_id", "record_id", "split", "text"})
    _unique(passages, "passage_id", "selected passages")
    _disjoint_articles(passages, "selected passages")
    try:
        validate_qas(qas, passages)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        _fail(f"effective QA/source validation failed: {exc}")

    sample = _frame(paths["answer_sample_sha256"], qa_fields)
    judges = _frame(paths["judge_sample_sha256"], qa_fields)
    _balanced_sample(sample, count=60, per_category=12, label="answer sample")
    _balanced_sample(judges, count=15, per_category=3, label="judge sample")
    _check_sample_records(sample, qas, "answer sample")
    _check_sample_records(judges, qas, "judge sample")
    expected_answers = answer_evaluation_sample(qas, per_category=12, seed=42)
    if fingerprint(sample.to_dict("records")) != fingerprint(expected_answers.to_dict("records")):
        _fail("answer sample differs from the deterministic pre-generation selection")
    if not set(judges.qa_id).issubset(set(sample.qa_id)):
        _fail("judge questions are not a subset of the paired answer sample")
    expected_judges = pd.concat([
        group.sort_values("qa_id").sample(n=3, random_state=43)
        for _, group in expected_answers.groupby("category")
    ]).sort_values(["category", "qa_id"])
    if fingerprint(judges.to_dict("records")) != fingerprint(expected_judges.to_dict("records")):
        _fail("judge sample differs from the deterministic pre-generation selection")

    runs = _object(paths["test_retrieval_runs_sha256"])
    if set(runs) != VARIANTS:
        _fail("retrieval variants must be exactly text_only, graph and graph_rerank")
    test_ids = set(qas.loc[qas.split.eq("test") & qas.qa_review_verdict.eq("supported"), "qa_id"].astype(str))
    for variant, current in runs.items():
        if not isinstance(current, dict) or set(current) != test_ids:
            _fail(f"{variant} retrieval IDs do not match the complete current held-out test set")
        for qa_id, hits in current.items():
            if not isinstance(hits, list) or len(hits) > 4 or any(not isinstance(hit, dict) for hit in hits):
                _fail(f"{variant}/{qa_id} does not have a valid four-passage retrieval result")
    return sample, runs, qa_manifest
