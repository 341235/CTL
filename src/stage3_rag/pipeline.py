"""Reproducible Stage 3 experiments used by the tutorial notebook and CLI."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import time

import networkx as nx
import pandas as pd
import torch
from transformers import AutoTokenizer

from .common import fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from .data import chunk_articles, load_sources, prepare_qa_dataset, answer_evaluation_sample, CATEGORIES
from .retrieval import RetrievalIndex
from .evaluation import evaluate_retrieval, evaluate_answers, judge_answer
from .generation import OllamaGenerator


def prepare_data(project_root):
    root = Path(project_root)
    data_dir = root / "data/processed/stage3"
    cache_dir = root / "outputs/models/stage3"
    output_dir = root / "outputs/results/stage3"
    for directory in (data_dir, cache_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    documents, reviews, handoff = load_sources(root)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
                                             local_files_only=True, use_fast=True)
    tokenizer.model_max_length = 10**9
    spec = {"corpus_sha256": handoff["inputs_sha256"]["corpus"],
            "eligible": documents[["record_id", "source_fingerprint", "article_review_status", "article_review_method"]].to_dict("records"),
            "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
            "tokenizer_vocab_hash": fingerprint(tokenizer.get_vocab()),
            "max_tokens": 200, "overlap_tokens": 32,
            "chunker_code": inspect.getsource(chunk_articles)}
    cache_id = fingerprint(spec)
    chunk_path = cache_dir / f"chunks_{cache_id}.pkl.gz"
    started = time.perf_counter()
    cache_hit = chunk_path.exists()
    if cache_hit:
        chunks = pd.read_pickle(chunk_path)
    else:
        print(f"Chunking {len(documents):,} source articles with exact tokenizer offsets...", flush=True)
        chunks = chunk_articles(documents, tokenizer)
        chunks.to_pickle(chunk_path, compression="gzip")
    passages = pd.DataFrame(read_jsonl(data_dir / "selected_passages.jsonl"))
    chunk_lookup = chunks.set_index("chunk_id")
    for row in passages.to_dict("records"):
        if row["passage_id"] not in chunk_lookup.index or chunk_lookup.loc[row["passage_id"], "text"] != row["text"]:
            raise ValueError("Selected Q&A passage is not an exact chunk of the current corpus")
    if not chunks.chunk_id.is_unique or int(chunks.embedding_tokens.max()) > 256:
        raise ValueError("Invalid source chunk identities or dense input coverage")
    graph = nx.read_graphml(root / handoff["artifacts"]["graph"], force_multigraph=True)
    entities = pd.read_csv(root / handoff["artifacts"]["entity_articles"])
    entities = entities.loc[entities.record_id.isin(documents.record_id)].copy()
    manifest = {"chunk_cache_id": cache_id, "chunk_cache_path": str(chunk_path.relative_to(root)),
                "chunk_cache_hit": cache_hit, "source_articles": len(documents), "chunks": len(chunks),
                "selected_passages": len(passages), "selected_articles": int(passages.record_id.nunique()),
                "max_embedding_tokens": int(chunks.embedding_tokens.max()),
                "chunk_preparation_seconds": time.perf_counter() - started,
                "selected_passages_sha256": sha256_file(data_dir / "selected_passages.jsonl"),
                "stage1_handoff_sha256": sha256_file(root / "outputs/validation/stage1/stage3_handoff.json"),
                "graph_sha256": sha256_file(root / handoff["artifacts"]["graph"]),
                "source_review": "AI-assisted positives for benchmark selection; unreviewed retrieval sources remain weak labels",
                "publication_start": "2024-09-01", "publication_end": "2026-08-31"}
    write_json(data_dir / "chunk_manifest.json", manifest)
    return chunks, passages, graph, entities, manifest


def build_index(project_root, chunks):
    torch.set_num_threads(4)
    print(f"Building/loading sparse and dense indices for {len(chunks):,} chunks...", flush=True)
    started = time.perf_counter()
    from src.stage1_validation.common import load_stage1_normalization
    aliases, _ = load_stage1_normalization(Path(project_root) / "notebooks/stage1_ai_governance/Stage1_AI_Governance.ipynb")
    index = RetrievalIndex(chunks, Path(project_root) / "outputs/models/stage3/retrieval", aliases=aliases)
    summary_path = Path(project_root) / "outputs/results/stage3/index_summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    elapsed = time.perf_counter() - started
    initial_seconds = elapsed if not index.index_cache_hit else None
    if index.index_cache_hit and previous.get("fingerprint") == index.cache_fingerprint:
        initial_seconds = previous.get("initial_index_build_seconds")
        if initial_seconds is None and previous.get("cache_hit") is False:
            initial_seconds = previous.get("wall_seconds")
    write_json(summary_path, {"fingerprint": index.cache_fingerprint, "cache_hit": index.index_cache_hit,
                "wall_seconds": elapsed, "initial_index_build_seconds": initial_seconds,
                "coverage": index.coverage_report,
                "embedding_model": index.embedding_model_id, "embedding_revision": index.model_revision,
                "sparse_model": "TF-IDF unigrams+bigrams, maximum 120000 features"})
    return index


def run_retrieval_experiments(project_root, index, qas, graph, entity_articles):
    root = Path(project_root)
    output = root / "outputs/results/stage3"
    graph_path = root / "outputs/validation/stage1/audited_retrieval_graph.graphml"
    cache_spec = {"index": index.cache_fingerprint, "code": sha256_file(root / "src/stage3_rag/retrieval.py"),
                  "graph_sha256": sha256_file(graph_path),
                  "memberships": fingerprint(entity_articles.to_dict("records")),
                  "reranker_identity": index.reranker_identity(),
                  "aliases": index.aliases, "top_k": 4, "candidate_k": 20,
                  "graph_weight": 0.35, "neighbor_cap": 12, "article_cap": 100}
    run_id = fingerprint(cache_spec)
    query_cache = root / "outputs/models/stage3/retrieval_runs" / run_id
    timings = []

    def query(qa, mode, with_graph=False, rerank=False, count=4):
        spec = {**cache_spec, "qa_id": qa["qa_id"], "question": qa["question"],
                "mode": mode, "graph": with_graph, "rerank": rerank, "count": count}
        path = query_cache / (fingerprint(spec) + ".json")
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("spec") == spec:
                timings.append({"qa_id": qa["qa_id"], "mode": mode, "graph": with_graph,
                                "rerank": rerank, "loaded_result_cache": True, **value["stats"]})
                return value["hits"]
        hits = index.retrieve(qa["question"], mode=mode, top_k=count, candidate_k=20,
            graph=graph if with_graph else None, entity_articles=entity_articles if with_graph else None,
            graph_weight=0.35, rerank=rerank, graph_neighbor_cap=12, graph_article_cap=100)
        value = {"spec": spec, "hits": hits, "stats": index.last_stats.copy()}
        write_json(path, value)
        timings.append({"qa_id": qa["qa_id"], "mode": mode, "graph": with_graph,
                        "rerank": rerank, "loaded_result_cache": False, **value["stats"]})
        return hits

    development = qas.loc[qas.split.eq("development") & qas.qa_review_verdict.eq("supported")]
    test = qas.loc[qas.split.eq("test") & qas.qa_review_verdict.eq("supported")]
    baselines = {}
    for mode in ("sparse", "dense", "hybrid"):
        print(f"Development retrieval: {mode}, {len(development)} questions", flush=True)
        baselines[mode] = {row["qa_id"]: query(row, mode, count=20) for row in development.to_dict("records")}
    _, baseline_scores = evaluate_retrieval(development, baselines, k=4)
    _, candidate_scores = evaluate_retrieval(development, baselines, k=20)
    overall = baseline_scores.loc[baseline_scores.category.eq("all")].sort_values(
        ["mrr", "recall_at_k", "variant"], ascending=[False, False, True])
    selected_mode = str(overall.iloc[0]["variant"])
    selected_top4 = overall.iloc[0]
    selected_top20 = candidate_scores.loc[candidate_scores.category.eq("all") & candidate_scores.variant.eq(selected_mode)].iloc[0]
    limitation = {"selection_split": "development", "selection_metric": "MRR@4, then Recall@4, deterministic name tie-break",
                  "selected_text_retriever": selected_mode,
                  "development_recall_at_4": float(selected_top4.recall_at_k),
                  "development_candidate_recall_at_20": float(selected_top20.recall_at_k),
                  "candidate_context_recall_gap": float(selected_top20.recall_at_k - selected_top4.recall_at_k),
                  "enhancement": "cross-encoder reranking of the same capped 20-candidate graph-aware pool",
                  "rationale": "Evaluate whether relevant evidence already present in the candidate pool is promoted into the four-passage answer context; reranking cannot recover evidence outside that pool.",
                  "test_used_for_selection": False}
    baseline_scores.to_csv(output / "development_retriever_comparison.csv", index=False)
    candidate_scores.to_csv(output / "development_candidate_coverage.csv", index=False)
    write_json(output / "enhancement_design.json", limitation)
    variants = {"text_only": (False, False), "graph": (True, False), "graph_rerank": (True, True)}
    all_runs = {}
    for split, frame in (("development", development), ("test", test)):
        runs = {}
        for variant, (with_graph, rerank) in variants.items():
            print(f"{split} retrieval: {variant}, {len(frame)} questions", flush=True)
            current = {}
            for number, row in enumerate(frame.to_dict("records"), 1):
                current[row["qa_id"]] = query(row, selected_mode, with_graph, rerank)
                if number % 30 == 0:
                    print(f"  {variant}: {number}/{len(frame)}", flush=True)
            runs[variant] = current
        per_question, summary = evaluate_retrieval(frame, runs, k=4)
        per_question.to_csv(output / f"{split}_retrieval_per_question.csv", index=False)
        summary.to_csv(output / f"{split}_retrieval_summary.csv", index=False)
        write_json(output / f"{split}_retrieval_runs.json", runs)
        all_runs[split] = runs
    pd.DataFrame(timings).to_csv(output / "retrieval_runtime.csv", index=False)
    # Freeze equal-category answer and judge samples before invoking any model.
    answer_set = answer_evaluation_sample(qas, per_category=12, seed=42)
    write_jsonl(root / "data/processed/stage3/answer_evaluation_sample.jsonl", answer_set.to_dict("records"))
    judge_set = pd.concat([group.sort_values("qa_id").sample(n=min(3, len(group)), random_state=43)
                          for _, group in answer_set.groupby("category")]).sort_values(["category", "qa_id"])
    write_jsonl(root / "data/processed/stage3/judge_evaluation_sample.jsonl", judge_set.to_dict("records"))
    write_json(output / "retrieval_manifest.json", {
        "experiment_fingerprint": run_id, "index_fingerprint": index.cache_fingerprint,
        "qa_fingerprint": fingerprint(qas.to_dict("records")), "selected_text_mode": selected_mode,
        "development_questions": len(development), "test_questions": len(test),
        "answer_questions": len(answer_set), "judge_questions": len(judge_set),
        "answer_sample_frozen_before_generation": True, "candidate_budget": 20, "answer_context_passages": 4,
        "qa_dataset_sha256": sha256_file(root / "data/processed/stage3/qa_dataset.jsonl"),
        "qa_dataset_manifest_sha256": sha256_file(root / "data/processed/stage3/qa_dataset_manifest.json"),
        "qa_review_sha256": sha256_file(root / "data/processed/stage3/qa_manual_review.csv"),
        "selected_passages_sha256": sha256_file(root / "data/processed/stage3/selected_passages.jsonl"),
        "test_retrieval_runs_sha256": sha256_file(output / "test_retrieval_runs.json"),
        "answer_sample_sha256": sha256_file(root / "data/processed/stage3/answer_evaluation_sample.jsonl"),
        "judge_sample_sha256": sha256_file(root / "data/processed/stage3/judge_evaluation_sample.jsonl"),
        "stage1_handoff_sha256": sha256_file(root / "outputs/validation/stage1/stage3_handoff.json"),
        "graph_sha256": sha256_file(graph_path),
        "retrieval_code_sha256": sha256_file(root / "src/stage3_rag/retrieval.py"),
        "chunk_cache_id": json.loads((root / "data/processed/stage3/chunk_manifest.json").read_text())["chunk_cache_id"],
        "variants": list(variants), "label_status": "provisional unless all effective QAs are human approved",
        "source_definition": "authored gold evidence chunk; overlapping/alternative relevant chunks are not exhaustively labelled"})
    return all_runs, limitation


def run_answer_experiments(project_root, run_judge=False):
    root = Path(project_root)
    output = root / "outputs/results/stage3"
    from .snapshot import preflight_answers
    sample, runs, qa_manifest = preflight_answers(root)
    generator = OllamaGenerator(model="llama3.2:latest", cache_dir=root / "outputs/models/stage3/answers",
                                seed=42, temperature=0, max_new_tokens=192, num_ctx=4096)
    if generator.model_identity().get("digest") is None:
        raise RuntimeError("Local llama3.2 model identity unavailable; start Ollama before answer evaluation")
    answers = {}
    for variant, retrieval in runs.items():
        answers[variant] = {}
        for number, qa in enumerate(sample.to_dict("records"), 1):
            answer = generator.answer(qa["question"], retrieval[qa["qa_id"]], question_category=qa["category"])
            answers[variant][qa["qa_id"]] = answer
            if number % 10 == 0:
                print(f"Local answers {variant}: {number}/{len(sample)}; latest={answer['status']}", flush=True)
                write_json(output / "answer_runs.json", answers)
        write_json(output / "answer_runs.json", answers)
    if run_judge:
        judge_sample = read_jsonl(root / "data/processed/stage3/judge_evaluation_sample.jsonl")
        judge = OllamaGenerator(model="mistral:latest", cache_dir=root / "outputs/models/stage3/judges",
                                seed=42, temperature=0, max_new_tokens=256, num_ctx=4096)
        if judge.model_identity().get("digest") is None:
            raise RuntimeError("Cached local Mistral judge model unavailable")
        for variant, retrieval in runs.items():
            for number, qa in enumerate(judge_sample, 1):
                answer = answers[variant][qa["qa_id"]]
                answer["judge"] = judge_answer(qa, answer, retrieval[qa["qa_id"]], judge)
                print(f"Local AI judge {variant}: {number}/{len(judge_sample)}; {answer['judge']['status']}", flush=True)
                write_json(output / "answer_runs.json", answers)
    per_question, summary = evaluate_answers(sample, answers)
    per_question.to_csv(output / "answer_per_question.csv", index=False)
    summary.to_csv(output / "answer_summary.csv", index=False)
    from .data import export_answer_review
    export_answer_review(root, sample, answers)
    judge_counts = {variant: sum(row.get("judge", {}).get("status") == "ok" for row in current.values())
                    for variant, current in answers.items()}
    write_json(output / "answer_manifest.json", {"generator": generator.model_identity(), "generation_config": generator.config,
        "questions_per_variant": len(sample), "variants": list(answers), "judge_successes": judge_counts,
        "qa_generator": "ChatGPT / Codex", "different_qa_and_answer_models": True,
        "qa_label_status": qa_manifest["evaluation_status"],
        "answer_sample_sha256": sha256_file(root / "data/processed/stage3/answer_evaluation_sample.jsonl"),
        "retrieval_runs_sha256": sha256_file(output / "test_retrieval_runs.json"),
        "answer_runs_sha256": sha256_file(output / "answer_runs.json"),
        "answer_per_question_sha256": sha256_file(output / "answer_per_question.csv"),
        "answer_summary_sha256": sha256_file(output / "answer_summary.csv"),
        "generation_code_sha256": sha256_file(root / "src/stage3_rag/generation.py"),
        "evaluation_code_sha256": sha256_file(root / "src/stage3_rag/evaluation.py")})
    return answers, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-index", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    if arguments.prepare_index or arguments.retrieval:
        chunks, passages, graph, entities, manifest = prepare_data(arguments.root)
        index = build_index(arguments.root, chunks)
        print(json.dumps({"articles": manifest["source_articles"], "chunks": len(chunks),
                          "passages": len(passages), "index": index.cache_fingerprint,
                          "coverage": index.coverage_report}, indent=2), flush=True)
        if arguments.retrieval:
            qas, review, qa_status = prepare_qa_dataset(arguments.root, passages)
            run_retrieval_experiments(arguments.root, index, qas, graph, entities)
    if arguments.answers:
        run_answer_experiments(arguments.root, run_judge=arguments.judge)


if __name__ == "__main__":
    main()
