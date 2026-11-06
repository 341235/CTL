"""Failures happen before model calls, using tiny file-only benchmark snapshots."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.stage3_rag.common import fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from src.stage3_rag.data import CATEGORIES, answer_evaluation_sample
from src.stage3_rag.snapshot import preflight_answers


class SnapshotFixture:
    def __init__(self, root):
        self.root = root
        self.data = root / "data/processed/stage3"
        self.output = root / "outputs/results/stage3"
        self.handoff_path = root / "outputs/validation/stage1/stage3_handoff.json"
        self.graph_path = root / "outputs/validation/stage1/audited_retrieval_graph.graphml"
        self.corpus_path = root / "data/processed/corpus/ai_media_clean.pkl.gz"
        self.review_path = root / "data/processed/validation/stage1/article_review.csv"
        for path in (self.graph_path, self.corpus_path, self.review_path,
                     self.data / "qa_manual_review.csv", root / "src/stage3_rag/retrieval.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        passages, qas = [], []
        for category_index, category in enumerate(CATEGORIES):
            for number in range(16):
                qa_id = f"q{category_index}_{number:02d}"
                record_id = category_index * 100 + number
                split = "development" if number == 0 else "test"
                passages.append({"passage_id": qa_id, "record_id": record_id, "split": split,
                                 "text": "AI governance requires human oversight."})
                qas.append({"qa_id": qa_id, "passage_id": qa_id, "record_id": record_id,
                            "split": split, "category": category, "question": f"What does {qa_id} require?",
                            "reference_answer": "Human oversight.", "gold_chunk_ids": [qa_id],
                            "gold_evidence": "human oversight", "source_fingerprint": f"source-{qa_id}",
                            "generation_model": "ChatGPT / Codex", "qa_review_verdict": "supported",
                            "label_status": "ai_draft_pending_human_review", "optional_notes": None})
        frame = pd.DataFrame(qas)
        write_jsonl(self.data / "selected_passages.jsonl", passages)
        write_jsonl(self.data / "qa_dataset.jsonl", frame.to_dict("records"))
        write_jsonl(self.data / "qa_drafts_codex.jsonl", frame.to_dict("records"))
        sample = answer_evaluation_sample(frame)
        judges = pd.concat([group.sort_values("qa_id").sample(n=3, random_state=43)
                            for _, group in sample.groupby("category")]).sort_values(["category", "qa_id"])
        write_jsonl(self.data / "answer_evaluation_sample.jsonl", sample.to_dict("records"))
        write_jsonl(self.data / "judge_evaluation_sample.jsonl", judges.to_dict("records"))
        test_ids = frame.loc[frame.split.eq("test"), "qa_id"].tolist()
        write_json(self.output / "test_retrieval_runs.json",
                   {variant: {qa_id: [{"chunk_id": qa_id, "text": "Source"}] for qa_id in test_ids}
                    for variant in ("text_only", "graph", "graph_rerank")})
        write_json(self.handoff_path, {"inputs_sha256": {"corpus": sha256_file(self.corpus_path)},
                   "review_sha256": {"article_review.csv": sha256_file(self.review_path)},
                   "artifacts": {"graph": str(self.graph_path.relative_to(root))}})
        self.refresh()

    def refresh(self):
        """Re-freeze fixture hashes so tests can reach semantic validation."""
        qas = pd.DataFrame(read_jsonl(self.data / "qa_dataset.jsonl"))
        write_json(self.data / "qa_dataset_manifest.json", {
            "effective_qa_fingerprint": fingerprint(qas.to_dict("records")),
            "review_sha256": sha256_file(self.data / "qa_manual_review.csv"),
            "draft_sha256": sha256_file(self.data / "qa_drafts_codex.jsonl"),
            "evaluation_status": "provisional_on_ai_drafts"})
        manifest = {"qa_fingerprint": fingerprint(qas.to_dict("records")), "chunk_cache_id": "chunks-v1"}
        paths = {
            "qa_dataset_sha256": self.data / "qa_dataset.jsonl",
            "qa_dataset_manifest_sha256": self.data / "qa_dataset_manifest.json",
            "qa_review_sha256": self.data / "qa_manual_review.csv",
            "selected_passages_sha256": self.data / "selected_passages.jsonl",
            "test_retrieval_runs_sha256": self.output / "test_retrieval_runs.json",
            "answer_sample_sha256": self.data / "answer_evaluation_sample.jsonl",
            "judge_sample_sha256": self.data / "judge_evaluation_sample.jsonl",
            "stage1_handoff_sha256": self.handoff_path,
            "graph_sha256": self.graph_path,
            "retrieval_code_sha256": self.root / "src/stage3_rag/retrieval.py",
        }
        manifest.update({key: sha256_file(path) for key, path in paths.items()})
        write_json(self.output / "retrieval_manifest.json", manifest)
        write_json(self.data / "chunk_manifest.json", {key: manifest[key] for key in
                   ("chunk_cache_id", "selected_passages_sha256", "stage1_handoff_sha256", "graph_sha256")})


class SnapshotTests(unittest.TestCase):
    def test_valid_snapshot_returns_paired_frozen_records_without_loading_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            sample, runs, manifest = preflight_answers(fixture.root)
            self.assertEqual(len(sample), 60)
            self.assertEqual(sample.category.value_counts().tolist(), [12] * 5)
            self.assertEqual(set(runs), {"text_only", "graph", "graph_rerank"})
            self.assertEqual(manifest["evaluation_status"], "provisional_on_ai_drafts")
            # The fake corpus is plain text: successful validation proves it was
            # hashed, rather than unpickled or loaded as an index.

    def test_every_hashed_experiment_input_rejects_changes(self):
        relative_paths = [
            "data/processed/stage3/qa_dataset.jsonl", "data/processed/stage3/qa_manual_review.csv",
            "data/processed/stage3/qa_dataset_manifest.json",
            "data/processed/stage3/selected_passages.jsonl", "outputs/results/stage3/test_retrieval_runs.json",
            "data/processed/stage3/answer_evaluation_sample.jsonl", "data/processed/stage3/judge_evaluation_sample.jsonl",
            "outputs/validation/stage1/stage3_handoff.json", "outputs/validation/stage1/audited_retrieval_graph.graphml",
            "src/stage3_rag/retrieval.py",
        ]
        for relative in relative_paths:
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as folder:
                fixture = SnapshotFixture(Path(folder))
                with (fixture.root / relative).open("a") as handle:
                    handle.write("\n")
                with self.assertRaisesRegex(ValueError, "changed since retrieval"):
                    preflight_answers(fixture.root)

    def test_manual_csv_change_cannot_reuse_old_effective_qa(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            (fixture.data / "qa_manual_review.csv").write_text("human edited a question\n")
            with self.assertRaisesRegex(ValueError, "qa_review_sha256"):
                preflight_answers(fixture.root)

    def test_current_qa_manifest_must_match_effective_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            path = fixture.data / "qa_dataset_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["effective_qa_fingerprint"] = "old-qa"
            write_json(path, manifest)
            retrieval_path = fixture.output / "retrieval_manifest.json"
            retrieval = json.loads(retrieval_path.read_text())
            retrieval["qa_dataset_manifest_sha256"] = sha256_file(path)
            write_json(retrieval_path, retrieval)
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                preflight_answers(fixture.root)

    def test_old_samples_cannot_be_relabelled_with_a_new_human_review_status(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            path = fixture.data / "qa_dataset_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["evaluation_status"] = "human_reviewed"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "qa_dataset_manifest_sha256"):
                preflight_answers(fixture.root)

    def test_changed_question_without_fresh_frozen_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            sample = read_jsonl(fixture.data / "answer_evaluation_sample.jsonl")
            qas = read_jsonl(fixture.data / "qa_dataset.jsonl")
            next(row for row in qas if row["qa_id"] == sample[0]["qa_id"])["question"] = "A human corrected question?"
            write_jsonl(fixture.data / "qa_dataset.jsonl", qas)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "differs from the current effective QA record"):
                preflight_answers(fixture.root)

    def test_source_article_leakage_is_rejected_even_when_hashes_are_current(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            qas = read_jsonl(fixture.data / "qa_dataset.jsonl")
            qas[1]["record_id"] = qas[0]["record_id"]
            passages = read_jsonl(fixture.data / "selected_passages.jsonl")
            passages[1]["record_id"] = passages[0]["record_id"]
            write_jsonl(fixture.data / "qa_dataset.jsonl", qas)
            write_jsonl(fixture.data / "selected_passages.jsonl", passages)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "article leakage"):
                preflight_answers(fixture.root)

    def test_answer_sample_cannot_contain_development_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            sample = read_jsonl(fixture.data / "answer_evaluation_sample.jsonl")
            sample[0]["split"] = "development"
            write_jsonl(fixture.data / "answer_evaluation_sample.jsonl", sample)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "held-out test"):
                preflight_answers(fixture.root)

    def test_resampled_judges_are_rejected_despite_valid_membership_and_balance(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            answers = read_jsonl(fixture.data / "answer_evaluation_sample.jsonl")
            judges = read_jsonl(fixture.data / "judge_evaluation_sample.jsonl")
            existing = {row["qa_id"] for row in judges}
            judges[0] = next(row for row in answers if row["category"] == judges[0]["category"]
                             and row["qa_id"] not in existing)
            judges.sort(key=lambda row: (row["category"], row["qa_id"]))
            write_jsonl(fixture.data / "judge_evaluation_sample.jsonl", judges)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "judge sample differs from the deterministic"):
                preflight_answers(fixture.root)

    def test_resampled_answers_are_rejected_despite_valid_membership_and_balance(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            answers = read_jsonl(fixture.data / "answer_evaluation_sample.jsonl")
            qas = read_jsonl(fixture.data / "qa_dataset.jsonl")
            existing = {row["qa_id"] for row in answers}
            answers[0] = next(row for row in qas if row["category"] == answers[0]["category"]
                              and row["split"] == "test" and row["qa_id"] not in existing)
            answers.sort(key=lambda row: (row["category"], row["qa_id"]))
            write_jsonl(fixture.data / "answer_evaluation_sample.jsonl", answers)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "answer sample differs from the deterministic"):
                preflight_answers(fixture.root)

    def test_variants_must_cover_the_same_complete_test_set(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            path = fixture.output / "test_retrieval_runs.json"
            runs = json.loads(path.read_text())
            runs["graph"].pop(next(iter(runs["graph"])))
            write_json(path, runs)
            fixture.refresh()
            with self.assertRaisesRegex(ValueError, "complete current held-out test set"):
                preflight_answers(fixture.root)

    def test_corpus_and_stage1_reviews_remain_bound_to_the_handoff(self):
        for field in ("corpus_path", "review_path"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                fixture = SnapshotFixture(Path(folder))
                getattr(fixture, field).write_text("changed source\n")
                with self.assertRaisesRegex(ValueError, "changed since"):
                    preflight_answers(fixture.root)

    def test_chunk_identity_must_match_prepared_source_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            path = fixture.data / "chunk_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["chunk_cache_id"] = "another-index-source"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "chunk_cache_id"):
                preflight_answers(fixture.root)

    def test_changed_ai_review_cannot_reuse_the_previous_effective_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = SnapshotFixture(Path(folder))
            path = fixture.data / "qa_ai_review.jsonl"
            path.write_text('{"review": "source-bound"}\n')
            manifest_path = fixture.data / "qa_dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["ai_review_sha256"] = sha256_file(path)
            write_json(manifest_path, manifest)
            retrieval_path = fixture.output / "retrieval_manifest.json"
            retrieval = json.loads(retrieval_path.read_text())
            retrieval["qa_dataset_manifest_sha256"] = sha256_file(manifest_path)
            write_json(retrieval_path, retrieval)
            preflight_answers(fixture.root)
            path.write_text('{"review": "changed"}\n')
            with self.assertRaisesRegex(ValueError, "AI Q&A review changed"):
                preflight_answers(fixture.root)


if __name__ == "__main__":
    unittest.main()
