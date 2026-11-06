"""Source offsets, document-level holdouts and genuine human-review persistence."""

from pathlib import Path
import re
import tempfile
import unittest

import pandas as pd

from src.stage3_rag.common import read_jsonl, write_jsonl
from src.stage3_rag.data import chunk_articles, select_passages, validate_qas, prepare_qa_dataset


class WordTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        output = {"input_ids": list(range(len(spans) + (2 if kwargs.get("add_special_tokens") else 0)))}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = spans
        return output


def documents():
    return pd.DataFrame([{
        "record_id": i, "title": f"AI governance source {i}", "date": "2025-01-01",
        "domain": "example", "url": f"https://example.test/{i}",
        "content_text": ("AI governance requires human oversight for each automated decision. " * 50),
        "source_fingerprint": f"source-{i}", "article_review_status": "relevant", "article_review_method": "ai_assisted",
        "security": i % 2 == 0, "accountability": i % 2 == 1, "bias_fairness": False, "regulation": False,
    } for i in range(8)])


class DataTests(unittest.TestCase):
    def test_chunks_are_exact_spans_and_cover_source_tokens(self):
        docs = documents().head(1)
        chunks = chunk_articles(docs, WordTokenizer(), max_tokens=80, overlap_tokens=10)
        source = docs.iloc[0].content_text
        covered = set()
        for row in chunks.to_dict("records"):
            self.assertEqual(source[row["char_start"]:row["char_end"]], row["text"])
            self.assertLessEqual(row["embedding_tokens"], 256)
            covered.update(range(row["char_start"], row["char_end"]))
        self.assertTrue(all(i in covered for i, c in enumerate(source) if not c.isspace()))
        again = chunk_articles(docs, WordTokenizer(), max_tokens=80, overlap_tokens=10)
        self.assertEqual(chunks.chunk_id.tolist(), again.chunk_id.tolist())

    def test_question_passages_split_by_article_before_question_generation(self):
        docs = documents()
        reviews = pd.DataFrame([{"record_id": i, "review_evidence": "AI governance", "review_method": "ai_assisted",
                                 "primary_theme": "security" if i % 2 == 0 else "accountability"} for i in range(8)])
        chunks = chunk_articles(docs, WordTokenizer(), max_tokens=80, overlap_tokens=10)
        passages = select_passages(chunks, reviews, docs, count=10)
        self.assertEqual(len(passages), 10)
        self.assertEqual(passages.groupby("record_id").split.nunique().max(), 1)
        self.assertTrue(passages.passage_id.is_unique)

    def _drafts(self, root):
        passages, qas = [], []
        for i in range(120):
            passage = {"passage_id": f"p{i}", "chunk_id": f"p{i}", "record_id": i,
                       "date": "2025-01-01", "title": "AI oversight", "domain": "example",
                       "url": "https://example.test/", "text": "AI governance requires human oversight.",
                       "char_start": 0, "char_end": 38, "source_fingerprint": f"s{i}",
                       "split": "development" if i < 30 else "test"}
            passages.append(passage)
            for j in range(2):
                qas.append({"qa_id": f"q{i}_{j}", "passage_id": f"p{i}", "record_id": i,
                            "question": f"Which oversight is described in source {i}, question {j}?",
                            "reference_answer": "Human oversight.", "category": "factual",
                            "gold_chunk_ids": [f"p{i}"], "gold_evidence": "AI governance",
                            "generation_model": "ChatGPT / Codex", "qa_review_method": "ai_assisted",
                            "qa_review_verdict": "supported", "split": passage["split"]})
        path = root / "data/processed/stage3/qa_drafts_codex.jsonl"
        write_jsonl(path, qas)
        return pd.DataFrame(passages), path

    def test_exact_quotes_do_not_fabricate_human_approval_and_edits_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _ = self._drafts(root)
            qas, review, status = prepare_qa_dataset(root, passages)
            self.assertFalse(status["assignment_manual_review_complete"])
            self.assertEqual(status["human_pending"], 240)
            path = root / "data/processed/stage3/qa_manual_review.csv"
            review.loc[0, ["human_review_verdict", "human_reviewer", "human_reviewed_at", "edited_question"]] = [
                "approved", "Student reviewer", "2026-11-05", "What human oversight is required?"]
            review.to_csv(path, index=False)
            qas, again, _ = prepare_qa_dataset(root, passages)
            self.assertEqual(qas.loc[0, "question"], "What human oversight is required?")
            self.assertEqual(qas.loc[0, "qa_review_method"], "human")
            self.assertEqual(again.loc[0, "human_reviewer"], "Student reviewer")
            packet = (root / "outputs/results/stage3/qa_review_packet.html").read_text()
            first_card = packet.split("<article>")[1].split("</article>")[0]
            self.assertIn("<h3>What human oversight is required?</h3>", first_card)
            self.assertIn("Recorded review: approved", first_card)
            self.assertNotIn("<h3>Which oversight is described in source 0, question 0?</h3>", first_card)

    def test_changed_draft_cannot_reuse_human_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, draft_path = self._drafts(root)
            _, review, _ = prepare_qa_dataset(root, passages)
            review.loc[0, ["human_review_verdict", "human_reviewer", "human_reviewed_at"]] = [
                "approved", "Student reviewer", "2026-11-05"]
            review.to_csv(root / "data/processed/stage3/qa_manual_review.csv", index=False)
            rows = read_jsonl(draft_path); rows[0]["question"] = "Changed source question?"
            write_jsonl(draft_path, rows)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                prepare_qa_dataset(root, passages)

    def test_ai_identity_is_not_human_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, _ = self._drafts(root)
            _, review, _ = prepare_qa_dataset(root, passages)
            review.loc[0, ["human_review_verdict", "human_reviewer", "human_reviewed_at"]] = [
                "approved", "ChatGPT / Codex", "2026-11-05"]
            review.to_csv(root / "data/processed/stage3/qa_manual_review.csv", index=False)
            with self.assertRaisesRegex(ValueError, "cannot be recorded as human"):
                prepare_qa_dataset(root, passages)

    def test_invented_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, draft_path = self._drafts(root)
            qas = pd.DataFrame(read_jsonl(draft_path))
            qas.loc[0, "gold_evidence"] = "An invented quotation"
            with self.assertRaisesRegex(ValueError, "Nonliteral"):
                validate_qas(qas, passages)

    def test_ai_corrections_stay_separate_from_preserved_human_edits(self):
        from src.stage3_rag.qa_review import review_input_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            passages, draft_path = self._drafts(root)
            sources = passages.set_index("passage_id").to_dict("index")
            reviews = []
            for row in read_jsonl(draft_path):
                source = {"passage_id": row["passage_id"], **sources[row["passage_id"]]}
                reviews.append({"qa_id": row["qa_id"], "passage_id": row["passage_id"], "record_id": row["record_id"],
                    "input_fingerprint": review_input_fingerprint(row, source), "review_method": "ai_assisted",
                    "reviewer": "ChatGPT / Codex", "reviewed_at": "2026-11-06", "verdict": "supported",
                    "clarity_ok": True, "answer_supported": True, "evidence_sufficient": True, "category_ok": True,
                    "issues": [], "rationale": "The original passage explicitly requires human oversight.",
                    "supporting_quote": "human oversight"})
            reviews[0].update(verdict="revised", proposed_reference_answer="Human oversight is required.")
            write_jsonl(root / "data/processed/stage3/qa_ai_review.jsonl", reviews)
            qas, review, status = prepare_qa_dataset(root, passages)
            self.assertEqual(status["ai_reviewed"], 240)
            self.assertEqual(status["ai_review_revised"], 1)
            self.assertEqual(status["human_approved"], 0)
            self.assertEqual(qas.loc[0, "label_status"], "ai_reviewed_pending_human_review")
            review.loc[0, ["human_review_verdict", "human_reviewer", "human_reviewed_at", "edited_reference_answer"]] = [
                "approved", "Student reviewer", "2026-11-06", "A human must oversee the system."]
            review.to_csv(root / "data/processed/stage3/qa_manual_review.csv", index=False)
            qas, preserved, status = prepare_qa_dataset(root, passages)
            self.assertEqual(qas.loc[0, "reference_answer"], "A human must oversee the system.")
            self.assertEqual(qas.loc[0, "qa_review_method"], "human")
            self.assertEqual(preserved.loc[0, "human_reviewer"], "Student reviewer")
            self.assertFalse(status["assignment_manual_review_complete"])


if __name__ == "__main__":
    unittest.main()
