"""AI corrections remain source-bound and cannot become human approvals."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.stage3_rag.common import write_jsonl
from src.stage3_rag.qa_review import apply_ai_review, review_input_fingerprint


class AIReviewTests(unittest.TestCase):
    def setUp(self):
        self.qa = {"qa_id": "q1", "passage_id": "p1", "record_id": 1, "split": "test",
                   "category": "factual", "question": "What oversight is required?",
                   "reference_answer": "Oversight.", "gold_evidence": "oversight"}
        self.source = {"passage_id": "p1", "record_id": 1, "source_fingerprint": "s1",
                       "text": "AI governance requires human oversight.", "char_start": 0, "char_end": 38,
                       "date": "2025-01-01", "title": "AI governance"}
        self.review = {"qa_id": "q1", "passage_id": "p1", "record_id": 1,
                       "review_method": "ai_assisted", "reviewer": "ChatGPT / Codex", "reviewed_at": "2026-11-06",
                       "input_fingerprint": review_input_fingerprint(self.qa, self.source),
                       "verdict": "revised", "clarity_ok": True, "answer_supported": True,
                       "evidence_sufficient": False, "category_ok": True,
                       "issues": ["incomplete_evidence"], "rationale": "The full quote specifies human oversight.",
                       "supporting_quote": "human oversight", "proposed_gold_evidence": "human oversight",
                       "proposed_reference_answer": "Human oversight."}

    def apply(self, root, review=None, qa=None, source=None):
        path = root / "qa_ai_review.jsonl"
        write_jsonl(path, [review or self.review])
        return apply_ai_review(pd.DataFrame([qa or self.qa]), pd.DataFrame([source or self.source]),
                               path, ["factual", "temporal"])[0]

    def test_correction_is_applied_without_altering_raw_draft_or_human_status(self):
        original = deepcopy(self.qa)
        with tempfile.TemporaryDirectory() as folder:
            result = self.apply(Path(folder)).iloc[0]
        self.assertEqual(result.reference_answer, "Human oversight.")
        self.assertEqual(result.gold_evidence, "human oversight")
        self.assertEqual(result.qa_review_method, "ai_assisted")
        self.assertNotIn("human_review_verdict", result)
        self.assertEqual(self.qa, original)

    def test_changed_source_or_draft_cannot_reuse_review(self):
        for field in ("question", "text"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                qa, source = self.qa.copy(), self.source.copy()
                (qa if field == "question" else source)[field] += " changed"
                with self.assertRaisesRegex(ValueError, "Stale"):
                    self.apply(Path(folder), qa=qa, source=source)

    def test_optional_history_columns_do_not_invalidate_the_same_reviewed_content(self):
        expanded = {**self.qa, "question_before_ai_audit": float("nan")}
        self.assertEqual(review_input_fingerprint(self.qa, self.source),
                         review_input_fingerprint(expanded, self.source))

    def test_partial_or_duplicate_reviews_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.jsonl"
            for rows in ([], [self.review, self.review]):
                write_jsonl(path, rows)
                with self.assertRaisesRegex(ValueError, "every draft exactly once"):
                    apply_ai_review(pd.DataFrame([self.qa]), pd.DataFrame([self.source]), path, ["factual"])

    def test_invented_review_or_replacement_quote_is_rejected(self):
        for field in ("supporting_quote", "proposed_gold_evidence"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                review = {**self.review, field: "An invented quotation"}
                with self.assertRaisesRegex(ValueError, "source"):
                    self.apply(Path(folder), review=review)

    def test_unresolved_review_is_not_evaluation_eligible(self):
        review = {key: value for key, value in self.review.items() if not key.startswith("proposed_")}
        review["verdict"] = "unresolved"
        with tempfile.TemporaryDirectory() as folder:
            result = self.apply(Path(folder), review=review).iloc[0]
        self.assertEqual(result.qa_review_verdict, "uncertain")

    def test_supported_cannot_conceal_failed_original_checks(self):
        review = {key: value for key, value in self.review.items() if not key.startswith("proposed_")}
        review["verdict"] = "supported"
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "conceal"):
                self.apply(Path(folder), review=review)

    def test_ai_record_cannot_claim_human_review_method(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "AI attribution"):
                self.apply(Path(folder), review={**self.review, "review_method": "human"})


if __name__ == "__main__":
    unittest.main()
