"""Meaningful checks for sampling design, independent reviews and source drift."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.stage1_validation.articles import (
    MAIN_GROUP, CANDIDATE_GROUP, apply_article_review,
    prepare_article_review, summarize_article_review,
)


def example_corpus():
    rows = []
    for i in range(80):
        tagged = i < 64
        theme = (i // 16) % 4
        rows.append({"record_id": i, "title": f"Article {i}",
                     "date": pd.Timestamp("2024-10-01") if i % 2 else pd.Timestamp("2025-05-01"),
                     "domain": f"source{i % 3}", "url": f"https://example.test/{i}",
                     "content_text": f"AI fairness and regulation source evidence {i}.",
                     "tags": ["weak-tag"], "governance_related": tagged,
                     "security": tagged and theme == 0,
                     "accountability": tagged and theme == 1,
                     "bias_fairness": tagged and theme == 2,
                     "regulation": tagged and theme == 3})
    # This source must never leak into the official time-window population.
    rows.append({**rows[0], "record_id": 1000, "date": pd.Timestamp("2026-09-01")})
    return pd.DataFrame(rows)


class ArticleValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "article_review.csv"
        self.corpus = example_corpus()
        self.topics = pd.DataFrame({"record_id": self.corpus["record_id"],
                                    "dominant_topic": self.corpus["record_id"] % 2,
                                    "dominant_topic_label": np.where(self.corpus["record_id"] % 2, "Broad marketing", "Regulation")})
        self.candidates = pd.DataFrame({"record_id": range(64, 80)})

    def prepare(self, **kwargs):
        review, sampling, coverage = prepare_article_review(
            self.corpus, self.topics, self.candidates,
            self.path, sample_size=24, candidate_sample_size=4, **kwargs)
        review["review_method"] = "human"
        return review, sampling, coverage

    def test_stratified_design_and_input_order_independence(self):
        first, sampling, coverage = self.prepare()
        other_path = self.path.with_name("shuffled.csv")
        shuffled, _, _ = prepare_article_review(
            self.corpus.sample(frac=1, random_state=1), self.topics, self.candidates,
            other_path, sample_size=24, candidate_sample_size=4)
        self.assertEqual(first.article_review_id.tolist(), shuffled.article_review_id.tolist())
        self.assertNotIn(1000, set(first.record_id))
        main = first[first.sample_group.eq(MAIN_GROUP)]
        self.assertEqual(len(main), 24)
        self.assertEqual(main.stratum.nunique(), 8)
        self.assertAlmostEqual(main.sample_weight.sum(), 64)
        main_sampling = sampling[sampling.sample_group.eq(MAIN_GROUP)]
        self.assertTrue((main_sampling.sample_size >= 1).all())
        self.assertTrue((main_sampling.inclusion_probability > 0).all())
        self.assertTrue({"month", "source", "nmf_topic", "theme"}.issubset(set(coverage.dimension)))

    def test_pending_reviews_are_not_accuracy_results(self):
        review, _, _ = self.prepare()
        summary, topics = summarize_article_review(review)
        main = summary.set_index("sample_group").loc[MAIN_GROUP]
        self.assertEqual(main.pending, 24)
        self.assertFalse(main.main_sample_complete)
        self.assertTrue(pd.isna(main.weighted_precision))
        self.assertTrue(topics.weighted_precision.isna().all())

    def test_annotations_survive_rerun_and_support_weighted_precision(self):
        review, _, _ = self.prepare()
        main = review.sample_group.eq(MAIN_GROUP)
        review.loc[main, "relevance_verdict"] = "relevant"
        review.loc[main, "reviewer"] = "Human reviewer"
        review.loc[main, "review_evidence"] = "AI fairness and regulation"
        review.to_csv(self.path, index=False)
        again, _, _ = self.prepare()
        self.assertEqual(again.loc[main, "reviewer"].tolist(), ["Human reviewer"] * 24)
        summary, _ = summarize_article_review(again)
        main_row = summary.set_index("sample_group").loc[MAIN_GROUP]
        self.assertTrue(main_row.main_sample_complete)
        self.assertAlmostEqual(main_row.weighted_precision, 1.0)
        candidate_row = summary.set_index("sample_group").loc[CANDIDATE_GROUP]
        self.assertTrue(pd.isna(candidate_row.weighted_precision))

    def test_invalid_human_quote_and_missing_reviewer_are_rejected(self):
        review, _, _ = self.prepare()
        review.loc[0, "relevance_verdict"] = "relevant"
        with self.assertRaisesRegex(ValueError, "reviewer"):
            summarize_article_review(review)
        review.loc[0, "reviewer"] = "Human reviewer"
        review.loc[0, "review_evidence"] = "An invented quotation"
        with self.assertRaisesRegex(ValueError, "exact source quote"):
            summarize_article_review(review)

    def test_irrelevant_articles_removed_candidates_not_promoted_source_change_rejected(self):
        review, _, _ = self.prepare()
        main_index = review.index[review.sample_group.eq(MAIN_GROUP)][0]
        candidate_index = review.index[review.sample_group.eq(CANDIDATE_GROUP)][0]
        rejected_id = review.loc[main_index, "record_id"]
        candidate_id = review.loc[candidate_index, "record_id"]
        review.loc[[main_index, candidate_index], "relevance_verdict"] = ["irrelevant", "relevant"]
        review.loc[[main_index, candidate_index], "reviewer"] = "Human reviewer"
        review.loc[main_index, "review_notes"] = "Generic technology content; no substantive Governance discussion."
        review.loc[candidate_index, "review_evidence"] = "AI fairness and regulation"
        eligible = apply_article_review(self.corpus, review)
        self.assertNotIn(rejected_id, set(eligible.record_id))
        self.assertNotIn(candidate_id, set(eligible.record_id))
        self.assertNotIn(1000, set(eligible.record_id))
        changed = self.corpus.copy()
        changed.loc[changed.record_id.eq(rejected_id), "content_text"] = "Changed source."
        with self.assertRaisesRegex(ValueError, "stale source_fingerprint"):
            apply_article_review(changed, review)

    def test_uncertain_is_reviewed_but_not_resolved(self):
        review, _, _ = self.prepare()
        main_index = review.index[review.sample_group.eq(MAIN_GROUP)][0]
        review.loc[main_index, "relevance_verdict"] = "uncertain"
        review.loc[main_index, "reviewer"] = "Human reviewer"
        review.loc[main_index, "review_notes"] = "Governance is mentioned, but its substantive role is unclear."
        summary, _ = summarize_article_review(review)
        main = summary.set_index("sample_group").loc[MAIN_GROUP]
        self.assertEqual(main.reviewed, 1)
        self.assertEqual(main.resolved, 0)
        self.assertEqual(main.uncertain, 1)
        self.assertFalse(main.main_sample_complete)
        self.assertTrue(pd.isna(main.weighted_precision))

    def test_irrelevant_or_uncertain_without_rationale_is_not_validated(self):
        review, _, _ = self.prepare()
        review.loc[0, "relevance_verdict"] = "irrelevant"
        review.loc[0, "reviewer"] = "Human reviewer"
        with self.assertRaisesRegex(ValueError, "review_notes"):
            summarize_article_review(review)
        review.loc[0, "relevance_verdict"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "review_notes"):
            summarize_article_review(review)

    def test_ai_reviews_cannot_count_as_human_validation(self):
        review, _, _ = self.prepare()
        main = review.sample_group.eq(MAIN_GROUP)
        review.loc[main, "relevance_verdict"] = "relevant"
        review.loc[main, "review_method"] = "ai_assisted"
        review.loc[main, "reviewer"] = "ChatGPT / Codex"
        review.loc[main, "reviewed_at"] = "2026-11-05"
        review.loc[main, "review_evidence"] = "AI fairness and regulation"
        summary, _ = summarize_article_review(review)
        row = summary.set_index("sample_group").loc[MAIN_GROUP]
        self.assertTrue(row.main_sample_complete)
        self.assertFalse(row.main_human_sample_complete)
        self.assertEqual(row.ai_reviewed, 24)
        self.assertEqual(row.human_reviewed, 0)

    def test_uncertain_label_bounds_are_not_a_complete_precision_estimate(self):
        review, _, _ = self.prepare()
        main = review.sample_group.eq(MAIN_GROUP)
        review.loc[main, "relevance_verdict"] = "relevant"
        review.loc[main, "reviewer"] = "Human reviewer"
        review.loc[main, "review_evidence"] = "AI fairness and regulation"
        uncertain_index = review.index[main][0]
        review.loc[uncertain_index, "relevance_verdict"] = "uncertain"
        review.loc[uncertain_index, "review_notes"] = "Scope remains unclear."
        summary, _ = summarize_article_review(review)
        row = summary.set_index("sample_group").loc[MAIN_GROUP]
        expected_min = 1 - review.loc[uncertain_index, "sample_weight"] / review.loc[main, "sample_weight"].sum()
        self.assertAlmostEqual(row.weighted_relevance_min, expected_min)
        self.assertAlmostEqual(row.weighted_relevance_max, 1.0)
        self.assertTrue(pd.isna(row.weighted_precision))


if __name__ == "__main__":
    unittest.main()
