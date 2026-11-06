import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.stage1_validation.common import merge_review_template, source_fingerprint


class ReviewPersistenceTests(unittest.TestCase):
    def test_preserves_annotations_but_refreshes_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            frame = pd.DataFrame([{"id": "a", "source_fingerprint": "v1", "title": "old"}])
            merge_review_template(frame, path, key="id", editable_fields=["verdict", "notes"])
            saved = pd.read_csv(path, keep_default_na=False)
            saved.loc[0, ["verdict", "notes"]] = ["supported", "Line one\nLine two, with commas"]
            saved.to_csv(path, index=False)
            frame["title"] = "refreshed metadata"
            result = merge_review_template(frame, path, key="id", editable_fields=["verdict", "notes"])
            self.assertEqual(result.loc[0, "title"], "refreshed metadata")
            self.assertEqual(result.loc[0, "notes"], "Line one\nLine two, with commas")
            original = path.read_bytes()
            frame["source_fingerprint"] = "v2"
            with self.assertRaisesRegex(ValueError, "Source changed"):
                merge_review_template(frame, path, key="id", editable_fields=["verdict", "notes"])
            self.assertEqual(original, path.read_bytes())

    def test_resampling_cannot_discard_an_edited_row(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            old = pd.DataFrame([{"id": "old", "source_fingerprint": "v1", "verdict": "supported"}])
            old.to_csv(path, index=False)
            fresh = pd.DataFrame([{"id": "new", "source_fingerprint": "v2"}])
            with self.assertRaisesRegex(ValueError, "left the sample"):
                merge_review_template(fresh, path, key="id", editable_fields=["verdict"])

    def test_fingerprint_changes_with_the_source(self):
        row = dict(record_id=1, date="2026-01-01", title="Title", content_text="Text")
        first = source_fingerprint(row)
        row["content_text"] = "Changed text"
        self.assertNotEqual(first, source_fingerprint(row))

    def test_legacy_human_reviews_are_migrated_without_losing_attribution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            pd.DataFrame([{"id": "a", "source_fingerprint": "v1", "human_relevance": "relevant"}]).to_csv(path, index=False)
            fresh = pd.DataFrame([{"id": "a", "source_fingerprint": "v1"}])
            result = merge_review_template(fresh, path, key="id", editable_fields=["relevance_verdict", "review_method"])
            self.assertEqual(result.loc[0, "relevance_verdict"], "relevant")
            self.assertEqual(result.loc[0, "review_method"], "human")

    def test_known_legacy_ai_reviewer_cannot_be_migrated_as_human(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            pd.DataFrame([{"id": "a", "source_fingerprint": "v1", "human_relevance": "relevant",
                           "reviewer": "ChatGPT / Codex"}]).to_csv(path, index=False)
            fresh = pd.DataFrame([{"id": "a", "source_fingerprint": "v1"}])
            result = merge_review_template(fresh, path, key="id", editable_fields=["relevance_verdict", "review_method", "reviewer"])
            self.assertEqual(result.loc[0, "review_method"], "ai_assisted")


if __name__ == "__main__":
    unittest.main()
