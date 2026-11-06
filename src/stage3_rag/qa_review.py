"""Source-bound AI review of immutable Q&A drafts, separate from human approval."""

from pathlib import Path

import pandas as pd

from .common import fingerprint, read_jsonl


EDIT_FIELDS = ("question", "reference_answer", "gold_evidence", "category")
CHECK_FIELDS = ("clarity_ok", "answer_supported", "evidence_sufficient", "category_ok")


def review_input_fingerprint(qa, passage):
    # Pandas adds nulls for optional historical notes present on only some rows.
    # Bind the actual reviewed draft fields, independent of those extra columns.
    draft = {key: qa.get(key) for key in ("qa_id", "passage_id", "record_id", "split", "category",
             "question", "reference_answer", "gold_evidence", "gold_chunk_ids", "generation_model")}
    return fingerprint({"draft": draft, "passage": {key: passage.get(key) for key in (
        "passage_id", "record_id", "source_fingerprint", "char_start", "char_end", "text", "date", "title")}})


def apply_ai_review(drafts, passages, path, categories):
    """Apply only explicit, complete, source-matching AI corrections.

    A review cannot change article identity, split or human-review fields.
    Unresolved rows remain visible but are excluded from evaluation eligibility.
    """
    path = Path(path)
    if not path.exists():
        return drafts.copy(), []
    reviews = read_jsonl(path)
    by_id = {row["qa_id"]: row for row in reviews}
    if len(by_id) != len(reviews) or set(by_id) != set(drafts.qa_id):
        raise ValueError("AI Q&A review must cover every draft exactly once")
    sources = passages.set_index("passage_id").to_dict("index")
    output = []
    for original in drafts.to_dict("records"):
        row = original.copy()
        review = by_id[row["qa_id"]]
        source = {"passage_id": row["passage_id"], **sources[row["passage_id"]]}
        if review.get("input_fingerprint") != review_input_fingerprint(original, source):
            raise ValueError(f"Stale AI Q&A review for {row['qa_id']}: draft/source changed")
        if (review.get("review_method") != "ai_assisted" or not str(review.get("reviewer", "")).strip()
                or pd.isna(pd.to_datetime(review.get("reviewed_at"), errors="coerce"))):
            raise ValueError("AI Q&A review requires its actual AI attribution and date")
        if (review.get("passage_id") != row["passage_id"]
                or review.get("record_id") != row["record_id"]):
            raise ValueError("AI Q&A review source identity mismatch")
        if (any(type(review.get(field)) is not bool for field in CHECK_FIELDS)
                or not isinstance(review.get("issues"), list)
                or not str(review.get("rationale", "")).strip()):
            raise ValueError("AI Q&A review requires explicit original-draft checks and rationale")
        quote = review.get("supporting_quote")
        if not isinstance(quote, str) or not quote or quote not in source["text"]:
            raise ValueError("AI Q&A review evidence must be literal source text")
        verdict = review.get("verdict")
        edits = {field: review.get("proposed_" + field) for field in EDIT_FIELDS
                 if review.get("proposed_" + field) is not None}
        if verdict not in {"supported", "revised", "unresolved"}:
            raise ValueError("Invalid AI Q&A review verdict")
        if verdict == "revised":
            if not edits or not any(value != row[field] for field, value in edits.items()):
                raise ValueError("A revised Q&A verdict requires an actual correction")
            if any(not isinstance(value, str) or not value.strip() for value in edits.values()):
                raise ValueError("AI Q&A corrections must be nonempty strings")
            row.update(edits)
        elif edits:
            raise ValueError("Only a revised AI Q&A verdict may contain corrections")
        if verdict == "supported" and not all(review[field] for field in CHECK_FIELDS):
            raise ValueError("A supported AI verdict cannot conceal a failed original-draft check")
        if row["category"] not in categories or row["gold_evidence"] not in source["text"]:
            raise ValueError("Invalid corrected Q&A category or source quote")
        row.update(ai_review_verdict=verdict, ai_review_rationale=review["rationale"],
                   ai_reviewer=review["reviewer"], ai_reviewed_at=review["reviewed_at"],
                   qa_review_method="ai_assisted",
                   qa_review_verdict="uncertain" if verdict == "unresolved" else "supported")
        output.append(row)
    return pd.DataFrame(output), reviews
