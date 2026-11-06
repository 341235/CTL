"""Stage 3 source snapshots, exact chunks and evidence-passage selection.

The selection ranks passages for the Q&A set; it does not redefine Stage 1's
corpus-wide tag filter. Unreviewed retrieval documents retain weak-label status.
"""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import fingerprint, is_ai_reviewer, read_jsonl, sha256_file, stable_id, write_json, write_jsonl

THEMES = ["security", "accountability", "bias_fairness", "regulation"]
CATEGORIES = ["factual", "temporal", "relational", "analytical", "comparative"]
HUMAN_QA_FIELDS = ["human_review_verdict", "human_reviewer", "human_reviewed_at", "human_review_notes",
                   "edited_question", "edited_reference_answer", "edited_gold_evidence"]
GOVERNANCE_TERMS = re.compile(
    r"\b(?:governance|accountab\w*|bias(?:es|ed)?|fairness|discriminat\w*|"
    r"transparen\w*|explainab\w*|interpretabil\w*|human oversight|human review|"
    r"human.in.the.loop|regulat\w*|legislat\w*|AI Act|GDPR|privacy|consent|"
    r"safeguard\w*|safety|prompt injection|red.team\w*|audit\w*|compliance|"
    r"responsible AI|risk management|data protection|misinformation|deepfake\w*)\b", re.I)


def load_sources(project_root):
    root = Path(project_root)
    handoff_path = root / "outputs/validation/stage1/stage3_handoff.json"
    handoff = json.loads(handoff_path.read_text())
    corpus_path = root / "data/processed/corpus/ai_media_clean.pkl.gz"
    if sha256_file(corpus_path) != handoff["inputs_sha256"]["corpus"]:
        raise ValueError("Corpus changed since Stage 1 validation; rerun the validation notebook first")
    for name, expected in handoff["review_sha256"].items():
        if sha256_file(root / "data/processed/validation/stage1" / name) != expected:
            raise ValueError(f"{name} changed since the Stage 1 handoff; rerun validation first")
    corpus = pd.read_pickle(corpus_path)
    eligible = pd.read_csv(root / handoff["artifacts"]["eligible_articles"])
    # Borderline sources are not used as Stage 3 benchmark passages or retrieval
    # documents. This applies the recorded verdict rather than a new classifier.
    eligible = eligible.loc[~eligible.article_review_status.isin(["irrelevant", "uncertain"])].copy()
    columns = ["record_id", "article_review_status", "article_review_method", "source_fingerprint"]
    documents = corpus.merge(eligible[columns], on="record_id", how="inner", validate="one_to_one")
    documents["date"] = pd.to_datetime(documents["date"])
    if not documents.record_id.is_unique or not documents.date.between("2024-09-01", "2026-08-31").all():
        raise ValueError("Invalid article IDs/publication scope in Stage 3 source documents")
    reviews = pd.read_csv(root / "data/processed/validation/stage1/article_review.csv", keep_default_na=False)
    reviews = reviews.loc[reviews.sample_group.eq("tagged_stratified") & reviews.relevance_verdict.eq("relevant")].copy()
    return documents.sort_values("record_id").reset_index(drop=True), reviews, handoff


def chunk_articles(documents, tokenizer, max_tokens=200, overlap_tokens=32):
    """Preserve exact original-text spans using actual fast-tokenizer offsets.

    Dense inputs contain at most 32 title wordpieces and 200 body wordpieces,
    leaving room under MiniLM's 256-wordpiece limit. Publication dates remain
    metadata and are supplied to the answer model, not invented event dates.
    """
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact chunk offsets require a fast tokenizer")
    result = []
    step = max_tokens - overlap_tokens
    for row in documents.to_dict("records"):
        text = str(row["content_text"])
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False)["offset_mapping"]
        title = str(row["title"])
        title_offsets = tokenizer(title, add_special_tokens=False, return_offsets_mapping=True,
                                  truncation=False)["offset_mapping"]
        short_title = title[:title_offsets[min(31, len(title_offsets) - 1)][1]] if title_offsets else ""
        for position in range(0, len(offsets), step):
            window = offsets[position:position + max_tokens]
            if not window:
                continue
            start, end = int(window[0][0]), int(window[-1][1])
            if start >= end:
                continue
            passage = text[start:end]
            embedding_text = short_title + ". " + passage
            token_count = len(tokenizer(embedding_text, add_special_tokens=True, truncation=False)["input_ids"])
            if token_count > 256:
                raise ValueError(f"Dense input exceeds 256 tokens for record {row['record_id']}")
            chunk_id = stable_id("chunk", row["record_id"], row["source_fingerprint"], start, end)
            result.append({
                "chunk_id": chunk_id, "record_id": int(row["record_id"]), "title": title,
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "domain": str(row["domain"]), "url": str(row["url"]), "text": passage,
                "char_start": start, "char_end": end, "source_fingerprint": row["source_fingerprint"],
                "article_review_status": row.get("article_review_status", "unreviewed"),
                "article_review_method": row.get("article_review_method", "unreviewed"),
                "embedding_text": embedding_text, "embedding_tokens": token_count,
            })
    frame = pd.DataFrame(result)
    if not frame.empty and not frame.chunk_id.is_unique:
        raise ValueError("Duplicate chunk identities")
    return frame


def _passage_score(chunk, review):
    text = chunk["text"]
    score = min(len(GOVERNANCE_TERMS.findall(text)), 15)
    if re.search(r"\b(?:AI|artificial intelligence|model\w*|algorithm\w*|agent\w*|LLM\w*|"
                 r"OpenAI|Anthropic|ChatGPT|Gemini|deepfake\w*|robot\w*|facial recognition|biometric\w*)\b", text, re.I):
        score += 2
    else:
        score -= 20
    quote = str(review.get("review_evidence", ""))
    if quote and quote in text:
        score += 10
    if len(text) < 450:
        score -= 12
    if re.search(r"cookie policy|all rights reserved|apply now|job description", text, re.I):
        score -= 10
    return score


def select_passages(chunks, reviews, documents, count=120, seed=42, development_fraction=0.25):
    """Select source-diverse benchmark passages from positively reviewed articles.

    At least one passage per available reviewed source is preferred; extra
    passages are capped at three per article and avoid near-duplicate overlap.
    Relevance scoring here selects benchmark evidence, not corpus-wide labels.
    """
    by_article = reviews.set_index("record_id").to_dict("index")
    pools = {}
    for record_id, group in chunks.loc[chunks.record_id.isin(by_article)].groupby("record_id"):
        pool = group.copy()
        pool["selection_score"] = [_passage_score(c, by_article[record_id]) for c in pool.to_dict("records")]
        pool = pool.loc[pool.text.str.len().ge(450)].sort_values(["selection_score", "chunk_id"], ascending=[False, True])
        if not pool.empty:
            pools[int(record_id)] = pool.to_dict("records")
    if not pools:
        raise ValueError("No source-reviewed articles can supply passages")
    chosen = []
    per_article = Counter()
    for record_id in sorted(pools):
        chosen.append(pools[record_id][0]); per_article[record_id] += 1
    if len(chosen) > count:
        chosen = sorted(chosen, key=lambda r: (-r["selection_score"], r["chunk_id"]))[:count]
        per_article = Counter(r["record_id"] for r in chosen)
    extra = sorted([c for group in pools.values() for c in group[1:]],
                   key=lambda r: (-r["selection_score"], per_article[r["record_id"]], r["chunk_id"]))
    for row in extra:
        if len(chosen) >= count:
            break
        if per_article[row["record_id"]] >= 3:
            continue
        previous = [c for c in chosen if c["record_id"] == row["record_id"]]
        if any(max(0, min(c["char_end"], row["char_end"]) - max(c["char_start"], row["char_start"])) /
               min(c["char_end"] - c["char_start"], row["char_end"] - row["char_start"]) > 0.35 for c in previous):
            continue
        chosen.append(row); per_article[row["record_id"]] += 1
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} nonredundant passages; requested {count}")
    article_ids = sorted({c["record_id"] for c in chosen})
    strata = [str(by_article[i].get("primary_theme", "other")) for i in article_ids]
    if min(Counter(strata).values()) < 2:
        strata = None
    development, test = train_test_split(article_ids, train_size=development_fraction,
                                          random_state=seed, stratify=strata)
    development = set(development)
    document_lookup = documents.set_index("record_id")
    for row in chosen:
        record_id = row["record_id"]
        row["passage_id"] = row["chunk_id"]
        row["split"] = "development" if record_id in development else "test"
        row["themes"] = [theme for theme in THEMES if bool(document_lookup.loc[record_id, theme])]
        row["source_review_method"] = by_article[record_id]["review_method"]
    return pd.DataFrame(chosen).sort_values(["split", "date", "record_id", "char_start"]).reset_index(drop=True)


def validate_qas(qas, passages):
    """Check evidence identities and source splits, without inventing human review."""
    if not qas.qa_id.is_unique:
        raise ValueError("Q&A IDs must be unique")
    lookup = passages.set_index("passage_id").to_dict("index")
    for row in qas.to_dict("records"):
        passage = lookup[row["passage_id"]]
        if int(row["record_id"]) != int(passage["record_id"]) or row["split"] != passage["split"]:
            raise ValueError("Q&A source identity/split mismatch")
        if row["category"] not in CATEGORIES or not row["question"].strip() or not row["reference_answer"].strip():
            raise ValueError("Invalid Q&A category/question/answer")
        if not row["gold_evidence"] or row["gold_evidence"] not in passage["text"]:
            raise ValueError(f"Nonliteral Q&A evidence in {row['qa_id']}")
        if row["gold_chunk_ids"] != [row["passage_id"]]:
            raise ValueError("Unexpected gold evidence identities")
        if row.get("generation_model") == "llama3.2:latest":
            raise ValueError("Q&A generator must differ from the RAG model")
    by_split = qas.groupby("split").record_id.agg(set)
    if len(by_split) > 1 and set.intersection(*by_split.tolist()):
        raise ValueError("Article leakage between development and test")
    return True


def answer_evaluation_sample(qas, per_category=12, seed=42):
    """Freeze a balanced, bounded test subset before any answer is generated."""
    test = qas.loc[qas.split.eq("test") & qas.qa_review_verdict.eq("supported")].copy()
    selected = []
    for category in CATEGORIES:
        group = test.loc[test.category.eq(category)].sort_values("qa_id")
        if group.empty:
            raise ValueError(f"No test questions for {category}")
        selected.append(group.sample(n=min(per_category, len(group)), random_state=seed))
    return pd.concat(selected).sort_values(["category", "qa_id"]).reset_index(drop=True)


def prepare_qa_dataset(project_root, passages):
    """Preserve genuine manual revisions separately from the LLM-authored draft.

    Draft-based evaluation is explicitly provisional. A human verdict is never
    inferred from an AI source check, an exact quote or a model-generated score.
    """
    from src.stage1_validation.common import merge_review_template

    root = Path(project_root)
    directory = root / "data/processed/stage3"
    drafts = pd.DataFrame(read_jsonl(directory / "qa_drafts_codex.jsonl"))
    validate_qas(drafts, passages)
    if len(drafts) != 240 or len(passages) != 120 or not drafts.groupby("passage_id").size().eq(2).all():
        raise ValueError("Expected 120 passages and two distinct Q&A drafts per passage")
    from .qa_review import apply_ai_review
    ai_path = directory / "qa_ai_review.jsonl"
    reviewed_drafts, ai_reviews = apply_ai_review(drafts, passages, ai_path, CATEGORIES)
    validate_qas(reviewed_drafts, passages)
    sources = passages.set_index("passage_id").to_dict("index")
    template = reviewed_drafts.copy()
    template["source_fingerprint"] = [fingerprint({"article": sources[row["passage_id"]]["source_fingerprint"],
        "passage_id": row["passage_id"], "span": [sources[row["passage_id"]]["char_start"], sources[row["passage_id"]]["char_end"]],
        "question": row["question"], "answer": row["reference_answer"], "evidence": row["gold_evidence"],
        "category": row["category"], "split": row["split"]}) for row in reviewed_drafts.to_dict("records")]
    for column in ("title", "date", "domain", "url", "text"):
        template[column] = [sources[p][column] for p in drafts.passage_id]
    for field in HUMAN_QA_FIELDS:
        template[field] = ""
    review = merge_review_template(template, directory / "qa_manual_review.csv", key="qa_id",
                                   editable_fields=HUMAN_QA_FIELDS)
    allowed = {"", "approved", "rejected", "needs_revision"}
    if not review.human_review_verdict.isin(allowed).all():
        raise ValueError("Human Q&A verdict must be approved/rejected/needs_revision or blank")
    effective = []
    for row in review.to_dict("records"):
        verdict = row["human_review_verdict"]
        if verdict and (not str(row["human_reviewer"]).strip() or
                        pd.isna(pd.to_datetime(row["human_reviewed_at"], errors="coerce"))):
            raise ValueError("Human Q&A decisions require an actual reviewer and valid date")
        if verdict and is_ai_reviewer(row["human_reviewer"]):
            raise ValueError("An AI/model reviewer cannot be recorded as human Q&A approval")
        if verdict in {"rejected", "needs_revision"}:
            continue
        if verdict == "approved":
            for original, edited in (("question", "edited_question"), ("reference_answer", "edited_reference_answer"),
                                     ("gold_evidence", "edited_gold_evidence")):
                if str(row[edited]).strip():
                    row[original] = str(row[edited]).strip()
            row["qa_review_method"] = "human"
            row["qa_review_verdict"] = "supported"
        row["label_status"] = ("human_approved" if verdict == "approved" else
                               "ai_reviewed_pending_human_review" if ai_reviews else "ai_draft_pending_human_review")
        effective.append(row)
    qas = pd.DataFrame(effective)
    validate_qas(qas, passages)
    write_jsonl(directory / "qa_dataset.jsonl", qas.to_dict("records"))
    manual_complete = bool(200 <= len(qas) <= 300 and qas.label_status.eq("human_approved").all())
    status = {"draft_questions": len(drafts), "effective_questions": len(qas),
              "human_approved": int(review.human_review_verdict.eq("approved").sum()),
              "human_rejected": int(review.human_review_verdict.eq("rejected").sum()),
              "human_needs_revision": int(review.human_review_verdict.eq("needs_revision").sum()),
              "human_pending": int(review.human_review_verdict.eq("").sum()),
              "assignment_manual_review_complete": manual_complete,
              "generation_model": "ChatGPT / Codex", "rag_model": "llama3.2:latest",
              "categories": qas.category.value_counts().to_dict(), "splits": qas.split.value_counts().to_dict(),
              "draft_sha256": sha256_file(directory / "qa_drafts_codex.jsonl"),
              "ai_review_sha256": sha256_file(ai_path) if ai_path.exists() else None,
              "ai_reviewed": len(ai_reviews),
              "ai_review_supported": sum(row["verdict"] == "supported" for row in ai_reviews),
              "ai_review_revised": sum(row["verdict"] == "revised" for row in ai_reviews),
              "ai_review_unresolved": sum(row["verdict"] == "unresolved" for row in ai_reviews),
              "review_sha256": sha256_file(directory / "qa_manual_review.csv"),
              "effective_qa_fingerprint": fingerprint(qas.to_dict("records")),
              "evaluation_status": "human_reviewed" if manual_complete else "provisional_on_ai_drafts"}
    write_json(directory / "qa_dataset_manifest.json", status)
    if ai_reviews:
        write_json(root / "outputs/results/stage3/qa_ai_review_summary.json", {
            "review_method": "ai_assisted", "reviewed_at": sorted({row["reviewed_at"] for row in ai_reviews}),
            "reviewed": len(ai_reviews), "supported_unchanged": status["ai_review_supported"],
            "revised": status["ai_review_revised"], "unresolved": status["ai_review_unresolved"],
            "original_checks_failed": {field: sum(not row[field] for row in ai_reviews)
                                       for field in ("clarity_ok", "answer_supported", "evidence_sufficient", "category_ok")},
            "issue_counts": dict(Counter(issue for row in ai_reviews for issue in row["issues"])),
            "draft_sha256": status["draft_sha256"], "ai_review_sha256": status["ai_review_sha256"],
            "human_review_complete": manual_complete,
            "review_scope": "Question, reference answer, gold evidence and category assessed only against the selected original passage; not an external fact check or human approval.",
            "revised_qa_ids": [row["qa_id"] for row in ai_reviews if row["verdict"] == "revised"]})
    cards = []
    original_by_id = drafts.set_index("qa_id").to_dict("index")
    for row in review.to_dict("records"):
        row = row.copy()
        if row["human_review_verdict"] == "approved":
            for field in ("question", "reference_answer", "gold_evidence"):
                edited = str(row["edited_" + field]).strip()
                if edited:
                    row[field] = edited
        review_note = ""
        if row["human_review_verdict"]:
            review_note = ('<p><strong>Recorded review: ' + html.escape(row["human_review_verdict"]) +
                           '</strong> · ' + html.escape(row["human_reviewer"]) + ' · ' +
                           html.escape(row["human_reviewed_at"]) + '</p>')
        ai_note = ""
        if row.get("ai_review_verdict"):
            ai_note = ('<p><strong>AI-assisted review: ' + html.escape(row["ai_review_verdict"]) +
                       '</strong> · ' + html.escape(row["ai_reviewer"]) + ' · ' + html.escape(row["ai_reviewed_at"]) +
                       '</p><p>' + html.escape(row["ai_review_rationale"]) + '</p>')
            if row["ai_review_verdict"] == "revised":
                old = original_by_id[row["qa_id"]]
                ai_note += '<details><summary>Original draft before AI correction</summary>' + ''.join(
                    '<p><strong>' + html.escape(field) + ':</strong> ' + html.escape(old[field]) + '</p>'
                    for field in ("question", "reference_answer", "gold_evidence", "category")) + '</details>'
        cards.append('<article><h2>' + html.escape(row["qa_id"]) + '</h2><p>' +
            html.escape(f"{row['category']} · {row['split']} · {row['date']} · record {row['record_id']}") +
            '</p>' + review_note + ai_note + '<h3>' + html.escape(row["question"]) + '</h3><p><strong>Reference answer:</strong> ' +
            html.escape(row["reference_answer"]) + '</p><blockquote>' + html.escape(row["gold_evidence"]) +
            '</blockquote><details><summary>Original source passage</summary><p>' + html.escape(row["title"]) +
            '</p><pre>' + html.escape(row["text"]) + '</pre></details></article>')
    ai_intro = (f'<p><strong>Complete AI-assisted source review:</strong> {len(ai_reviews)} questions checked; '
                f'{status["ai_review_supported"]} retained, {status["ai_review_revised"]} revised, '
                f'{status["ai_review_unresolved"]} unresolved. Original drafts are preserved. '
                'AI reviews and corrections are not human approvals.</p>') if ai_reviews else ''
    report = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Stage 3 Q&A review</title>' + \
        '<style>body{font:16px/1.6 system-ui;max-width:960px;margin:30px auto;padding:20px}article{border-top:1px solid #bbb;padding:20px 0}pre{white-space:pre-wrap;font:inherit}</style>' + \
        '<h1>Source review of the Stage 3 Q&A draft</h1>' + ai_intro + '<p>These are AI-authored drafts. Check clarity, correctness, diversity and answerability against the passage. Record actual human decisions in qa_manual_review.csv; optional corrections use edited_question, edited_reference_answer and edited_gold_evidence. Rerun the notebook after saving.</p>' + ''.join(cards) + '</html>'
    report_path = root / "outputs/results/stage3/qa_review_packet.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return qas, review, status


def export_answer_review(project_root, qas, answers):
    """Optional human answer audit, source-bound and separate from AI judges."""
    from src.stage1_validation.common import merge_review_template

    root = Path(project_root)
    by_id = qas.set_index("qa_id").to_dict("index")
    rows = []
    for variant, results in answers.items():
        for qa_id, answer in results.items():
            qa = by_id[qa_id]
            snapshot = {"question": qa["question"], "reference_answer": qa["reference_answer"],
                        "answer": answer.get("answer"), "citations": answer.get("citations"),
                        "context": answer.get("context"), "cache_key": answer.get("cache_key")}
            rows.append({"answer_review_id": stable_id("answer_review", variant, qa_id),
                         "source_fingerprint": fingerprint(snapshot), "variant": variant, "qa_id": qa_id,
                         "category": qa["category"], "question": qa["question"],
                         "draft_reference_answer": qa["reference_answer"], "generated_answer": answer.get("answer", ""),
                         "generation_status": answer.get("status"),
                         "citations": json.dumps(answer.get("citations", [])),
                         "source_context": json.dumps(answer.get("context", []), ensure_ascii=False),
                         "ai_judge": json.dumps({key: (answer.get("judge") or {}).get(key) for key in
                             ("faithfulness", "relevance", "context_grounding", "temporal_grounding", "explanation")}, ensure_ascii=False)})
    fields = ["human_review_verdict", "human_reviewer", "human_reviewed_at", "human_review_notes"]
    template = pd.DataFrame(rows)
    for field in fields:
        template[field] = ""
    result = merge_review_template(template, root / "data/processed/stage3/answer_manual_review.csv",
                                   key="answer_review_id", editable_fields=fields)
    if not result.human_review_verdict.isin({"", "supported", "unsupported", "uncertain"}).all():
        raise ValueError("Invalid human answer-review verdict")
    for row in result.loc[result.human_review_verdict.ne("")].to_dict("records"):
        if not str(row["human_reviewer"]).strip() or is_ai_reviewer(row["human_reviewer"]):
            raise ValueError("Human answer review requires an actual human reviewer")
        if pd.isna(pd.to_datetime(row["human_reviewed_at"], errors="coerce")):
            raise ValueError("Human answer review requires a valid date")
    return result
