"""Small, offline utilities shared by the validation notebook and its checks."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

import pandas as pd


def stable_id(*parts: object) -> str:
    """Return a deterministic identifier without relying on Python's hash seed."""
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def source_fingerprint(row) -> str:
    """Bind a decision to its exact article text, title, date and record ID."""
    payload = {
        "record_id": str(int(row["record_id"])),
        "date": pd.Timestamp(row["date"]).isoformat(),
        "title": str(row["title"]),
        "content_text": str(row["content_text"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Replace one small review CSV atomically, preserving it if writing fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def merge_review_template(template, path, *, key, editable_fields):
    """Refresh source metadata while preserving only explicitly editable fields.

    An edited row whose source changes or leaves the sample stops the run. It is
    never silently dropped or transferred to a different source snapshot. A
    reviewer can archive the existing CSV and explicitly start a new review.
    """
    path = Path(path)
    fresh = template.copy()
    if fresh[key].duplicated().any():
        raise ValueError(f"Duplicate {key} in new review template")
    for field in editable_fields:
        if field not in fresh:
            fresh[field] = ""
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Older files called all decisions human_*; migrate genuine legacy
        # decisions without labelling new AI-assisted decisions as human work.
        legacy_decisions = pd.Series(False, index=old.index)
        for legacy, generic in (("human_relevance", "relevance_verdict"),
                                ("human_verdict", "relation_verdict")):
            if legacy not in old:
                continue
            legacy_decisions |= old[legacy].str.strip().ne("")
            if generic in old:
                conflict = old[legacy].ne("") & old[generic].ne("") & old[legacy].ne(old[generic])
                if conflict.any():
                    raise ValueError(f"Conflicting legacy and current decisions in {path}")
                old[generic] = old[generic].where(old[generic].ne(""), old[legacy])
                old = old.drop(columns=legacy)
            else:
                old = old.rename(columns={legacy: generic})
        if "review_method" in editable_fields and "review_method" not in old:
            old["review_method"] = legacy_decisions.map({True: "human", False: ""})
            if "reviewer" in old:
                known_ai = old["reviewer"].str.strip().str.casefold().isin(
                    {"chatgpt / codex", "chatgpt", "codex", "google gemini", "gemini", "claude ai"}
                )
                old.loc[legacy_decisions & known_ai, "review_method"] = "ai_assisted"
        required = {key, "source_fingerprint", *editable_fields}
        missing = required - set(old.columns)
        if missing:
            raise ValueError(f"Review CSV {path} is missing columns: {sorted(missing)}")
        if old[key].duplicated().any():
            raise ValueError(f"Duplicate {key} in existing review CSV {path}")
        edited = old[list(editable_fields)].apply(lambda col: col.str.strip().ne("")).any(axis=1)
        lookup = fresh.set_index(key)["source_fingerprint"].astype(str)
        for _, row in old.loc[edited].iterrows():
            if row[key] not in lookup.index:
                raise ValueError(f"Reviewed row {row[key]} left the sample; archive {path} before resampling")
            if lookup.loc[row[key]] != row["source_fingerprint"]:
                raise ValueError(f"Source changed for reviewed row {row[key]}; archive {path} and re-review")
        saved = old.set_index(key)
        for field in editable_fields:
            fresh[field] = fresh[key].map(saved[field]).fillna("")
    write_csv_atomic(fresh, path)
    return fresh


def load_stage1_normalization(notebook_path: Path):
    """Read the original literal alias/type dictionaries without executing cells."""
    wanted = {"ENTITY_ALIASES", "ENTITY_TYPE_OVERRIDES"}
    found = {}
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        found[target.id] = ast.literal_eval(node.value)
    if found.keys() != wanted:
        raise ValueError(f"Cannot read original normalization dictionaries: {wanted - found.keys()}")
    return found["ENTITY_ALIASES"], found["ENTITY_TYPE_OVERRIDES"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_review_packet(frame: pd.DataFrame, path: Path, *, kind: str) -> None:
    """Write escaped, offline source reading material; decisions live in the CSV.

    Article packets deliberately omit tags, classifier scores and NMF topics so
    that reviewers can judge the source independently of the original selection.
    """
    if kind not in {"article", "relation"}:
        raise ValueError("kind must be article or relation")
    cards = []
    for _, row in frame.iterrows():
        identifier = row["article_review_id" if kind == "article" else "relation_id"]
        claim = ""
        if kind == "relation":
            triple = f"{row['subject']} — {row['predicate']} → {row['object']}"
            claim = f"<h3>{html.escape(triple)}</h3><p>Original evidence: {html.escape(str(row['evidence']))}</p>"
        source = row.get("full_source_text", row.get("source_text", row.get("content_text", "")))
        verdict = row.get("relevance_verdict" if kind == "article" else "relation_verdict", "")
        recorded = ""
        if str(verdict).strip():
            recorded = (
                '<details><summary>Recorded review</summary>'
                f'<p><strong>{html.escape(str(verdict))}</strong> · '
                f'{html.escape(str(row.get("review_method", "")))} · '
                f'{html.escape(str(row.get("reviewer", "")))}</p>'
                f'<p>{html.escape(str(row.get("review_notes", "")))}</p>'
                f'<blockquote>{html.escape(str(row.get("review_evidence", row.get("reviewed_evidence", ""))))}</blockquote></details>'
            )
        cards.append(
            f'<article id="{html.escape(str(identifier))}"><h2>Record {int(row["record_id"])}: '
            f'{html.escape(str(row.get("title", "")))}</h2>'
            f'<p>Review ID: <code>{html.escape(str(identifier))}</code><br>'
            f'{html.escape(str(row.get("date", "")))} · {html.escape(str(row.get("domain", "")))}</p>'
            f'{claim}<details><summary>Read complete source text</summary>'
            f'<pre>{html.escape(str(source))}</pre></details>{recorded}</article>'
        )
    instructions = (
        "Judge whether AI security/safety, accountability, bias/fairness or regulation is a substantive topic. "
        "Generic business, software, security or marketing content alone is insufficient. Record the verdict, "
        "your name and a source quote in article_review.csv; use uncertain when the evidence is unclear."
        if kind == "article" else
        "Check the subject, predicate, object, direction and explicit support in the source. Mere co-mention "
        "or a plausible external fact is insufficient. Record your verdict and name in relation_review.csv. "
        "For a supported verdict, supply reviewed_evidence when the original quote is not exact."
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Stage 1 {kind} source review</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;}'
        'article{border-top:1px solid #bbb;padding:24px 0}pre{white-space:pre-wrap;font:inherit;}'
        'code{font-size:13px}summary{cursor:pointer}</style>'
        f'<h1>Stage 1 {kind} source review</h1><p>{html.escape(instructions)}</p>'
        '<p>This file is reading material. Save decisions in the corresponding CSV, then rerun the notebook. '
        'Blank decisions are pending. AI-assisted judgments are explicitly labelled; they are not human annotations.</p>'
        + "\n".join(cards) + "</html>"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(document, encoding="utf-8")
