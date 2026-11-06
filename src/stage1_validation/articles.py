"""Independent article-review sampling for the Stage 1 handoff.

Tags and NMF topics describe the sampling frame; they never become human
relevance judgments. Reviews are persisted separately from the original corpus.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import merge_review_template, source_fingerprint, stable_id


ARTICLE_REVIEW_FIELDS = [
    "relevance_verdict", "review_method", "reviewer", "reviewed_at", "review_evidence", "review_notes"
]
THEME_COLUMNS = ["security", "accountability", "bias_fairness", "regulation"]
THEME_PRIORITY = ["regulation", "bias_fairness", "security", "accountability"]
MAIN_GROUP = "tagged_stratified"
CANDIDATE_GROUP = "stage2_candidates"
VERDICTS = {"relevant", "irrelevant", "uncertain"}


def _clean(value):
    return "" if pd.isna(value) else str(value).strip()


def _frame(corpus, start, end):
    required = {"record_id", "title", "date", "domain", "url", "content_text",
                "governance_related", *THEME_COLUMNS}
    missing = required.difference(corpus.columns)
    if missing:
        raise ValueError(f"Article corpus is missing columns: {sorted(missing)}")
    if corpus["record_id"].duplicated().any():
        raise ValueError("Article corpus record_id must be unique.")
    result = corpus.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    return result.loc[result["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def _allocate(populations, total):
    """Proportional largest-remainder allocation with one item per stratum."""
    populations = populations.sort_index().astype(int)
    total = min(int(total), int(populations.sum()))
    if total < len(populations):
        raise ValueError(
            f"sample_size={total} cannot cover {len(populations)} nonempty strata; "
            "increase sample_size."
        )
    if not len(populations):
        return populations
    counts = pd.Series(1, index=populations.index, dtype=int)
    capacities = populations - counts
    extra = total - int(counts.sum())
    if extra and capacities.sum():
        quotas = capacities * extra / capacities.sum()
        additions = np.floor(quotas).astype(int)
        counts += additions
        remaining = total - int(counts.sum())
        # Stable sorting also makes equal remainders deterministic.
        order = (quotas - additions).sort_values(ascending=False, kind="stable").index
        for key in order:
            if remaining <= 0:
                break
            if counts[key] < populations[key]:
                counts[key] += 1
                remaining -= 1
    return counts


def _add_sampling_metadata(frame, start):
    frame = frame.copy()
    theme_matrix = frame[THEME_PRIORITY].astype(bool)
    if (~theme_matrix.any(axis=1)).any():
        raise ValueError("Tagged Governance articles must have at least one Governance theme.")
    frame["primary_theme"] = theme_matrix.idxmax(axis=1)
    origin = pd.Timestamp(start)
    months = (frame["date"].dt.year - origin.year) * 12 + frame["date"].dt.month - origin.month
    frame["six_month_period"] = months.floordiv(6).astype(int)
    frame["stratum"] = frame["primary_theme"] + "|period_" + frame["six_month_period"].astype(str)
    return frame


def _coverage(population, sample, group):
    rows = []
    if population.empty:
        return rows
    # Sources are grouped only for a compact coverage diagnostic, not sampling.
    top_sources = population["domain"].value_counts().head(10).index
    for dimension in ("month", "source", "nmf_topic", "theme"):
        if dimension == "month":
            all_values = population["date"].dt.strftime("%Y-%m")
            sampled_values = sample["date"].dt.strftime("%Y-%m")
        elif dimension == "source":
            all_values = population["domain"].where(population["domain"].isin(top_sources), "Other sources")
            sampled_values = sample["domain"].where(sample["domain"].isin(top_sources), "Other sources")
        elif dimension == "nmf_topic":
            all_values = population["dominant_topic_label"].fillna("No Stage 1 NMF assignment")
            sampled_values = sample["dominant_topic_label"].fillna("No Stage 1 NMF assignment")
        else:
            for theme in THEME_COLUMNS:
                rows.append({"sample_group": group, "dimension": "theme", "value": theme,
                             "population_count": int(population[theme].sum()),
                             "sample_count": int(sample[theme].sum())})
            continue
        all_counts = all_values.value_counts()
        sample_counts = sampled_values.value_counts()
        for value, count in all_counts.items():
            rows.append({"sample_group": group, "dimension": dimension, "value": value,
                         "population_count": int(count), "sample_count": int(sample_counts.get(value, 0))})
    return rows


def prepare_article_review(corpus, topics, candidates, review_path: Path, seed=42,
                           sample_size=120, candidate_sample_size=40,
                           start="2024-09-01", end="2026-08-31"):
    """Create a reproducible stratified review and a separate candidate audit.

    Within each rare-first primary-theme × six-month stratum, articles are
    sampled uniformly without replacement. Inverse inclusion weights support a
    design-weighted precision estimate only after the entire main sample is
    independently reviewed. The untagged candidate audit cannot estimate recall.
    Existing editable annotations are preserved by ``merge_review_template``.
    """
    if sample_size <= 0 or candidate_sample_size < 0:
        raise ValueError("sample_size must be positive and candidate_sample_size nonnegative.")
    window = _frame(corpus, start, end)
    if topics["record_id"].duplicated().any():
        raise ValueError("Topic assignments must contain unique record_id values.")
    topic_columns = [c for c in ("record_id", "dominant_topic", "dominant_topic_label") if c in topics]
    window = window.merge(topics[topic_columns], on="record_id", how="left", validate="one_to_one")
    if "dominant_topic_label" not in window:
        window["dominant_topic_label"] = pd.NA
    tagged = _add_sampling_metadata(window.loc[window["governance_related"].astype(bool)], start)
    if tagged.empty:
        raise ValueError("No tagged Governance articles exist in the requested time window.")
    populations = tagged.groupby("stratum", sort=True).size()
    allocations = _allocate(populations, sample_size)
    rng = np.random.default_rng(seed)
    sample_parts, sampling_rows = [], []
    for stratum, count in allocations.items():
        pool = tagged.loc[tagged["stratum"].eq(stratum)].sort_values("record_id")
        sample = pool.iloc[rng.choice(len(pool), size=int(count), replace=False)].copy()
        probability = float(count / len(pool))
        sample["sample_group"] = MAIN_GROUP
        sample["inclusion_probability"] = probability
        sample["sample_weight"] = 1 / probability
        sample_parts.append(sample)
        sampling_rows.append({"sample_group": MAIN_GROUP, "stratum": stratum,
                              "population_size": len(pool), "sample_size": int(count),
                              "inclusion_probability": probability, "sample_weight": 1 / probability})
    main_sample = pd.concat(sample_parts, ignore_index=True)
    candidate_ids = set(candidates["record_id"]) if candidates is not None else set()
    candidate_pool = window.loc[
        ~window["governance_related"].astype(bool) & window["record_id"].isin(candidate_ids)
    ].sort_values("record_id").copy()
    candidate_n = min(int(candidate_sample_size), len(candidate_pool))
    candidate_sample = candidate_pool.iloc[
        rng.choice(len(candidate_pool), size=candidate_n, replace=False)
    ].copy()
    if candidate_n:
        probability = candidate_n / len(candidate_pool)
        candidate_sample["sample_group"] = CANDIDATE_GROUP
        candidate_sample["stratum"] = "uniform_untagged_stage2_candidates"
        candidate_sample["inclusion_probability"] = probability
        candidate_sample["sample_weight"] = 1 / probability
        sampling_rows.append({"sample_group": CANDIDATE_GROUP,
                              "stratum": "uniform_untagged_stage2_candidates",
                              "population_size": len(candidate_pool), "sample_size": candidate_n,
                              "inclusion_probability": probability, "sample_weight": 1 / probability})
    template = pd.concat([main_sample, candidate_sample], ignore_index=True)
    template["source_fingerprint"] = template.apply(source_fingerprint, axis=1)
    template["article_review_id"] = [
        stable_id("article", row.record_id, row.source_fingerprint)
        for row in template.itertuples(index=False)
    ]
    for field in ARTICLE_REVIEW_FIELDS:
        template[field] = ""
    template = template.sort_values(["sample_group", "stratum", "record_id"], kind="stable").reset_index(drop=True)
    review = merge_review_template(template, Path(review_path), key="article_review_id",
                                   editable_fields=ARTICLE_REVIEW_FIELDS)
    _validate_reviews(review)
    coverage_rows = _coverage(tagged, main_sample, MAIN_GROUP)
    coverage_rows += _coverage(candidate_pool, candidate_sample, CANDIDATE_GROUP)
    return review, pd.DataFrame(sampling_rows), pd.DataFrame(coverage_rows)


def _validate_reviews(review):
    """Validate source-based judgments and distinguish human from AI review."""
    if review["record_id"].duplicated().any():
        raise ValueError("Article review contains duplicate record_id values.")
    labels = review["relevance_verdict"].map(_clean)
    invalid = ~labels.isin(VERDICTS | {""})
    if invalid.any():
        raise ValueError(f"Unknown article review verdicts: {sorted(set(labels[invalid]))}")
    for _, row in review.loc[labels.ne("")].iterrows():
        if source_fingerprint(row) != row.source_fingerprint:
            raise ValueError(f"Article {row.record_id}: review source_fingerprint does not match its source text.")
        if not _clean(row["reviewer"]):
            raise ValueError(f"Article {row.record_id}: a verdict requires reviewer.")
        if _clean(row.get("review_method")) not in {"human", "ai_assisted"}:
            raise ValueError(f"Article {row.record_id}: review_method must be human or ai_assisted.")
        if _clean(row.get("review_method")) == "ai_assisted" and pd.isna(pd.to_datetime(row.get("reviewed_at"), errors="coerce")):
            raise ValueError(f"Article {row.record_id}: AI-assisted review requires a valid reviewed_at date.")
        if _clean(row["relevance_verdict"]) in {"irrelevant", "uncertain"} and not _clean(row["review_notes"]):
            raise ValueError(f"Article {row.record_id}: irrelevant or uncertain requires review_notes explaining the judgment.")
        if _clean(row["relevance_verdict"]) == "relevant":
            quote = _clean(row["review_evidence"])
            source = f"{row['title']}. {row['content_text']}"
            if not quote or quote not in source:
                raise ValueError(f"Article {row.record_id}: relevant requires an exact source quote in review_evidence.")
    reviewed = labels.ne("")
    resolved = labels.isin({"relevant", "irrelevant"})
    return labels, reviewed, resolved


def summarize_article_review(review):
    """Report review progress; withhold precision until the main sample resolves.

    No candidate-pool result is a recall estimate. NMF grouping is a diagnostic
    of source/topic contamination, not independent Ground Truth.
    """
    labels, reviewed, resolved = _validate_reviews(review)
    annotated = review.assign(_label=labels, _reviewed=reviewed, _resolved=resolved)
    rows = []
    main = annotated.loc[annotated["sample_group"].eq(MAIN_GROUP)]
    main_complete = bool(len(main) and main["_resolved"].all())
    for group, frame in annotated.groupby("sample_group", sort=True):
        n = len(frame)
        weights = frame["sample_weight"].astype(float)
        weighted_precision = np.nan
        if group == MAIN_GROUP and main_complete:
            weighted_precision = float((weights * frame["_label"].eq("relevant")).sum() / weights.sum())
        assessed = bool(frame["_reviewed"].all())
        lower = float((weights * frame["_label"].eq("relevant")).sum() / weights.sum()) if assessed else np.nan
        upper = float((weights * frame["_label"].isin({"relevant", "uncertain"})).sum() / weights.sum()) if assessed else np.nan
        rows.append({"sample_group": group, "sample_size": n,
                     "reviewed": int(frame["_reviewed"].sum()), "resolved": int(frame["_resolved"].sum()),
                     "pending": int((~frame["_reviewed"]).sum()),
                     "uncertain": int(frame["_label"].eq("uncertain").sum()),
                     "relevant": int(frame["_label"].eq("relevant").sum()),
                     "irrelevant": int(frame["_label"].eq("irrelevant").sum()),
                     "human_reviewed": int((frame["_reviewed"] & frame["review_method"].eq("human")).sum()),
                     "ai_reviewed": int((frame["_reviewed"] & frame["review_method"].eq("ai_assisted")).sum()),
                     "review_completion": float(frame["_reviewed"].mean()) if n else 0.0,
                     "main_sample_complete": main_complete if group == MAIN_GROUP else False,
                     "main_human_sample_complete": bool(main_complete and frame["review_method"].eq("human").all()) if group == MAIN_GROUP else False,
                     "weighted_relevance_min": lower if group == MAIN_GROUP else np.nan,
                     "weighted_relevance_max": upper if group == MAIN_GROUP else np.nan,
                     "weighted_precision": weighted_precision})
    topic_rows = []
    for topic, frame in main.groupby("dominant_topic_label", dropna=False, sort=True):
        weights = frame["sample_weight"].astype(float)
        topic_rows.append({"dominant_topic_label": topic, "sample_size": len(frame),
                           "reviewed": int(frame["_reviewed"].sum()),
                           "resolved": int(frame["_resolved"].sum()),
                           "pending": int((~frame["_reviewed"]).sum()),
                           "uncertain": int(frame["_label"].eq("uncertain").sum()),
                           "relevant": int(frame["_label"].eq("relevant").sum()),
                           "irrelevant": int(frame["_label"].eq("irrelevant").sum()),
                           "weighted_relevance_min": float((weights * frame["_label"].eq("relevant")).sum() / weights.sum())
                           if frame["_reviewed"].all() else np.nan,
                           "weighted_relevance_max": float((weights * frame["_label"].isin({"relevant", "uncertain"})).sum() / weights.sum())
                           if frame["_reviewed"].all() else np.nan,
                           "weighted_precision": float((weights * frame["_label"].eq("relevant")).sum() / weights.sum())
                           if main_complete else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(topic_rows)


def apply_article_review(corpus, review, start="2024-09-01", end="2026-08-31"):
    """Return the in-window tagged frame with transparent review status.

    Explicitly irrelevant reviewed articles are excluded. Unreviewed and
    uncertain articles retain their weak-label status; they are never silently
    promoted to verified evidence. Candidate articles are not added automatically.
    """
    labels, _, _ = _validate_reviews(review)
    if corpus["record_id"].duplicated().any():
        raise ValueError("Article corpus record_id must be unique.")
    indexed = corpus.set_index("record_id", drop=False)
    for position, (_, row) in enumerate(review.iterrows()):
        if row.record_id not in indexed.index:
            raise ValueError(f"Reviewed article {row.record_id} is absent from the corpus.")
        if labels.iloc[position] and source_fingerprint(indexed.loc[row.record_id]) != row.source_fingerprint:
            raise ValueError(f"Reviewed article {row.record_id} has a stale source_fingerprint; re-review changed source.")
    eligible = _frame(corpus, start, end)
    eligible = eligible.loc[eligible["governance_related"].astype(bool)].copy()
    verdicts = dict(zip(review["record_id"], labels))
    eligible["article_review_status"] = eligible["record_id"].map(verdicts).fillna("").replace("", "unreviewed")
    methods = dict(zip(review["record_id"], review["review_method"]))
    eligible["article_review_method"] = eligible["record_id"].map(methods).fillna("").replace("", "unreviewed")
    eligible = eligible.loc[~eligible["article_review_status"].eq("irrelevant")].copy()
    eligible["source_fingerprint"] = eligible.apply(source_fingerprint, axis=1)
    return eligible.sort_values("record_id").reset_index(drop=True)
