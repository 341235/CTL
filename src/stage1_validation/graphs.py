"""Source audits and conservative graph linkage for the Stage 1 handoff.

An exact quotation is a provenance check, not a correctness judgement. Semantic
edges become retrievable assertions only after an explicit, source-bound review.
The entity/article index records NER membership; semantic-edge provenance stays
on the semantic edge so it never inflates a co-occurrence count.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import networkx as nx
import numpy as np
import pandas as pd

from .common import merge_review_template, source_fingerprint, stable_id


RELATION_REVIEW_FIELDS = [
    "relation_verdict", "review_method", "reviewer", "reviewed_at", "review_notes",
    "reviewed_evidence", "reviewed_char_start", "reviewed_char_end",
]
VERDICTS = {"supported", "weak", "incorrect", "uncertain"}
THEME_FLAGS = {
    "Security": "security", "Accountability": "accountability",
    "Bias / Fairness": "bias_fairness", "Regulation": "regulation",
}
# Endpoint constraints in the Stage 1 notebook's schema/data dictionary.
ENDPOINT_TYPES = {
    "REGULATES": ({"LAW", "REGULATOR", "GPE", "ORG"}, {"ORG", "TECHNOLOGY", "MODEL", "PRODUCT"}),
    "INVESTIGATES": ({"REGULATOR", "GPE", "ORG"}, {"ORG", "PERSON"}),
    "FINES": ({"REGULATOR", "GPE", "ORG"}, {"ORG"}),
    "SUES": (None, {"ORG", "PERSON"}),
    "COMPLIES_WITH": ({"ORG", "PRODUCT", "MODEL"}, {"LAW", "STANDARD"}),
    "ADOPTS": ({"GPE", "ORG", "REGULATOR"}, {"LAW", "STANDARD"}),
    "PROPOSES": ({"GPE", "ORG", "REGULATOR", "PERSON"}, {"LAW", "STANDARD"}),
    "DEVELOPS": ({"ORG", "PERSON"}, {"MODEL", "PRODUCT", "TECHNOLOGY", "STANDARD"}),
    "PARTNERS_WITH": ({"ORG", "GPE"}, {"ORG", "GPE"}),
}


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _integer(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def _date(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(timestamp) else timestamp.strftime("%Y-%m-%d")


def _canonical(name, entity_type, aliases, overrides):
    name = _text(name).strip()
    canonical = aliases.get(name, name)
    return canonical.strip(), overrides.get(canonical, _text(entity_type).upper().strip())


def _entity_id(name, entity_type):
    return f"{entity_type}::{name}"


def _split_id(entity_id):
    entity_type, name = entity_id.split("::", 1)
    return name, entity_type


def _matches(text, quote):
    if not quote:
        return []
    positions, start = [], 0
    while True:
        position = text.find(quote, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def _exact_span(text, quote, start, end):
    return bool(quote and start is not None and end is not None
                and 0 <= start < end <= len(text) and text[start:end] == quote)


def audit_relations(relations, corpus, schema, aliases, overrides):
    """Check schema, dates and actual quotation alignment, without judging claims.

    Original offsets refer to ``(title + '. ' + content_text)[:3000]``.
    Full-source text is retained for an independent review and exact re-anchoring.
    """
    if not corpus["record_id"].is_unique:
        raise ValueError("Corpus record_id must be unique")
    articles = corpus.set_index("record_id", drop=False)
    allowed_types = set(schema["node_types"]) - {"GOVERNANCE_THEME"}
    predicates = {key for key, spec in schema["relation_types"].items()
                  if spec.get("source") == "LLM extraction"}
    rows = []
    for relation in relations.to_dict("records"):
        record_id = relation["record_id"]
        source_found = record_id in articles.index
        article = articles.loc[record_id] if source_found else {}
        title, content = _text(article.get("title")), _text(article.get("content_text"))
        source_text = f"{title}. {content}" if source_found else ""
        extraction_input = source_text[:3000]
        quote = _text(relation.get("evidence"))
        start, end = _integer(relation.get("char_start")), _integer(relation.get("char_end"))
        subject, subject_type = _canonical(relation.get("subject"), relation.get("subject_type"), aliases, overrides)
        obj, object_type = _canonical(relation.get("object"), relation.get("object_type"), aliases, overrides)
        predicate = _text(relation.get("predicate")).strip().upper()
        permitted_subjects, permitted_objects = ENDPOINT_TYPES.get(predicate, (None, None))
        schema_valid = (predicate in predicates and subject_type in allowed_types and object_type in allowed_types
                        and bool(subject) and bool(obj) and subject != obj
                        and (permitted_subjects is None or subject_type in permitted_subjects)
                        and (permitted_objects is None or object_type in permitted_objects))
        source_matches, input_matches = _matches(source_text, quote), _matches(extraction_input, quote)
        exact_span = _exact_span(extraction_input, quote, start, end)
        if not source_found:
            alignment = "missing_source"
        elif exact_span:
            alignment = "exact_input_span"
        elif len(input_matches) == 1:
            alignment = "unique_input_quote_offset_mismatch"
        elif len(input_matches) > 1:
            alignment = "ambiguous_input_quote"
        elif len(source_matches) > 0:
            alignment = "quote_only_outside_extraction_input"
        else:
            alignment = "no_exact_source_quote"
        row = dict(relation)
        row.update({
            "relation_id": stable_id("relation", record_id, relation.get("subject_type"), relation.get("subject"),
                                     predicate, relation.get("object_type"), relation.get("object"),
                                     _entity_id(subject, subject_type), _entity_id(obj, object_type),
                                     quote, start, end),
            "source_fingerprint": source_fingerprint(article) if source_found else "",
            "source_found": source_found, "source_date": _date(article.get("date")),
            "source_date_matches": source_found and _date(relation.get("date")) == _date(article.get("date"))
                                   and bool(_date(article.get("date"))),
            "schema_valid": schema_valid,
            "subject_id": _entity_id(subject, subject_type), "object_id": _entity_id(obj, object_type),
            "original_subject_id": _entity_id(_text(relation.get("subject")), _text(relation.get("subject_type"))),
            "original_object_id": _entity_id(_text(relation.get("object")), _text(relation.get("object_type"))),
            "exact_span_match": exact_span,
            "evidence_in_input": bool(input_matches), "evidence_in_source": bool(source_matches),
            "input_quote_occurrences": len(input_matches), "source_quote_occurrences": len(source_matches),
            "alignment_status": alignment,
            "reanchored_start": input_matches[0] if len(input_matches) == 1 else None,
            "reanchored_end": input_matches[0] + len(quote) if len(input_matches) == 1 else None,
            "title": title, "content_text": content, "source_text": source_text,
            "full_source_text": source_text,
            "extraction_input": extraction_input,
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) and not result["relation_id"].is_unique:
        raise ValueError("Duplicate relation identities; disambiguate source relations before review")
    return result


def prepare_relation_review(audit, co_graph, review_path: Path, sample_size=40, seed=42):
    """Select high-impact edges plus a random tail; keep both groups explicit.

    The importance sample is diagnostic, not a population precision estimate.
    Legal/enforcement predicates receive priority alongside endpoint centrality.
    """
    ranking = nx.pagerank(co_graph, weight="weight") if len(co_graph) else {}
    scores = []
    legal = {"REGULATES", "INVESTIGATES", "FINES", "SUES", "COMPLIES_WITH", "ADOPTS", "PROPOSES"}
    for row in audit.to_dict("records"):
        scores.append(ranking.get(row["subject_id"], 0) + ranking.get(row["object_id"], 0)
                      + (0.01 if row["predicate"] in legal else 0))
    ranked = audit.assign(priority_score=scores).sort_values(
        ["priority_score", "relation_id"], ascending=[False, True])
    count = min(max(int(sample_size), 0), len(ranked))
    important_count = min(20, (count + 1) // 2)
    important = ranked.head(important_count).assign(review_selection="important")
    remainder = ranked.iloc[important_count:].sort_values("relation_id")
    random = remainder.sample(n=count - important_count, random_state=seed).assign(review_selection="random_tail")
    template = pd.concat([important, random], ignore_index=True)
    for field in RELATION_REVIEW_FIELDS:
        template[field] = ""
    review = merge_review_template(template, Path(review_path), key="relation_id", editable_fields=RELATION_REVIEW_FIELDS)
    invalid = set(review["relation_verdict"].fillna("").astype(str).str.strip()) - VERDICTS - {""}
    if invalid:
        raise ValueError(f"Unknown relation review verdicts: {sorted(invalid)}")
    return review


def build_entity_links(co_graph, audit, entities, eligible_articles, aliases, overrides):
    """Link typed IDs conservatively and build a source-bound NER inverted index.

    Exact aliases/type overrides reuse Stage 1 rules. Case-insensitive resolution
    is allowed only to a unique existing name of the same type. Type conflicts
    remain visible and are never merged by fuzzy name matching.
    """
    eligible = eligible_articles.set_index("record_id", drop=False)
    if not eligible.index.is_unique or not entities["record_id"].is_unique:
        raise ValueError("Article/entity record_id values must be unique")
    base_candidates = defaultdict(set)
    for node in co_graph:
        name, entity_type = _split_id(node)
        name, entity_type = _canonical(name, entity_type, aliases, overrides)
        base_candidates[(entity_type, name.casefold())].add(_entity_id(name, entity_type))

    def resolve(name, entity_type):
        original_name, original_type = name, entity_type
        name, entity_type = _canonical(name, entity_type, aliases, overrides)
        canonical_id = _entity_id(name, entity_type)
        method = "exact"
        if name != original_name or entity_type != original_type:
            method = "stage1_alias_or_type_override"
        candidates = base_candidates.get((entity_type, name.casefold()), set())
        if canonical_id not in candidates and len(candidates) == 1:
            canonical_id = next(iter(candidates))
            method += "+unique_same_type_casefold"
        return canonical_id, method

    mapping = {}

    def register(name, entity_type, origin):
        original_id = _entity_id(name, entity_type)
        canonical_id, method = resolve(name, entity_type)
        if original_id not in mapping:
            mapping[original_id] = {"original_id": original_id, "canonical_id": canonical_id,
                                    "match_method": method, "origins": set()}
        mapping[original_id]["origins"].add(origin)
        return canonical_id

    base_ids = set()
    for node in co_graph:
        name, entity_type = _split_id(node)
        base_ids.add(register(name, entity_type, "cooccurrence_graph"))
    relation_ids = set()
    for row in audit.to_dict("records"):
        for side in ("subject", "object"):
            name, entity_type = _split_id(row[f"original_{side}_id"])
            relation_ids.add(register(name, entity_type, "semantic_graph"))
    requested_ids = base_ids | relation_ids
    requested_names = {_split_id(node)[0].casefold() for node in requested_ids}
    source_info = {
        record_id: {"date": _date(article["date"]),
                    "source_fingerprint": article.get("source_fingerprint") or source_fingerprint(article),
                    "article_review_status": article.get("article_review_status", "unreviewed"),
                    "article_review_method": article.get("article_review_method", "unreviewed")}
        for record_id, article in eligible.iterrows()
    }
    index_rows = []
    for row in entities.itertuples(index=False):
        if row.record_id not in eligible.index:
            continue
        article = source_info[row.record_id]
        for name, entity_type in row.entities_normalized:
            canonical_id, _ = resolve(name, entity_type)
            if _split_id(canonical_id)[0].casefold() not in requested_names:
                continue
            register(name, entity_type, "ner_cache")
            if canonical_id not in requested_ids or canonical_id.startswith("GOVERNANCE_THEME::"):
                continue
            index_rows.append({
                "entity_id": canonical_id, "record_id": row.record_id,
                "date": article["date"],
                "source_fingerprint": article["source_fingerprint"],
                "article_review_status": article.get("article_review_status", "unreviewed"),
                "article_review_method": article.get("article_review_method", "unreviewed"),
                "source": "ner",
            })
    types_by_name = defaultdict(set)
    for entry in mapping.values():
        name, entity_type = _split_id(entry["canonical_id"])
        types_by_name[name.casefold()].add(entity_type)
    mapping_rows = []
    for entry in mapping.values():
        name, _ = _split_id(entry["canonical_id"])
        mapping_rows.append({**entry, "origins": ";".join(sorted(entry["origins"])),
                             "in_cooccurrence_graph": entry["canonical_id"] in base_ids,
                             "in_relation_graph": entry["canonical_id"] in relation_ids,
                             "type_conflict": len(types_by_name[name.casefold()]) > 1,
                             "conflicting_types": ";".join(sorted(types_by_name[name.casefold()]))})
    mapping_df = pd.DataFrame(mapping_rows).sort_values("original_id").reset_index(drop=True)
    columns = ["entity_id", "record_id", "date", "source_fingerprint", "article_review_status", "article_review_method", "source"]
    index_df = pd.DataFrame(index_rows, columns=columns).drop_duplicates(["entity_id", "record_id"])
    index_df = index_df.sort_values(["entity_id", "record_id"]).reset_index(drop=True)
    summary = pd.DataFrame([{
        "cooccurrence_entity_ids": len(base_ids), "semantic_entity_ids": len(relation_ids),
        "shared_entity_ids": len(base_ids & relation_ids),
        "semantic_only_entity_ids": len(relation_ids - base_ids),
        "type_conflict_names": sum(len(types) > 1 for types in types_by_name.values()),
        "entity_article_memberships": len(index_df), "indexed_articles": index_df["record_id"].nunique(),
    }])
    return mapping_df, index_df, summary


def supporting_articles(entity_article, entity_a, entity_b=None):
    """Return dated source articles for one entity or an exact co-mention pair."""
    matches = entity_article.loc[entity_article["entity_id"] == entity_a]
    if entity_b is not None:
        other = set(entity_article.loc[entity_article["entity_id"] == entity_b, "record_id"])
        matches = matches.loc[matches["record_id"].isin(other)]
    return matches.drop(columns="entity_id").drop_duplicates("record_id").sort_values("record_id").reset_index(drop=True)


def _review_decision(row, review, eligible_ids=None, eligible_fingerprints=None):
    """Resolve an explicitly attributed judgement and exact source quotation."""
    verdict = _text(review.get("relation_verdict")).strip()
    if verdict and verdict not in VERDICTS:
        raise ValueError(f"Unknown relation review verdict: {verdict}")
    has_review = bool(verdict)
    if has_review and _text(review.get("source_fingerprint")) != row["source_fingerprint"]:
        raise ValueError(f"Stale source fingerprint for reviewed relation {row['relation_id']}")
    reviewer = _text(review.get("reviewer")).strip()
    method = _text(review.get("review_method")).strip()
    if has_review and method not in {"human", "ai_assisted"}:
        raise ValueError("A relation verdict requires review_method human or ai_assisted")
    reviewed_at = _date(review.get("reviewed_at"))
    notes_required = verdict in {"weak", "incorrect", "uncertain"}
    has_notes = bool(_text(review.get("review_notes")).strip())
    recorded = bool(has_review and reviewer and reviewed_at and (not notes_required or has_notes))
    quote = _text(review.get("reviewed_evidence")) or _text(row.get("evidence"))
    positions = _matches(row["source_text"], quote)
    specified_start = _text(review.get("reviewed_char_start")).strip()
    specified_end = _text(review.get("reviewed_char_end")).strip()
    if specified_start or specified_end:
        start, end = _integer(specified_start), _integer(specified_end)
        exact = _exact_span(row["source_text"], quote, start, end)
        anchor_method = "reviewer_supplied_offsets"
    elif not _text(review.get("reviewed_evidence")) and row["exact_span_match"]:
        start, end = _integer(row["char_start"]), _integer(row["char_end"])
        exact = True
        anchor_method = "original_exact_offsets"
    elif len(positions) == 1:
        start, end = positions[0], positions[0] + len(quote)
        exact = True
        anchor_method = "unique_exact_source_reanchor"
    else:
        start, end, exact, anchor_method = None, None, False, "missing_or_ambiguous_quote"
    source_valid = bool(row["source_found"] and row["source_date_matches"] and row["schema_valid"])
    eligible = eligible_ids is None or row["record_id"] in eligible_ids
    current_source = (eligible_fingerprints is None or not eligible
                      or eligible_fingerprints.get(row["record_id"]) == row["source_fingerprint"])
    # A source-supported judgement is resolved even when the source article is
    # excluded from this topic's corpus. Scope controls retrieval, not whether
    # the source review actually happened (e.g. a genuine neurotechnology bill).
    complete = recorded and verdict != "uncertain" and (verdict != "supported" or (exact and source_valid and current_source))
    accepted = complete and verdict == "supported" and eligible
    if not has_review:
        reason = "unreviewed"
    elif notes_required and not has_notes:
        reason = "review_notes_missing"
    elif not recorded:
        reason = "reviewer_or_date_missing"
    elif verdict != "supported":
        reason = f"{method}_{verdict}"
    elif not eligible:
        reason = "article_outside_eligible_corpus"
    elif not current_source:
        reason = "eligible_source_changed"
    elif not source_valid:
        reason = "schema_source_or_date_invalid"
    elif not exact:
        reason = "quote_missing_or_ambiguous"
    else:
        reason = "reviewed_supported_exact_source"
    return {"relation_verdict": verdict, "review_method": method, "review_recorded": recorded, "review_complete": bool(complete),
            "eligible_for_factual_retrieval": bool(accepted), "decision_reason": reason,
            "effective_evidence": quote, "effective_char_start": start, "effective_char_end": end,
            "anchor_method": anchor_method, "reviewer": reviewer, "reviewed_at": reviewed_at}


def summarize_relation_review(audit, review, eligible_articles=None):
    """Report review progress; never extrapolate a targeted sample's accuracy."""
    by_id = audit.set_index("relation_id", drop=False)
    eligible_ids = set(eligible_articles["record_id"]) if eligible_articles is not None else None
    eligible_fingerprints = ({row["record_id"]: source_fingerprint(row)
                              for row in eligible_articles.to_dict("records")}
                             if eligible_articles is not None else None)
    decisions = []
    for row in review.to_dict("records"):
        if row["relation_id"] not in by_id.index:
            raise ValueError("Relation review contains unknown identities")
        decisions.append({**_review_decision(by_id.loc[row["relation_id"]], row, eligible_ids, eligible_fingerprints),
                          "review_selection": row.get("review_selection", "")})
    resolved = pd.DataFrame(decisions)
    important = resolved["review_selection"].eq("important") if len(resolved) else pd.Series(dtype=bool)
    complete = resolved["review_complete"] if len(resolved) else pd.Series(dtype=bool)
    return pd.DataFrame([{
        "relations_available": len(audit), "sampled_relations": len(review),
        "resolved_reviews": int(complete.sum()), "pending_or_unresolved_reviews": len(review) - int(complete.sum()),
        "important_relations": int(important.sum()),
        "important_pending_or_unresolved": int((important & ~complete).sum()),
        "supported_exact_source_in_sample": int(resolved["eligible_for_factual_retrieval"].sum()) if len(resolved) else 0,
        "human_reviewed": int((resolved["review_recorded"] & resolved["review_method"].eq("human")).sum()) if len(resolved) else 0,
        "ai_reviewed": int((resolved["review_recorded"] & resolved["review_method"].eq("ai_assisted")).sum()) if len(resolved) else 0,
        "human_review_complete": bool(len(review) > 0 and complete.all() and resolved["review_method"].eq("human").all()),
        "review_complete": bool(len(review) > 0 and complete.all()),
    }])


def build_retrieval_graph(co_graph, audit, review, mapping, entity_article, eligible_articles, schema):
    """Build a graph with dated association support and reviewed semantic facts.

    Association edges are represented in both directions and explicitly marked
    ``direction='undirected'`` / ``fact_assertion=False``. Their support is the
    intersection of the NER article memberships, never an inferred relationship.
    Original Stage 1 weights remain in the edge audit, not as current evidence.
    """
    eligible = eligible_articles.set_index("record_id", drop=False)
    eligible_ids = set(eligible.index)
    publication_dates = {record_id: _date(value) for record_id, value in eligible["date"].items()}
    map_ids = mapping.set_index("original_id")["canonical_id"].to_dict()
    memberships = {node: set(group["record_id"]) & eligible_ids
                   for node, group in entity_article.groupby("entity_id", sort=False)}
    result = nx.MultiDiGraph()
    result.graph.update({"description": "Stage 1 source-linked associations and attributed source-reviewed assertions",
                         "association_direction": "undirected relations represented bidirectionally",
                         "entity_article_index": "entity_articles.csv.gz",
                         "semantic_policy": "supported review, exact quotation, schema/date/source eligibility required"})
    for node, attrs in co_graph.nodes(data=True):
        canonical = map_ids.get(node, node)
        name, entity_type = _split_id(canonical)
        frequency = len(memberships.get(canonical, set()))
        if entity_type == "GOVERNANCE_THEME":
            flag = THEME_FLAGS.get(name)
            frequency = int(eligible[flag].sum()) if flag in eligible.columns else 0
        result.add_node(canonical, name=name, entity_type=entity_type,
                        article_frequency=frequency, origin="stage1_cooccurrence")
    for node in set(audit.get("subject_id", [])) | set(audit.get("object_id", [])):
        canonical = map_ids.get(node, node)
        if canonical not in result:
            name, entity_type = _split_id(canonical)
            result.add_node(canonical, name=name, entity_type=entity_type,
                            article_frequency=len(memberships.get(canonical, set())), origin="semantic_only")
    audits, handled = [], set()
    minimum = int(schema.get("parameters", {}).get("minimum_cooccurrence", 1))
    for u, v, attrs in co_graph.edges(data=True):
        source, target = map_ids.get(u, u), map_ids.get(v, v)
        relation = attrs.get("relation", "CO_MENTIONED_WITH")
        identity = (tuple(sorted([source, target])), relation)
        if identity in handled or source == target:
            continue
        handled.add(identity)
        if relation == "CO_MENTIONED_WITH":
            support = memberships.get(source, set()) & memberships.get(target, set())
            required = minimum
        elif relation == "ASSOCIATED_WITH_THEME":
            theme_id, entity_id = (source, target) if source.startswith("GOVERNANCE_THEME::") else (target, source)
            flag = THEME_FLAGS.get(_split_id(theme_id)[0])
            theme_articles = set(eligible.index[eligible[flag].astype(bool)]) if flag in eligible.columns else set()
            support = memberships.get(entity_id, set()) & theme_articles
            required = 1
        else:
            raise ValueError(f"Unexpected association relation: {relation}")
        included = len(support) >= required
        audits.append({"edge_kind": "association", "source": source, "target": target,
                       "predicate": relation, "original_weight": attrs.get("weight", 0),
                       "current_support": len(support), "included": included,
                       "decision_reason": "dated_ner_support" if included else "insufficient_eligible_support"})
        if included:
            dates = [publication_dates[record_id] for record_id in support]
            edge_id = stable_id("association", *identity[0], relation)
            edge_attrs = {"relation": relation, "direction": "undirected", "fact_assertion": False,
                          "weight": len(support), "first_publication": min(dates), "last_publication": max(dates),
                          "provenance": "entity_article_index_intersection" if relation == "CO_MENTIONED_WITH"
                                        else "entity_article_index_and_weak_theme_tags",
                          "evidence_group": edge_id}
            result.add_edge(source, target, key=f"{edge_id}:forward", **edge_attrs)
            result.add_edge(target, source, key=f"{edge_id}:reverse", **edge_attrs)
    if not review["relation_id"].is_unique:
        raise ValueError("Relation review IDs must be unique")
    review_by_id = review.set_index("relation_id", drop=False).to_dict("index")
    if set(review_by_id) - set(audit["relation_id"]):
        raise ValueError("Relation review contains unknown identities")
    fingerprints = {row["record_id"]: source_fingerprint(row) for row in eligible_articles.to_dict("records")}
    for row in audit.to_dict("records"):
        decision = _review_decision(row, review_by_id.get(row["relation_id"], {}), eligible_ids, fingerprints)
        source = map_ids.get(row["original_subject_id"], row["subject_id"])
        target = map_ids.get(row["original_object_id"], row["object_id"])
        audits.append({"edge_kind": "semantic", "relation_id": row["relation_id"],
                       "record_id": row["record_id"], "source": source, "target": target,
                       "predicate": row["predicate"], "included": decision["eligible_for_factual_retrieval"],
                       **decision})
        if not decision["eligible_for_factual_retrieval"]:
            continue
        for node in (source, target):
            if node not in result:
                name, entity_type = _split_id(node)
                result.add_node(node, name=name, entity_type=entity_type, origin="reviewed_relation")
        result.add_edge(source, target, key=row["relation_id"], relation=row["predicate"],
                        direction="directed", fact_assertion=True,
                        provenance=f"{decision['review_method']}_reviewed_source_assertion",
                        record_id=int(row["record_id"]), date=row["source_date"],
                        evidence=decision["effective_evidence"], char_start=int(decision["effective_char_start"]),
                        char_end=int(decision["effective_char_end"]), source_fingerprint=row["source_fingerprint"],
                        reviewer=decision["reviewer"], reviewed_at=decision["reviewed_at"],
                        review_method=decision["review_method"],
                        anchor_method=decision["anchor_method"])
    return result, pd.DataFrame(audits)
