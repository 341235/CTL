"""Regression checks for evidence, identity and retrieval eligibility policies."""

import tempfile
from pathlib import Path
import unittest

import networkx as nx
import pandas as pd

from src.stage1_validation.common import source_fingerprint
from src.stage1_validation.graphs import (
    audit_relations, build_entity_links, build_retrieval_graph,
    prepare_relation_review, supporting_articles, summarize_relation_review,
)


SCHEMA = {"node_types": {"ORG": "", "MODEL": "", "PERSON": "", "GOVERNANCE_THEME": ""},
          "relation_types": {"DEVELOPS": {"source": "LLM extraction"}}}


def corpus_row(record_id=1, text="OpenAI released Sora.", date="2026-01-02"):
    row = {"record_id": record_id, "title": "Release", "content_text": text,
           "date": date, "domain": "example", "security": False, "accountability": True,
           "bias_fairness": False, "regulation": False, "article_review_status": "unreviewed"}
    row["source_fingerprint"] = source_fingerprint(row)
    return row


def relation_row(evidence="OpenAI released Sora", start=9, end=29):
    return {"record_id": 1, "date": "2026-01-02", "domain": "example", "subject": "OpenAI",
            "subject_type": "ORG", "predicate": "DEVELOPS", "object": "Sora", "object_type": "MODEL",
            "evidence": evidence, "char_start": start, "char_end": end}


class GraphValidationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = pd.DataFrame([corpus_row()])
        self.graph = nx.Graph()
        self.graph.add_node("ORG::OpenAI", name="OpenAI", entity_type="ORG")
        self.graph.add_node("MODEL::Sora", name="Sora", entity_type="MODEL")
        self.graph.add_edge("ORG::OpenAI", "MODEL::Sora", relation="CO_MENTIONED_WITH", weight=1)
        self.entities = pd.DataFrame([{"record_id": 1, "entities_normalized": [("OpenAI", "ORG"), ("Sora", "MODEL")]}])

    def audit(self, relation=None, corpus=None):
        return audit_relations(pd.DataFrame([relation or relation_row()]),
                               self.corpus if corpus is None else corpus, SCHEMA, {}, {})

    def review(self, audit, **fields):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_relation_review(audit, self.graph, Path(directory) / "reviews.csv", sample_size=1)
        result["review_method"] = "human"
        for field, value in fields.items():
            result.loc[:, field] = value
        return result

    def build(self, audit, review, corpus=None, entities=None):
        corpus = self.corpus if corpus is None else corpus
        mapping, index, _ = build_entity_links(self.graph, audit, self.entities if entities is None else entities,
                                               corpus, {}, {})
        graph, edges = build_retrieval_graph(self.graph, audit, review, mapping, index, corpus, SCHEMA)
        return graph, edges, mapping, index

    def test_nonmissing_offsets_are_not_grounding(self):
        row = self.audit(relation_row(evidence="OpenAI invented Sora", start=9, end=29)).iloc[0]
        self.assertFalse(row.exact_span_match)
        self.assertFalse(row.evidence_in_source)
        self.assertEqual(row.alignment_status, "no_exact_source_quote")

    def test_changed_extraction_cannot_inherit_relation_review_identity(self):
        original_id = self.audit().iloc[0].relation_id
        changed_quote = self.audit(relation_row(evidence="released Sora", start=16, end=29)).iloc[0].relation_id
        changed_offset = self.audit(relation_row(start=0, end=20)).iloc[0].relation_id
        self.assertNotEqual(original_id, changed_quote)
        self.assertNotEqual(original_id, changed_offset)

    def test_title_prefix_offsets_are_used_and_unique_quote_reanchors(self):
        audit = self.audit(relation_row(start=0, end=20))
        self.assertFalse(audit.iloc[0].exact_span_match)
        self.assertEqual(audit.iloc[0].reanchored_start, len("Release. "))
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        graph, edges, _, _ = self.build(audit, review)
        semantic = [attrs for _, _, attrs in graph.edges(data=True) if attrs["fact_assertion"]]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0]["char_start"], 9)
        self.assertEqual(semantic[0]["anchor_method"], "unique_exact_source_reanchor")

    def test_repeated_quote_requires_correct_offsets(self):
        corpus = pd.DataFrame([corpus_row(text="OpenAI released Sora. OpenAI released Sora.")])
        audit = self.audit(relation_row(start=0, end=20), corpus)
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        graph, edges, _, _ = self.build(audit, review, corpus)
        self.assertFalse(any(attrs["fact_assertion"] for _, _, attrs in graph.edges(data=True)))
        self.assertEqual(edges.loc[edges.edge_kind == "semantic", "decision_reason"].iloc[0], "quote_missing_or_ambiguous")
        review["reviewed_char_start"], review["reviewed_char_end"] = "9", "29"
        graph, _, _, _ = self.build(audit, review, corpus)
        self.assertTrue(any(attrs["fact_assertion"] for _, _, attrs in graph.edges(data=True)))

    def test_type_conflicts_are_reported_without_merging(self):
        entities = pd.DataFrame([{"record_id": 1, "entities_normalized": [("OpenAI", "ORG"), ("Sora", "PERSON")]}])
        audit = self.audit()
        _, _, mapping, index = self.build(audit, self.review(audit), entities=entities)
        conflict = mapping.loc[mapping.original_id == "PERSON::Sora"].iloc[0]
        self.assertEqual(conflict.canonical_id, "PERSON::Sora")
        self.assertTrue(conflict.type_conflict)
        self.assertNotIn("MODEL::Sora", set(index.entity_id))

    def test_unique_casefold_match_preserves_type_and_source_membership(self):
        entities = pd.DataFrame([{"record_id": 1, "entities_normalized": [("OPENAI", "ORG"), ("Sora", "MODEL")]}])
        audit = self.audit()
        _, _, mapping, index = self.build(audit, self.review(audit), entities=entities)
        resolved = mapping.loc[mapping.original_id == "ORG::OPENAI"].iloc[0]
        self.assertEqual(resolved.canonical_id, "ORG::OpenAI")
        self.assertEqual(supporting_articles(index, "ORG::OpenAI").record_id.tolist(), [1])

    def test_unreviewed_semantics_are_excluded_and_associations_are_not_facts(self):
        audit = self.audit()
        graph, edges, _, _ = self.build(audit, self.review(audit))
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertTrue(all(attrs["direction"] == "undirected" and not attrs["fact_assertion"]
                            for _, _, attrs in graph.edges(data=True)))
        self.assertEqual(edges.loc[edges.edge_kind == "semantic", "decision_reason"].iloc[0], "unreviewed")
        summary = summarize_relation_review(audit, self.review(audit)).iloc[0]
        self.assertFalse(summary.review_complete)
        self.assertEqual(summary.important_pending_or_unresolved, 1)

    def test_support_intersection_and_date_filtered_population(self):
        corpus = pd.DataFrame([corpus_row(1), corpus_row(2, date="2026-09-02")])
        entities = pd.DataFrame([
            {"record_id": 1, "entities_normalized": [("OpenAI", "ORG"), ("Sora", "MODEL")]},
            {"record_id": 2, "entities_normalized": [("OpenAI", "ORG"), ("Sora", "MODEL")]},
        ])
        audit = self.audit()
        eligible = corpus.loc[corpus.record_id == 1]
        graph, _, _, index = self.build(audit, self.review(audit), eligible, entities)
        support = supporting_articles(index, "ORG::OpenAI", "MODEL::Sora")
        self.assertEqual(support.record_id.tolist(), [1])
        self.assertEqual(next(iter(graph.edges(data=True)))[2]["weight"], 1)

    def test_stale_review_is_rejected(self):
        audit = self.audit()
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        review["source_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "Stale source fingerprint"):
            self.build(audit, review)

    def test_relation_date_mismatch_prevents_fact_export(self):
        relation = relation_row()
        relation["date"] = "2025-01-01"
        audit = self.audit(relation)
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        graph, edges, _, _ = self.build(audit, review)
        self.assertFalse(any(attrs["fact_assertion"] for _, _, attrs in graph.edges(data=True)))
        self.assertEqual(edges.loc[edges.edge_kind == "semantic", "decision_reason"].iloc[0], "schema_source_or_date_invalid")

    def test_resolved_review_stays_recorded_when_article_is_excluded(self):
        audit = self.audit()
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        eligible = self.corpus.iloc[:0]
        summary = summarize_relation_review(audit, review, eligible).iloc[0]
        self.assertTrue(summary.review_complete)
        self.assertEqual(summary.supported_exact_source_in_sample, 0)
        graph, edges, _, _ = self.build(audit, review, eligible)
        self.assertEqual(graph.number_of_edges(), 0)

    def test_supported_review_cannot_reuse_stale_audit_text(self):
        audit = self.audit()
        review = self.review(audit, relation_verdict="supported", reviewer="Reviewer", reviewed_at="2026-11-05")
        changed = self.corpus.copy()
        changed.loc[:, "content_text"] = "Different article content."
        summary = summarize_relation_review(audit, review, changed).iloc[0]
        self.assertFalse(summary.review_complete)
        graph, edges, _, _ = self.build(audit, review, changed)
        self.assertFalse(any(attrs["fact_assertion"] for _, _, attrs in graph.edges(data=True)))
        self.assertEqual(edges.loc[edges.edge_kind == "semantic", "decision_reason"].iloc[0], "eligible_source_changed")

    def test_rejections_require_explanatory_notes(self):
        audit = self.audit()
        review = self.review(audit, relation_verdict="incorrect", reviewer="Reviewer", reviewed_at="2026-11-05")
        self.assertFalse(summarize_relation_review(audit, review).iloc[0].review_complete)
        review["review_notes"] = "Wrong relation type."
        self.assertTrue(summarize_relation_review(audit, review).iloc[0].review_complete)

    def test_ai_fact_provenance_cannot_become_human_review(self):
        audit = self.audit()
        review = self.review(audit, relation_verdict="supported", review_method="ai_assisted",
                             reviewer="ChatGPT / Codex", reviewed_at="2026-11-05")
        graph, _, _, _ = self.build(audit, review)
        fact = next(d for _, _, d in graph.edges(data=True) if d["fact_assertion"])
        self.assertEqual(fact["review_method"], "ai_assisted")
        self.assertEqual(fact["provenance"], "ai_assisted_reviewed_source_assertion")
        summary = summarize_relation_review(audit, review).iloc[0]
        self.assertTrue(summary.review_complete)
        self.assertFalse(summary.human_review_complete)
        self.assertEqual(summary.ai_reviewed, 1)


if __name__ == "__main__":
    unittest.main()
