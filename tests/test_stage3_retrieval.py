"""Retrieval checks use deterministic local doubles; no model downloads."""

import hashlib
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from src.stage3_rag.retrieval import RetrievalIndex


class Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[101] + list(range(len(text.split()))) + [102] for text in texts]}


class CountingTokenizer(Tokenizer):
    name_or_path = "local-test-tokenizer"

    def __init__(self):
        self.calls = []
        self.vocabulary = {"privacy": 1, "regulation": 2}

    def get_vocab(self):
        return self.vocabulary

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return super().__call__(texts, **kwargs)


class Encoder:
    revision = "test-encoder-revision-1"
    max_seq_length = 256
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = np.zeros((len(texts), 128), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower())[:254]:
                dimension = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 128
                result[i, dimension] += 1
        return result


class Reranker:
    revision = "test-reranker-revision-1"
    max_length = 512

    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return np.array([10 if "rare" in text else 0 for _, text in pairs])


def passage(record_id, text, **fields):
    return {"chunk_id": f"chunk-{record_id}", "record_id": record_id,
            "title": f"Article {record_id}", "date": "2026-01-02", "domain": "example.org",
            "url": f"https://example.org/{record_id}", "text": text,
            "char_start": 0, "char_end": len(text), "source_fingerprint": f"source-{record_id}",
            "article_review_status": "ai_assisted_supported", "article_review_method": "ai_assisted", **fields}


def membership(entity, record_id, **fields):
    return {"entity_id": entity, "record_id": record_id, "date": "2026-01-02",
            "source_fingerprint": f"source-{record_id}", "source": "ner", **fields}


def add_node(graph, node):
    entity_type, name = node.split("::", 1)
    graph.add_node(node, name=name, entity_type=entity_type)


def semantic(record_id, quote, **fields):
    return {"relation": "FINES", "fact_assertion": True, "record_id": record_id,
            "date": "2026-01-02", "source_fingerprint": f"source-{record_id}",
            "evidence": quote, "review_method": "ai_assisted", "reviewer": "source-review",
            "char_start": 0, "char_end": len(quote), **fields}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name)
        self.chunks = pd.DataFrame([
            passage(1, "OpenAI model development announcement and technical release."),
            passage(2, "European Union privacy regulation fined Acme for data practices."),
            passage(3, "Acme rare compliance settlement and privacy remedies."),
        ])

    def tearDown(self):
        self.temporary.cleanup()

    def index(self, chunks=None, **kwargs):
        return RetrievalIndex(self.chunks if chunks is None else chunks, self.cache,
                              encoder=kwargs.pop("encoder", Encoder()), **kwargs)

    def test_sparse_is_query_ranked_and_preserves_source_date(self):
        index = self.index()
        hits = index.rank("European Union privacy regulation", mode="sparse", top_k=2)
        self.assertEqual(hits[0]["record_id"], 2)
        self.assertEqual(hits[0]["date"], "2026-01-02")
        self.assertEqual(hits[0]["source_fingerprint"], "source-2")
        self.assertEqual(hits[0]["graph_evidence"], [])
        self.assertEqual(index.rank("zzzzunmatchable", mode="sparse"), [])

    def test_hybrid_and_dense_ranking_use_query_only(self):
        index = self.index()
        for mode in ["dense", "hybrid"]:
            with self.subTest(mode=mode):
                self.assertEqual(index.rank("OpenAI model development announcement", mode=mode)[0]["record_id"], 1)
        self.assertFalse(index.last_stats["query_truncated"])

    def test_index_and_query_caches_reuse_arrays_and_tfidf(self):
        first_encoder = Encoder()
        first = self.index(encoder=first_encoder)
        expected = first.rank("privacy regulation")
        self.assertEqual(len(first_encoder.calls), 2)
        second_encoder = Encoder()
        second = self.index(encoder=second_encoder)
        actual = second.rank("privacy regulation")
        self.assertTrue(second.index_cache_hit)
        self.assertTrue(second.last_stats["query_embedding_cache_hit"])
        self.assertEqual(second_encoder.calls, [])
        self.assertEqual([h["chunk_id"] for h in expected], [h["chunk_id"] for h in actual])
        self.assertFalse(list(self.cache.rglob("*.pkl")))

    def test_text_id_source_date_and_revision_invalidate_cache(self):
        original = self.index()
        for field, value in [("text", "changed source text"), ("chunk_id", "new-chunk-id"),
                             ("source_fingerprint", "new-source-hash"), ("date", "2026-01-03")]:
            changed = self.chunks.copy()
            changed.loc[0, field] = value
            fresh = self.index(changed)
            self.assertNotEqual(fresh.cache_fingerprint, original.cache_fingerprint)
            self.assertFalse(fresh.index_cache_hit)
        revised = self.index(embedding_revision="new-model-revision")
        self.assertNotEqual(revised.cache_fingerprint, original.cache_fingerprint)

    def test_encoder_truncation_is_audited_and_embedding_text_can_prevent_it(self):
        long_text = "privacy " * 300
        chunks = pd.DataFrame([passage(1, long_text)])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            index = self.index(chunks)
        self.assertTrue(any("truncated" in str(item.message) for item in captured))
        self.assertEqual(index.coverage_report["embedding_truncated_chunks"], 1)
        self.assertTrue(index.rank("privacy")[0]["embedding_truncated"])
        chunks["embedding_text"] = "privacy " * 100
        short = self.index(chunks)
        self.assertEqual(short.coverage_report["embedding_truncated_chunks"], 0)
        self.assertEqual(short.rank("privacy")[0]["text"], long_text)

    def test_coverage_cache_bypasses_tokenization_and_preserves_index_fingerprint(self):
        encoder = Encoder()
        encoder.tokenizer = CountingTokenizer()
        first = self.index(encoder=encoder)
        self.assertFalse(first.coverage_cache_hit)
        self.assertEqual(len(encoder.tokenizer.calls), 1)
        self.assertFalse(encoder.tokenizer.calls[0][1]["truncation"])
        second_encoder = Encoder()
        second_encoder.tokenizer = CountingTokenizer()
        second = self.index(encoder=second_encoder)
        self.assertTrue(second.coverage_report["coverage_cache_hit"])
        self.assertEqual(second_encoder.tokenizer.calls, [])
        self.assertEqual(first.cache_fingerprint, second.cache_fingerprint)
        pd.testing.assert_frame_equal(first.embedding_coverage, second.embedding_coverage)

    def test_coverage_cache_invalidates_for_text_ids_model_sequence_and_tokenizer(self):
        encoder = Encoder()
        encoder.tokenizer = CountingTokenizer()
        original = self.index(encoder=encoder)
        for change in ["text", "chunk_id", "model", "revision", "max_length", "tokenizer", "vocabulary", "library"]:
            with self.subTest(change=change):
                chunks = self.chunks.copy()
                fresh_encoder = Encoder()
                fresh_encoder.tokenizer = CountingTokenizer()
                options = {}
                if change in {"text", "chunk_id"}:
                    chunks.loc[0, change] = "changed"
                elif change == "model":
                    options["embedding_model_id"] = "local-different-model"
                elif change == "revision":
                    options["embedding_revision"] = "revision-2"
                elif change == "max_length":
                    fresh_encoder.max_seq_length = 128
                elif change == "tokenizer":
                    fresh_encoder.tokenizer.name_or_path = "another-local-tokenizer"
                elif change == "vocabulary":
                    fresh_encoder.tokenizer.vocabulary["new-token"] = 3
                versions = {"transformers": "changed"} if change == "library" else None
                if versions is not None:
                    with patch("src.stage3_rag.retrieval._tokenizer_library_versions", return_value=versions):
                        fresh = self.index(chunks, encoder=fresh_encoder, **options)
                else:
                    fresh = self.index(chunks, encoder=fresh_encoder, **options)
                self.assertFalse(fresh.coverage_cache_hit)
                self.assertEqual(len(fresh_encoder.tokenizer.calls), 1)
                self.assertNotEqual(fresh.coverage_cache_fingerprint, original.coverage_cache_fingerprint)

    def test_missing_tokenizer_warns_on_first_pass_and_cached_unknown_coverage(self):
        for repeat in range(2):
            encoder = Encoder()
            encoder.tokenizer = None
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                index = self.index(encoder=encoder)
            self.assertTrue(any("could not be verified" in str(w.message) for w in captured))
            self.assertFalse(index.coverage_report["tokenizer_verified"])
            self.assertEqual(index.coverage_cache_hit, bool(repeat))
            self.assertTrue(index.embedding_coverage.input_tokens.eq(-1).all())

    def test_invalid_coverage_array_is_recomputed(self):
        encoder = Encoder()
        encoder.tokenizer = CountingTokenizer()
        first = self.index(encoder=encoder)
        path = self.cache / "embedding_coverage" / first.coverage_cache_fingerprint / "counts.npz"
        np.savez_compressed(path, input_tokens=np.array([999]))
        fresh_encoder = Encoder()
        fresh_encoder.tokenizer = CountingTokenizer()
        fresh = self.index(encoder=fresh_encoder)
        self.assertFalse(fresh.coverage_cache_hit)
        self.assertEqual(len(fresh_encoder.tokenizer.calls), 1)
        pd.testing.assert_frame_equal(first.embedding_coverage, fresh.embedding_coverage)

    def test_semantic_only_source_is_found_without_ner_membership(self):
        graph = nx.MultiDiGraph()
        for node in ["REGULATOR::Authority", "ORG::Acme"]:
            add_node(graph, node)
        graph.add_edge("REGULATOR::Authority", "ORG::Acme",
                       **semantic(3, "Acme rare compliance settlement"))
        index = self.index()
        # Source 3 has no NER index entry; the reviewed edge supplies provenance.
        self.assertNotIn(3, [hit["record_id"] for hit in index.rank("Authority regulation", mode="sparse", top_k=2)])
        hits = index.retrieve("Authority regulation", mode="sparse", top_k=2,
                              candidate_k=2, graph=graph, entity_articles=pd.DataFrame(), graph_weight=2)
        source = next(hit for hit in hits if hit["record_id"] == 3)
        evidence = source["graph_evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["kind"], "reviewed_semantic")
        self.assertTrue(evidence[0]["fact_assertion"])
        self.assertEqual(evidence[0]["record_id"], source["record_id"])
        self.assertEqual(evidence[0]["date"], source["date"])

    def test_mutating_membership_provenance_invalidates_graph_context(self):
        graph = nx.MultiDiGraph()
        add_node(graph, "ORG::Acme")
        members = pd.DataFrame([membership("ORG::Acme", 3)])
        index = self.index()
        initial = index.retrieve("Acme privacy", graph=graph, entity_articles=members, top_k=3)
        self.assertTrue(any(hit["graph_evidence"] for hit in initial))
        members.loc[0, "source_fingerprint"] = "changed-source"
        updated = index.retrieve("Acme privacy", graph=graph, entity_articles=members, top_k=3)
        self.assertTrue(all(not hit["graph_evidence"] for hit in updated))

    def test_association_requires_both_nodes_in_same_current_source(self):
        graph = nx.MultiDiGraph()
        for node in ["ORG::OpenAI", "ORG::Acme"]:
            add_node(graph, node)
        graph.add_edge("ORG::OpenAI", "ORG::Acme", relation="CO_MENTIONED_WITH", fact_assertion=False, weight=999)
        memberships = pd.DataFrame([membership("ORG::OpenAI", 1), membership("ORG::OpenAI", 3),
                                    membership("ORG::Acme", 2), membership("ORG::Acme", 3)])
        hits = self.index().retrieve("OpenAI privacy", top_k=3, graph=graph, entity_articles=memberships)
        for hit in hits:
            associations = [item for item in hit["graph_evidence"] if item["kind"] == "association"]
            if hit["record_id"] == 3:
                self.assertEqual(len(associations), 1)
                self.assertFalse(associations[0]["fact_assertion"])
                self.assertTrue(associations[0]["retrieval_only"])
            else:
                self.assertEqual(associations, [])

    def test_stale_dates_fingerprints_unreviewed_edges_and_themes_cannot_assert_facts(self):
        graph = nx.MultiDiGraph()
        for node in ["ORG::Acme", "REGULATOR::Authority", "GOVERNANCE_THEME::Privacy"]:
            add_node(graph, node)
        graph.add_edge("REGULATOR::Authority", "ORG::Acme", **semantic(3, "privacy", date="2025-01-02"))
        graph.add_edge("REGULATOR::Authority", "ORG::Acme", **semantic(3, "privacy", source_fingerprint="stale"))
        graph.add_edge("REGULATOR::Authority", "ORG::Acme", **semantic(3, "privacy", review_method=""))
        graph.add_edge("GOVERNANCE_THEME::Privacy", "ORG::Acme", **semantic(3, "privacy"))
        members = pd.DataFrame([membership("REGULATOR::Authority", 3, date="2025-01-02")])
        hits = self.index().retrieve("Authority Privacy", graph=graph, entity_articles=members, top_k=3)
        self.assertTrue(all(not hit["graph_evidence"] for hit in hits))

    def test_boundary_and_abbreviation_anchors(self):
        graph = nx.MultiDiGraph()
        for node in ["ORG::OpenAI", "GPE::United States", "GPE::European Union"]:
            add_node(graph, node)
        index = self.index(aliases={"America": "United States"})
        self.assertEqual(set(index._anchors("OpenAI policy in EU and America", graph, None)),
                         {"ORG::OpenAI", "GPE::United States", "GPE::European Union"})
        self.assertEqual(index._anchors("help us explain this OpenAIR policy", graph, None), [])

    def test_high_degree_graph_expansion_is_capped_and_not_degree_ranked(self):
        graph = nx.MultiDiGraph()
        add_node(graph, "ORG::Anchor")
        chunks, rows = [], []
        for i in range(1, 31):
            node = f"ORG::Neighbor{i}"
            add_node(graph, node)
            graph.add_edge("ORG::Anchor", node, relation="CO_MENTIONED_WITH", fact_assertion=False, weight=1000-i)
            chunks.append(passage(i, "Anchor privacy enforcement" if i == 30 else f"Anchor unrelated sports {i}"))
            rows.extend([membership("ORG::Anchor", i), membership(node, i)])
        index = self.index(pd.DataFrame(chunks))
        hits = index.retrieve("Anchor privacy enforcement", graph=graph, entity_articles=pd.DataFrame(rows),
                              graph_neighbor_cap=2, graph_article_cap=3, top_k=2, candidate_k=4)
        self.assertEqual(hits[0]["record_id"], 30)
        self.assertLessEqual(index.last_stats["graph_neighbors_selected"], 2)
        self.assertLessEqual(index.last_stats["graph_candidate_articles"], 3)

    def test_reranking_uses_same_fixed_candidate_pool_and_caches_pairs(self):
        reranker = Reranker()
        index = self.index(reranker=reranker)
        index.retrieve("privacy", top_k=2, candidate_k=3)
        baseline_pool = index.last_stats["candidate_chunk_ids"]
        hits = index.retrieve("privacy", top_k=2, candidate_k=3, rerank=True)
        self.assertEqual(index.last_stats["candidate_chunk_ids"], baseline_pool)
        self.assertEqual(hits[0]["record_id"], 3)
        self.assertEqual(index.last_stats["reranker_pairs_scored"], 3)
        self.assertEqual(len(reranker.calls[0]), 3)
        index.retrieve("privacy", top_k=2, candidate_k=3, rerank=True)
        self.assertEqual(index.last_stats["reranker_pairs_scored"], 0)
        self.assertEqual(index.last_stats["reranker_pair_cache_hits"], 3)
        self.assertEqual(len(reranker.calls), 1)

        changed = self.chunks.copy()
        changed.loc[0, "text"] = "Changed OpenAI passage"
        fresh = self.index(changed, reranker=reranker)
        fresh.retrieve("privacy", top_k=2, candidate_k=3, rerank=True)
        self.assertEqual(fresh.last_stats["reranker_pairs_scored"], 1)
        self.assertEqual(fresh.last_stats["reranker_pair_cache_hits"], 2)

    def test_invalid_inputs_fail_clearly(self):
        index = self.index()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            index.retrieve("")
        with self.assertRaisesRegex(ValueError, "graph mode"):
            index.retrieve("privacy", mode="graph")
        with self.assertRaisesRegex(ValueError, "local_files_only"):
            self.index(local_files_only=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            self.index(pd.concat([self.chunks, self.chunks]))


if __name__ == "__main__":
    unittest.main()
