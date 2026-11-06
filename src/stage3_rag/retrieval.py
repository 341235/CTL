"""Offline passage retrieval and conservative, source-bound graph expansion.

The sparse baseline is word TF-IDF (unigrams and bigrams), not BM25. Dense
retrieval uses a cached pretrained encoder without fitting it. Association edges
are retrieval hints; only attributed, quoted semantic edges are assertions.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import hashlib
import json
import re
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


VERSION = "stage3-retrieval-v1"
COVERAGE_CACHE_VERSION = "embedding-coverage-v1"
SPARSE_CONFIG = {"ngram_range": (1, 2), "stop_words": "english", "sublinear_tf": True,
                 "max_features": 120000, "min_df": 1}
SAFE_ENTITY_TYPES = {"ORG", "PERSON", "GPE", "LOC", "LAW", "REGULATOR", "MODEL", "PRODUCT", "TECHNOLOGY", "STANDARD"}
ABBREVIATIONS = {
    "AI": "Artificial Intelligence", "EU": "European Union", "E.U.": "European Union",
    "US": "United States", "U.S.": "United States", "USA": "United States",
    "UK": "United Kingdom", "U.K.": "United Kingdom",
    "FTC": "Federal Trade Commission", "DOJ": "Department of Justice",
    "NIST": "National Institute of Standards and Technology",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _text(value):
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return ""
    return str(value)


@lru_cache(maxsize=4096)
def _date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _truth(value):
    return value is True or isinstance(value, (np.bool_, bool)) and bool(value) or str(value).lower() == "true"


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _model_location(model_id, revision=None):
    """Resolve only existing local snapshots; this function never downloads."""
    path = Path(model_id).expanduser()
    if path.is_dir():
        return str(path), str(path.resolve())
    from huggingface_hub import snapshot_download
    location = snapshot_download(model_id, revision=revision, local_files_only=True)
    return location, Path(location).name


def _normalise(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _tokenizer_library_versions():
    versions = {}
    for package in ("transformers", "tokenizers", "sentence-transformers"):
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _tokenizer_identity(tokenizer):
    if tokenizer is None:
        return None
    identity = {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"}
    for name in ("name_or_path", "init_kwargs", "model_max_length", "padding_side",
                 "truncation_side", "special_tokens_map"):
        if hasattr(tokenizer, name):
            identity[name] = getattr(tokenizer, name)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        state = json.loads(backend.to_str())
        # These two fields describe the most recent call, not tokenizer identity.
        state["padding"] = None
        state["truncation"] = None
        identity["backend_fingerprint"] = _digest(state)
    elif hasattr(tokenizer, "get_vocab"):
        identity["vocabulary_fingerprint"] = _digest(tokenizer.get_vocab())
    return identity


class RetrievalIndex:
    """Build/load a fingerprinted passage index, with no answer/gold inputs.

    ``encoder`` and ``reranker`` permit offline test doubles or already-loaded
    models. Cached sparse arrays and vocabulary use JSON/NPZ, never pickle.
    ``rank`` and ``retrieve`` return complete source rows plus retrieval scores.
    """

    def __init__(self, chunks, cache_dir, embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
                 local_files_only=True, seed=42, *, encoder=None, reranker=None,
                 aliases=None, embedding_revision=None,
                 reranker_model_id="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        required = {"chunk_id", "record_id", "text", "date", "source_fingerprint"}
        if not required.issubset(chunks.columns):
            raise ValueError(f"Missing chunk columns: {sorted(required - set(chunks.columns))}")
        if chunks.empty or not chunks["chunk_id"].is_unique:
            raise ValueError("Chunks must be nonempty with unique chunk_id values")
        if not local_files_only:
            raise ValueError("This retrieval experiment requires local_files_only=True; model downloads are disabled")
        self.chunks = chunks.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_id = embedding_model_id
        self.reranker_model_id = reranker_model_id
        self.seed = seed
        self.aliases = dict(aliases or {})
        self.reranker = reranker
        self._reranker_identity = None
        self._query_vectors = {}
        self._graph_contexts = {}
        self.last_stats = {}
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            location, revision = _model_location(embedding_model_id, embedding_revision)
            encoder = SentenceTransformer(location, device="cpu", local_files_only=True,
                                          trust_remote_code=False)
        else:
            revision = embedding_revision or getattr(encoder, "revision", "injected-encoder")
        self.encoder = encoder
        self.model_revision = str(revision)
        self.max_seq_length = int(getattr(encoder, "max_seq_length", 256))
        self._texts = self.chunks.get("embedding_text", self.chunks["text"]).fillna("").astype(str).tolist()
        self.embedding_coverage = self._check_coverage(self._texts)
        self.coverage_report = {
            "chunks": len(self.chunks), "embedding_max_seq_length": self.max_seq_length,
            "tokenizer_verified": bool(self.embedding_coverage["tokenizer_verified"].all()),
            "embedding_truncated_chunks": int(self.embedding_coverage["embedding_truncated"].sum()),
            "input_tokens_max": int(self.embedding_coverage["input_tokens"].max()),
            "coverage_cache_hit": self.coverage_cache_hit,
            "encoding_scope": "passage embedding_text; encoder input can be shorter than the displayed source passage",
        }
        self.cache_fingerprint = _digest({
            "version": VERSION, "chunks": self.chunks.to_dict("records"), "texts": self._texts,
            "model": embedding_model_id, "revision": self.model_revision,
            "max_seq_length": self.max_seq_length, "sparse": SPARSE_CONFIG, "seed": seed,
        })
        self._index_dir = self.cache_dir / "indices" / self.cache_fingerprint
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._build_or_load()
        self._rows = self.chunks.to_dict("records")
        self._record_positions = defaultdict(list)
        self._source_meta = {}
        for position, row in enumerate(self._rows):
            record = _text(row["record_id"])
            metadata = (_text(row["source_fingerprint"]), _date(row["date"]))
            if record in self._source_meta and metadata != self._source_meta[record]:
                raise ValueError(f"Conflicting source fingerprints/dates for record_id {record}")
            self._source_meta[record] = metadata
            self._record_positions[record].append(position)

    def _token_counts(self, texts):
        tokenizer = getattr(self.encoder, "tokenizer", None)
        if tokenizer is None:
            return [-1] * len(texts), False
        counts = []
        # Untruncated tokenizer calls provide an explicit coverage audit; only
        # the encoder call below applies its documented maximum sequence size.
        for start in range(0, len(texts), 256):
            batch = tokenizer(texts[start:start + 256], add_special_tokens=True,
                              truncation=False, padding=False)
            counts.extend(len(ids) for ids in batch["input_ids"])
        return counts, True

    def _check_coverage(self, texts):
        tokenizer = getattr(self.encoder, "tokenizer", None)
        identity = {"version": COVERAGE_CACHE_VERSION, "model": self.embedding_model_id,
                    "revision": self.model_revision, "max_seq_length": self.max_seq_length,
                    "tokenizer": _tokenizer_identity(tokenizer),
                    "libraries": _tokenizer_library_versions()}
        # Stream passage strings into the key instead of copying the complete
        # corpus into one additional JSON document during a cache reload.
        digest = hashlib.sha256(_digest(identity).encode())
        for chunk_id, text in zip(self.chunks["chunk_id"], texts):
            digest.update(json.dumps([chunk_id, text], ensure_ascii=False, default=str).encode())
            digest.update(b"\n")
        self.coverage_cache_fingerprint = digest.hexdigest()
        directory = self.cache_dir / "embedding_coverage" / self.coverage_cache_fingerprint
        manifest_path = directory / "manifest.json"
        counts_path = directory / "counts.npz"
        self.coverage_cache_hit = False
        if manifest_path.exists() and counts_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                with np.load(counts_path, allow_pickle=False) as arrays:
                    cached = arrays["input_tokens"]
                verified = manifest["tokenizer_verified"]
                valid_counts = (cached.dtype.kind in "iu" and cached.shape == (len(texts),)
                                and (bool((cached >= 0).all()) if verified else bool((cached == -1).all())))
                if (manifest.get("fingerprint") == self.coverage_cache_fingerprint
                        and manifest.get("chunk_count") == len(texts)
                        and isinstance(verified, bool) and verified == (tokenizer is not None)
                        and valid_counts):
                    counts = cached
                    self.coverage_cache_hit = True
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if not self.coverage_cache_hit:
            counts, verified = self._token_counts(texts)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(counts_path, input_tokens=np.asarray(counts, dtype=np.int64))
            _json_write(manifest_path, {"fingerprint": self.coverage_cache_fingerprint,
                                       "chunk_count": len(texts), "tokenizer_verified": verified,
                                       "identity": identity})
        coverage = pd.DataFrame({"chunk_id": self.chunks["chunk_id"], "input_tokens": counts,
                                 "tokenizer_verified": verified,
                                 "embedding_truncated": np.asarray(counts) > self.max_seq_length})
        count = int(coverage["embedding_truncated"].sum())
        if count:
            warnings.warn(f"{count} passage embedding inputs exceed {self.max_seq_length} tokens and will be "
                          "truncated by the encoder; see embedding_coverage. Full passage text is preserved.",
                          RuntimeWarning, stacklevel=3)
        if not verified:
            warnings.warn("Injected encoder has no tokenizer: embedding token coverage could not be verified",
                          RuntimeWarning, stacklevel=3)
        return coverage

    def _encode(self, texts):
        return _normalise(self.encoder.encode(list(texts), batch_size=64,
                                              show_progress_bar=False, convert_to_numpy=True,
                                              normalize_embeddings=True))

    def _build_or_load(self):
        manifest_path = self._index_dir / "manifest.json"
        cache_hit = False
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("fingerprint") == self.cache_fingerprint:
                try:
                    self.embeddings = np.load(self._index_dir / "dense.npy", allow_pickle=False)
                    self.sparse_matrix = sparse.load_npz(self._index_dir / "sparse.npz")
                    vocabulary = json.loads((self._index_dir / "vocabulary.json").read_text())
                    self.vectorizer = TfidfVectorizer(**SPARSE_CONFIG, vocabulary=vocabulary, dtype=np.float32)
                    self.vectorizer.idf_ = np.load(self._index_dir / "idf.npy", allow_pickle=False)
                    cache_hit = (len(self.embeddings) == len(self.chunks)
                                 and self.sparse_matrix.shape[0] == len(self.chunks))
                except (OSError, ValueError, KeyError):
                    cache_hit = False
        if not cache_hit:
            self.vectorizer = TfidfVectorizer(**SPARSE_CONFIG, dtype=np.float32)
            try:
                self.sparse_matrix = self.vectorizer.fit_transform(self.chunks["text"].fillna("").astype(str))
            except ValueError as error:
                if "empty vocabulary" not in str(error):
                    raise
                self.vectorizer = TfidfVectorizer(**SPARSE_CONFIG, vocabulary={"__empty__": 0}, dtype=np.float32)
                self.sparse_matrix = self.vectorizer.fit_transform(self.chunks["text"].fillna("").astype(str))
            self.embeddings = self._encode(self._texts)
            np.save(self._index_dir / "dense.npy", self.embeddings, allow_pickle=False)
            sparse.save_npz(self._index_dir / "sparse.npz", self.sparse_matrix)
            np.save(self._index_dir / "idf.npy", self.vectorizer.idf_, allow_pickle=False)
            _json_write(self._index_dir / "vocabulary.json", {term: int(index) for term, index in self.vectorizer.vocabulary_.items()})
            _json_write(manifest_path, {"fingerprint": self.cache_fingerprint, "version": VERSION,
                                       "model": self.embedding_model_id, "revision": self.model_revision,
                                       "coverage": self.coverage_report, "sparse_config": SPARSE_CONFIG})
        self.index_cache_hit = cache_hit

    def _query(self, question):
        key = _digest({"model": self.embedding_model_id, "revision": self.model_revision,
                       "max_seq_length": self.max_seq_length, "question": question})
        counts, verified = self._token_counts([question])
        self.last_stats.update(query_tokens=counts[0], query_tokenizer_verified=verified,
                               query_truncated=counts[0] > self.max_seq_length)
        if counts[0] > self.max_seq_length:
            warnings.warn(f"Query exceeds {self.max_seq_length} tokens and is truncated for dense retrieval",
                          RuntimeWarning, stacklevel=3)
        if key in self._query_vectors:
            self.last_stats["query_embedding_cache_hit"] = True
            return self._query_vectors[key]
        path = self.cache_dir / "queries" / f"{key}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            self.last_stats["query_embedding_cache_hit"] = True
        else:
            vector = self._encode([question])[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector, allow_pickle=False)
            self.last_stats["query_embedding_cache_hit"] = False
        self._query_vectors[key] = vector
        return vector

    def _scores(self, question, need_dense=True):
        query = self.vectorizer.transform([question])
        lexical = np.asarray((self.sparse_matrix @ query.T).toarray()).ravel()
        dense = self.embeddings @ self._query(question) if need_dense else np.zeros(len(self.chunks), dtype=float)
        self.last_stats["dense_query_computed"] = bool(need_dense)
        return lexical, dense

    def _order(self, scores, indices=None):
        indices = np.arange(len(scores)) if indices is None else np.asarray(list(indices), dtype=int)
        return sorted(indices.tolist(), key=lambda index: (-float(scores[index]), str(self._rows[index]["chunk_id"])))

    def _base_scores(self, lexical, dense, mode):
        if mode == "sparse":
            return lexical.copy()
        if mode == "dense":
            return dense.copy()
        if mode not in {"hybrid", "graph"}:
            raise ValueError("mode must be sparse, dense, hybrid, or graph")
        fused = np.zeros(len(self.chunks), dtype=np.float64)
        for values in (lexical, dense):
            order = self._order(values)
            for rank, index in enumerate(order, start=1):
                if values is lexical and values[index] <= 0:
                    continue
                fused[index] += 1.0 / (60 + rank)
        return fused

    def _hit(self, index, score, method, lexical, dense, evidence=None):
        row = dict(self._rows[index])
        row.update(score=float(score), sparse_score=float(lexical[index]), dense_score=float(dense[index]),
                   retrieval_method=method, graph_evidence=evidence or [],
                   embedding_input_tokens=int(self.embedding_coverage.iloc[index]["input_tokens"]),
                   embedding_truncated=bool(self.embedding_coverage.iloc[index]["embedding_truncated"]))
        return row

    def rank(self, question, mode="hybrid", top_k=20):
        return self.retrieve(question, mode=mode, top_k=top_k, candidate_k=top_k)

    def _anchors(self, question, graph, aliases):
        names = defaultdict(set)
        for node, attrs in graph.nodes(data=True):
            entity_type = _text(attrs.get("entity_type")) or str(node).split("::", 1)[0]
            if entity_type not in SAFE_ENTITY_TYPES or "::" not in str(node):
                continue
            name = _text(attrs.get("name")) or str(node).split("::", 1)[1]
            names[name.casefold()].add(node)
        lookup = dict(ABBREVIATIONS)
        lookup.update(self.aliases)
        lookup.update(aliases or {})
        candidates = [(name, node) for name, nodes in names.items() for node in nodes if len(name) >= 2]
        for alias, canonical in lookup.items():
            if isinstance(canonical, str):
                matches = {canonical} if canonical in graph else names.get(canonical.casefold(), set())
                candidates.extend((str(alias), node) for node in matches)
        matched = {}
        for name, node in candidates:
            # Short abbreviations are case-sensitive to avoid matching ordinary
            # pronouns such as "us" or "uk" inside a larger word.
            flags = 0 if len(name) <= 3 and name.isupper() else re.IGNORECASE
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, flags):
                matched[node] = max(matched.get(node, 0), len(name))
        return sorted(matched, key=lambda node: (-matched[node], str(node)))[:4]

    def _source_matches(self, row):
        record = _text(row.get("record_id"))
        expected = self._source_meta.get(record)
        return bool(expected and expected[0] and expected[1]
                    and expected == (_text(row.get("source_fingerprint")), _date(row.get("date"))))

    def _memberships(self, graph, entity_articles):
        identity = "none"
        if entity_articles is not None and not entity_articles.empty:
            columns = [column for column in ["entity_id", "record_id", "date", "source_fingerprint", "source"]
                       if column in entity_articles.columns]
            identity = hashlib.sha256(pd.util.hash_pandas_object(
                entity_articles[columns], index=False).values.tobytes()).hexdigest()
        key = (id(graph), id(entity_articles), identity)
        if key in self._graph_contexts:
            return self._graph_contexts[key][2]
        memberships = defaultdict(dict)
        if entity_articles is not None and not entity_articles.empty:
            required = {"entity_id", "record_id", "date", "source_fingerprint"}
            if not required.issubset(entity_articles.columns):
                raise ValueError(f"entity_articles missing columns: {sorted(required - set(entity_articles.columns))}")
            for row in entity_articles.to_dict("records"):
                if self._source_matches(row) and row.get("source", "ner") == "ner":
                    memberships[row["entity_id"]][_text(row["record_id"])] = row
        # Holding references prevents object-id reuse from borrowing memberships.
        self._graph_contexts[key] = (graph, entity_articles, memberships)
        return memberships

    def _semantic(self, attrs):
        return (_truth(attrs.get("fact_assertion")) and self._source_matches(attrs)
                and bool(_text(attrs.get("evidence")).strip())
                and attrs.get("review_method") in {"human", "ai_assisted"})

    def _graph_candidates(self, question, graph, entity_articles, lexical, dense,
                          aliases, neighbor_cap, article_cap):
        anchors = self._anchors(question, graph, aliases)
        memberships = self._memberships(graph, entity_articles)
        relevance = 0.5 * lexical + 0.5 * np.clip(dense, 0, 1)
        evidence_by_record = defaultdict(list)
        semantic_records = set()
        selected_neighbors = set()
        candidate_records = set()
        record_relevance = {record: max(float(relevance[i]) for i in positions)
                            for record, positions in self._record_positions.items()}

        def relevance_of(records):
            return max((record_relevance.get(record, 0.0) for record in records), default=0)

        def limited_records(records):
            return sorted(records, key=lambda record: (-record_relevance.get(record, 0.0), record))[:max(0, int(article_cap))]

        def add(record, evidence):
            if record in self._record_positions:
                candidate_records.add(record)
                if evidence not in evidence_by_record[record]:
                    evidence_by_record[record].append(evidence)

        for anchor in anchors:
            for record in limited_records(memberships.get(anchor, {})):
                row = memberships[anchor][record]
                add(record, {"kind": "entity_mention", "entity_id": anchor, "record_id": self._rows[self._record_positions[record][0]]["record_id"],
                             "date": row["date"], "fact_assertion": False, "retrieval_only": True,
                             "support": "source-bound NER membership"})
            edges = list(graph.edges(anchor, data=True))
            if graph.is_directed():
                edges += list(graph.in_edges(anchor, data=True))
            neighbors = defaultdict(list)
            for source, target, attrs in edges:
                neighbor = target if source == anchor else source
                neighbor_type = graph.nodes[neighbor].get("entity_type", str(neighbor).split("::", 1)[0])
                if neighbor_type not in SAFE_ENTITY_TYPES:
                    continue
                if self._semantic(attrs):
                    records = {_text(attrs["record_id"])}
                    factual = True
                elif (not _truth(attrs.get("fact_assertion"))
                      and attrs.get("relation") == "CO_MENTIONED_WITH"):
                    records = set(memberships.get(anchor, {})) & set(memberships.get(neighbor, {}))
                    factual = False
                else:
                    continue
                if records:
                    neighbors[neighbor].append((source, target, attrs, records, factual))
            ordered = sorted(neighbors, key=lambda neighbor: (
                -int(any(edge[4] for edge in neighbors[neighbor])),
                -max(relevance_of(edge[3]) for edge in neighbors[neighbor]), str(neighbor)))
            for neighbor in ordered[:max(0, int(neighbor_cap))]:
                selected_neighbors.add(neighbor)
                for source, target, attrs, records, factual in neighbors[neighbor]:
                    for record in limited_records(records):
                        original_id = self._rows[self._record_positions[record][0]]["record_id"]
                        if factual:
                            semantic_records.add(record)
                            evidence = {"kind": "reviewed_semantic", "source": source, "target": target,
                                        "relation": attrs["relation"], "fact_assertion": True,
                                        "retrieval_only": False, "record_id": original_id,
                                        "date": attrs["date"], "quote": attrs["evidence"],
                                        "source_fingerprint": attrs["source_fingerprint"],
                                        "review_method": attrs["review_method"],
                                        "reviewer": attrs.get("reviewer", ""),
                                        "char_start": attrs.get("char_start"), "char_end": attrs.get("char_end")}
                        else:
                            evidence = {"kind": "association", "source": source, "target": target,
                                        "relation": "CO_MENTIONED_WITH", "fact_assertion": False,
                                        "retrieval_only": True, "record_id": original_id,
                                        "date": memberships[anchor][record]["date"],
                                        "support": "both endpoints have NER membership in this article; no semantic relationship asserted"}
                        add(record, evidence)
        # No centrality/degree score and no unrelated articles for a neighbor:
        # every source is ranked against the original query. A modest preference
        # keeps rare reviewed sources ahead of equally relevant associations.
        ordered_records = sorted(candidate_records, key=lambda record: (
            -(relevance_of([record]) + (0.05 if record in semantic_records else 0)), record))[:max(0, int(article_cap))]
        positions = []
        for record in ordered_records:
            positions.extend(self._order(relevance, self._record_positions[record])[:2])
        self.last_stats.update(graph_anchors=list(anchors), graph_neighbors_selected=len(selected_neighbors),
                               graph_candidate_articles=len(ordered_records), graph_candidate_chunks=len(positions))
        return positions, evidence_by_record

    def _load_reranker(self):
        if self._reranker_identity is not None:
            return
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            location, revision = _model_location(self.reranker_model_id)
            self.reranker = CrossEncoder(location, device="cpu", local_files_only=True,
                                         trust_remote_code=False, max_length=512)
        else:
            revision = getattr(self.reranker, "revision", "injected-reranker")
        self._reranker_identity = {"model": self.reranker_model_id, "revision": str(revision),
                                   "max_length": int(getattr(self.reranker, "max_length", 512))}

    def reranker_identity(self):
        """Return the actual local reranker snapshot for outer experiment caches."""
        self._load_reranker()
        return dict(self._reranker_identity)

    def _rerank(self, question, indices):
        started = time.perf_counter()
        self._load_reranker()
        pending, scores, paths = [], {}, {}
        for index in indices:
            key = _digest({**self._reranker_identity, "question": question, "text": self._rows[index]["text"],
                           "source_fingerprint": self._rows[index]["source_fingerprint"]})
            path = self.cache_dir / "reranker" / f"{key}.json"
            if path.exists():
                scores[index] = float(json.loads(path.read_text())["score"])
            else:
                pending.append(index)
                paths[index] = path
        model_seconds = 0.0
        if pending:
            before = time.perf_counter()
            values = np.asarray(self.reranker.predict([(question, self._rows[i]["text"]) for i in pending],
                                                       batch_size=32, show_progress_bar=False)).ravel()
            model_seconds = time.perf_counter() - before
            if len(values) != len(pending):
                raise ValueError("Reranker returned an unexpected number of scores")
            for index, score in zip(pending, values):
                scores[index] = float(score)
                _json_write(paths[index], {"score": float(score), **self._reranker_identity})
        self.last_stats.update(reranker_candidate_count=len(indices), reranker_pairs_scored=len(pending),
                               reranker_pair_cache_hits=len(indices) - len(pending),
                               reranker_model_seconds=model_seconds,
                               reranker_seconds=time.perf_counter() - started)
        return sorted(indices, key=lambda index: (-scores[index], str(self._rows[index]["chunk_id"]))), scores

    def retrieve(self, question, mode="hybrid", top_k=4, candidate_k=20, graph=None,
                 entity_articles=None, graph_weight=0.35, rerank=False, *, aliases=None,
                 graph_neighbor_cap=12, graph_article_cap=100):
        started = time.perf_counter()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if int(top_k) < 0 or int(candidate_k) < 1 or float(graph_weight) < 0:
            raise ValueError("top_k must be nonnegative, candidate_k positive, and graph_weight nonnegative")
        if mode == "graph" and graph is None:
            raise ValueError("graph mode requires an audited graph")
        self.last_stats = {"index_cache_hit": self.index_cache_hit, "reranker_pairs_scored": 0,
                           "reranker_candidate_count": 0, "reranker_seconds": 0.0}
        lexical, dense = self._scores(question, need_dense=mode != "sparse" or graph is not None)
        base = self._base_scores(lexical, dense, mode)
        pool_size = min(len(self.chunks), int(candidate_k))
        valid = np.flatnonzero(lexical > 0) if mode == "sparse" else np.arange(len(base))
        indices = self._order(base, valid)[:pool_size]
        combined = base.copy()
        evidence_by_record = {}
        method = mode
        if graph is not None and graph_weight > 0:
            graph_indices, evidence_by_record = self._graph_candidates(
                question, graph, entity_articles, lexical, dense, aliases,
                graph_neighbor_cap, graph_article_cap)
            union = set(indices) | set(graph_indices)
            # Rank normalisation makes the graph weight comparable across sparse,
            # dense, and RRF baselines. Candidate reranking below sees one fixed
            # pool, independent of whether the cross encoder is enabled.
            combined = np.zeros(len(base), dtype=float)
            for rank, index in enumerate(self._order(base, union), start=1):
                combined[index] = 1 / (60 + rank)
            for rank, index in enumerate(graph_indices, start=1):
                combined[index] += float(graph_weight) / (60 + rank)
            indices = self._order(combined, union)[:pool_size]
            method = f"{mode}+entity_graph"
        self.last_stats["candidate_chunk_ids"] = [self._rows[i]["chunk_id"] for i in indices]
        self.last_stats["candidate_count"] = len(indices)
        reranked_scores = None
        if rerank and indices:
            indices, reranked_scores = self._rerank(question, indices)
            method += "+cross_encoder"
        hits = []
        for index in indices[:int(top_k)]:
            evidence = evidence_by_record.get(_text(self._rows[index]["record_id"]), [])
            evidence = [dict(item, quote_in_chunk=item.get("quote", "") in self._rows[index]["text"])
                        if item.get("kind") == "reviewed_semantic" else dict(item) for item in evidence]
            hit = self._hit(index, combined[index], method, lexical, dense, evidence)
            if reranked_scores is not None:
                hit["retrieval_score"] = hit["score"]
                hit["score"] = reranked_scores[index]
                hit["reranker_score"] = reranked_scores[index]
            hits.append(hit)
        self.last_stats["retrieval_seconds"] = time.perf_counter() - started
        return hits
