"""
Query-Adaptive Strategies (C28-C29)

Strategies that adapt based on query characteristics.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    QueryAdaptiveStrategy,
    FusionStrategy,
    BaseStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C28_QueryTypeRouter(QueryAdaptiveStrategy, FusionStrategy):
    """C28: Query Type Router

    Routes to different strategies based on query type:
    - Person query (has names): BM25-heavy (0.8, 0.2)
    - Temporal query (recent/latest): Temporal-boosted CombMNZ
    - Graph query (related/connected): Graph-heavy (0.3, 0.3, 0.4, 0.0)
    - Conceptual query: Semantic-heavy (0.3, 0.7)
    - Default: Balanced CombMNZ
    """

    STRATEGY_ID = "c28_query_type_router"
    CATEGORY = "query_adaptive"
    DESCRIPTION = "Routes queries to optimal strategy by type"

    # Keywords for classification
    PERSON_PATTERNS = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # "First Last"
        r'\b1x1\b', r'\b1:1\b',
        r'\bwho\b', r'\bperson\b',
    ]
    TEMPORAL_KEYWORDS = {"recent", "latest", "today", "yesterday", "last", "new", "this week", "status"}
    GRAPH_KEYWORDS = {"related", "connected", "linked", "similar", "hub", "references"}

    def __init__(
        self,
        top_k: int = 20,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        FusionStrategy.__init__(self, **kwargs)
        self.top_k = top_k
        self.temporal_decay_days = temporal_decay_days

    def classify_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Classify query type."""
        query_lower = query.lower()

        # Check for person queries
        for pattern in self.PERSON_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return "person"

        # Check for temporal queries
        if any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS):
            return "temporal"

        # Check for graph queries
        if any(kw in query_lower for kw in self.GRAPH_KEYWORDS):
            return "graph"

        # Default to conceptual (semantic-heavy)
        return "conceptual"

    def get_strategy_for_query_type(self, query_type: str) -> BaseStrategy:
        """Get strategy for query type (not used directly, logic embedded in rank)."""
        return self  # We handle routing internally

    def get_temporal_scores(self, chunks: List[Dict]) -> np.ndarray:
        from datetime import datetime
        scores = np.zeros(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created", "")
            try:
                dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                days = (now - dt).days
                scores[i] = np.exp(-days / self.temporal_decay_days)
            except (ValueError, TypeError):
                pass
        return scores

    def get_graph_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            scores[i] = pagerank if pagerank > 0 else np.log1p(link_count) / 10.0
        return scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        if query_embedding is None:
            raise ValueError("C28 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C28 requires BM25 scores")

        query_type = self.classify_query(query, query_embedding)

        # Compute dense scores
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize
        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        if query_type == "person":
            # BM25-heavy for exact name matching
            fused = 0.8 * bm25_norm + 0.2 * dense_norm

        elif query_type == "temporal":
            # Temporal-boosted CombMNZ
            temporal = self.get_temporal_scores(chunks)
            temporal_norm = self.normalize_scores(temporal)
            base = self.combmnz([dense_scores, bm25_scores], normalize=True)
            fused = 0.7 * base + 0.3 * temporal_norm

        elif query_type == "graph":
            # Graph-heavy
            graph = self.get_graph_scores(chunks)
            graph_norm = self.normalize_scores(graph)
            fused = 0.3 * bm25_norm + 0.3 * dense_norm + 0.4 * graph_norm

        else:  # conceptual
            # Semantic-heavy
            fused = 0.3 * bm25_norm + 0.7 * dense_norm

        indices = np.argsort(fused)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(fused[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "query_type": query_type},
        )


@register_strategy
class C29_AcronymAware(FusionStrategy):
    """C29: Acronym-Aware Routing

    If query has acronyms (CKG, PsW, Lr, etc.):
        Use BM25-heavy fusion (0.85, 0.15)
    Else:
        Use Semantic-heavy (0.4, 0.6)

    Test: Acronyms need exact matching.
    """

    STRATEGY_ID = "c29_acronym_aware"
    CATEGORY = "query_adaptive"
    DESCRIPTION = "Routes based on acronym detection (BM25 for acronyms)"

    # Common acronyms in the vault
    KNOWN_ACRONYMS = {
        "ckg", "psw", "lr", "ai", "ml", "srl", "ner", "qa", "rag",
        "sdl", "ps", "ae", "cc", "dxf", "api", "ui", "ux", "pm",
        "prm", "okr", "kpi", "sdc", "kb", "llm", "gpu", "cpu",
    }

    # Pattern for potential acronyms (2-5 uppercase letters)
    ACRONYM_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight_acronym: float = 0.85,
        bm25_weight_normal: float = 0.40,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight_acronym = bm25_weight_acronym
        self.bm25_weight_normal = bm25_weight_normal

    def has_acronyms(self, query: str) -> bool:
        """Check if query contains acronyms."""
        # Check known acronyms (case-insensitive)
        query_lower = query.lower()
        for acronym in self.KNOWN_ACRONYMS:
            if acronym in query_lower.split():
                return True

        # Check for uppercase patterns
        if self.ACRONYM_PATTERN.search(query):
            return True

        return False

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        if query_embedding is None:
            raise ValueError("C29 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C29 requires BM25 scores")

        # Detect acronyms
        has_acro = self.has_acronyms(query)
        bm25_weight = self.bm25_weight_acronym if has_acro else self.bm25_weight_normal
        semantic_weight = 1.0 - bm25_weight

        # Compute scores
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        fused = bm25_weight * bm25_norm + semantic_weight * dense_norm

        indices = np.argsort(fused)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(fused[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={
                "strategy": self.STRATEGY_ID,
                "has_acronyms": has_acro,
                "bm25_weight": bm25_weight,
            },
        )
