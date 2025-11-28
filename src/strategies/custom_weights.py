"""
Custom Weights (C10-C14) - Data-Driven Weight Configurations

Different weight configurations for multi-signal fusion.
Based on benchmark insights that BM25 is surprisingly strong.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C10_BM25Heavy(FusionStrategy):
    """C10: BM25-Heavy Fusion (80% BM25, 20% Semantic)

    Rationale: Benchmarks show BM25 is strong for exact matches.
    """

    STRATEGY_ID = "c10_bm25_heavy"
    CATEGORY = "custom_weights"
    DESCRIPTION = "BM25-heavy fusion: 80% BM25, 20% Semantic"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight

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
            raise ValueError("C10 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C10 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize
        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        # Weighted fusion
        fused = self.bm25_weight * bm25_norm + self.semantic_weight * dense_norm

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
            metadata={"strategy": self.STRATEGY_ID, "weights": [self.bm25_weight, self.semantic_weight]},
        )


@register_strategy
class C11_SemanticHeavy(FusionStrategy):
    """C11: Semantic-Heavy Fusion (30% BM25, 70% Semantic)

    Test: Baseline comparison for semantic-first approach.
    """

    STRATEGY_ID = "c11_semantic_heavy"
    CATEGORY = "custom_weights"
    DESCRIPTION = "Semantic-heavy fusion: 30% BM25, 70% Semantic"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight

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
            raise ValueError("C11 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C11 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        fused = self.bm25_weight * bm25_norm + self.semantic_weight * dense_norm

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
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C12_EqualWeight(FusionStrategy):
    """C12: Equal Weight Fusion (50% BM25, 50% Semantic)

    Test: Pure score-based fusion, no bias.
    """

    STRATEGY_ID = "c12_equal_weight"
    CATEGORY = "custom_weights"
    DESCRIPTION = "Equal weight fusion: 50% BM25, 50% Semantic"

    def __init__(
        self,
        top_k: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k

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
            raise ValueError("C12 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C12 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        fused = 0.5 * bm25_norm + 0.5 * dense_norm

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
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C13_MultiSignalOptimized(FusionStrategy):
    """C13: Multi-Signal Optimized Weights

    Balanced multi-signal: BM25 (45%) + Semantic (35%) + Graph (10%) + Temporal (10%)
    """

    STRATEGY_ID = "c13_multi_signal_optimized"
    CATEGORY = "custom_weights"
    DESCRIPTION = "Multi-signal: 45% BM25, 35% Semantic, 10% Graph, 10% Temporal"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.45,
        semantic_weight: float = 0.35,
        graph_weight: float = 0.10,
        temporal_weight: float = 0.10,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.temporal_weight = temporal_weight
        self.temporal_decay_days = temporal_decay_days

    def get_graph_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            scores[i] = pagerank if pagerank > 0 else np.log1p(link_count) / 10.0
        return scores

    def get_temporal_scores(self, chunks: List[Dict]) -> np.ndarray:
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
                scores[i] = 0.0
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
            raise ValueError("C13 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C13 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        graph_scores = self.get_graph_scores(chunks)
        temporal_scores = self.get_temporal_scores(chunks)

        # Normalize all signals
        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        graph_norm = self.normalize_scores(graph_scores)
        temporal_norm = self.normalize_scores(temporal_scores)

        # Weighted combination
        fused = (
            self.bm25_weight * bm25_norm +
            self.semantic_weight * dense_norm +
            self.graph_weight * graph_norm +
            self.temporal_weight * temporal_norm
        )

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
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C14_MultiSignalBM25Dominant(FusionStrategy):
    """C14: Multi-Signal BM25-Dominant

    Let BM25 lead, others support: BM25 (60%) + Semantic (25%) + Graph (10%) + Temporal (5%)
    """

    STRATEGY_ID = "c14_multi_signal_bm25_dominant"
    CATEGORY = "custom_weights"
    DESCRIPTION = "BM25-dominant multi-signal: 60% BM25, 25% Semantic, 10% Graph, 5% Temporal"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.60,
        semantic_weight: float = 0.25,
        graph_weight: float = 0.10,
        temporal_weight: float = 0.05,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight
        self.temporal_weight = temporal_weight
        self.temporal_decay_days = temporal_decay_days

    def get_graph_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            scores[i] = pagerank if pagerank > 0 else np.log1p(link_count) / 10.0
        return scores

    def get_temporal_scores(self, chunks: List[Dict]) -> np.ndarray:
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
                scores[i] = 0.0
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
            raise ValueError("C14 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C14 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        graph_scores = self.get_graph_scores(chunks)
        temporal_scores = self.get_temporal_scores(chunks)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        graph_norm = self.normalize_scores(graph_scores)
        temporal_norm = self.normalize_scores(temporal_scores)

        fused = (
            self.bm25_weight * bm25_norm +
            self.semantic_weight * dense_norm +
            self.graph_weight * graph_norm +
            self.temporal_weight * temporal_norm
        )

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
            metadata={"strategy": self.STRATEGY_ID},
        )
