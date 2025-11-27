"""
Weighted Hybrid Strategies (C13-C18)

Hybrid retrieval with tunable weights and parameterized fusion.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C13_DenseHeavy(FusionStrategy):
    """C13: Dense-heavy fusion (0.7 dense + 0.3 BM25)."""

    STRATEGY_ID = "c13_dense_heavy"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "Dense-heavy: 0.7*dense + 0.3*BM25"

    def __init__(self, top_k: int = 20, dense_weight: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.dense_weight = dense_weight

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

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        fused = self.dense_weight * dense_norm + (1 - self.dense_weight) * bm25_norm

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
            metadata={"strategy": self.STRATEGY_ID, "dense_weight": self.dense_weight},
        )


@register_strategy
class C14_BM25Heavy(FusionStrategy):
    """C14: BM25-heavy fusion (0.3 dense + 0.7 BM25)."""

    STRATEGY_ID = "c14_bm25_heavy"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "BM25-heavy: 0.3*dense + 0.7*BM25"

    def __init__(self, top_k: int = 20, dense_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.dense_weight = dense_weight

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

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        fused = self.dense_weight * dense_norm + (1 - self.dense_weight) * bm25_norm

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
            metadata={"strategy": self.STRATEGY_ID, "dense_weight": self.dense_weight},
        )


@register_strategy
class C15_WeightedRRF(FusionStrategy):
    """C15: Weighted RRF (different weights for each ranking)."""

    STRATEGY_ID = "c15_weighted_rrf"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "Weighted RRF with configurable per-system weights"

    def __init__(
        self,
        top_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def weighted_rrf(
        self,
        rankings: List[np.ndarray],
        weights: List[float],
        k: int = 60,
    ) -> np.ndarray:
        """Weighted RRF."""
        n_docs = rankings[0].shape[0]
        fused = np.zeros(n_docs)

        for ranks, weight in zip(rankings, weights):
            for doc_idx in range(n_docs):
                fused[doc_idx] += weight / (k + ranks[doc_idx] + 1)

        return fused

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
            raise ValueError("C15 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C15 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        fused = self.weighted_rrf(
            [dense_ranks, bm25_ranks],
            [self.dense_weight, self.bm25_weight],
            k=self.rrf_k,
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
            metadata={
                "strategy": self.STRATEGY_ID,
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight,
            },
        )


@register_strategy
class C16_ZScoreNorm(FusionStrategy):
    """C16: Fusion with z-score normalization."""

    STRATEGY_ID = "c16_zscore_norm"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "Fusion with z-score normalization before combining"

    def __init__(self, top_k: int = 20, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha

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
            raise ValueError("C16 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C16 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Z-score normalization
        dense_norm = self.normalize_scores(dense_scores, method="zscore")
        bm25_norm = self.normalize_scores(bm25_scores, method="zscore")

        # Combine
        fused = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

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
            metadata={"strategy": self.STRATEGY_ID, "normalization": "zscore"},
        )


@register_strategy
class C17_RankNorm(FusionStrategy):
    """C17: Fusion with rank-based normalization."""

    STRATEGY_ID = "c17_rank_norm"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "Fusion with rank-based normalization"

    def __init__(self, top_k: int = 20, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha

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
            raise ValueError("C17 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C17 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Rank-based normalization
        dense_norm = self.normalize_scores(dense_scores, method="rank")
        bm25_norm = self.normalize_scores(bm25_scores, method="rank")

        # Combine
        fused = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

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
            metadata={"strategy": self.STRATEGY_ID, "normalization": "rank"},
        )


@register_strategy
class C18_MultiRRF(FusionStrategy):
    """C18: Multi-signal RRF (dense + BM25 + title match)."""

    STRATEGY_ID = "c18_multi_rrf"
    CATEGORY = "weighted_hybrid"
    DESCRIPTION = "Multi-signal RRF: dense + BM25 + title matching"

    def __init__(self, top_k: int = 20, rrf_k: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k

    def compute_title_scores(self, query: str, chunks: List[Dict]) -> np.ndarray:
        """Simple title matching score."""
        query_terms = set(query.lower().split())
        scores = np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            title = chunk.get("title", "").lower()
            title_terms = set(title.split())
            overlap = len(query_terms & title_terms)
            scores[i] = overlap / max(len(query_terms), 1)

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
            raise ValueError("C18 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C18 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Title scores
        title_scores = self.compute_title_scores(query, chunks)

        # Convert all to ranks
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))
        title_ranks = np.argsort(np.argsort(-title_scores))

        # Multi-signal RRF
        fused = self.reciprocal_rank_fusion(
            [dense_ranks, bm25_ranks, title_ranks],
            k=self.rrf_k,
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
            metadata={"strategy": self.STRATEGY_ID, "signals": 3},
        )
