"""
Score Normalization Variants (C6-C9)

Different normalization methods for hybrid fusion.
All use BM25 + Semantic with different score transformations.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C6_SoftmaxWeighted(FusionStrategy):
    """C6: Softmax Normalization + Weighted Hybrid

    Test: Does softmax preserve relative differences better?
    """

    STRATEGY_ID = "c6_softmax_weighted"
    CATEGORY = "normalization"
    DESCRIPTION = "Softmax normalization with weighted fusion (70% BM25)"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.7,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight
        self.temperature = temperature

    def softmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax normalization with temperature."""
        # Shift for numerical stability
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / self.temperature)
        return exp_scores / (np.sum(exp_scores) + 1e-8)

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
            raise ValueError("C6 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C6 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Softmax normalization
        dense_norm = self.softmax_normalize(dense_scores)
        bm25_norm = self.softmax_normalize(bm25_scores)

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
            metadata={"strategy": self.STRATEGY_ID, "temperature": self.temperature},
        )


@register_strategy
class C7_RankWeighted(FusionStrategy):
    """C7: Rank Normalization + Weighted Hybrid

    Pure rank-based like RRF, but with configurable weights.
    Formula: score = 1 - rank/total_docs
    """

    STRATEGY_ID = "c7_rank_weighted"
    CATEGORY = "normalization"
    DESCRIPTION = "Rank-based normalization with weighted fusion (70% BM25)"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight

    def rank_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to rank-based normalization (0-1)."""
        ranks = np.argsort(np.argsort(-scores))  # Higher score = lower rank number
        n = len(scores)
        return 1.0 - (ranks / (n - 1)) if n > 1 else np.ones_like(scores)

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
            raise ValueError("C7 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C7 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Rank normalization
        dense_norm = self.rank_normalize(dense_scores)
        bm25_norm = self.rank_normalize(bm25_scores)

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
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C8_PercentileWeighted(FusionStrategy):
    """C8: Percentile Normalization + Weighted Hybrid

    Robust to outliers - uses percentile rank (0-100).
    """

    STRATEGY_ID = "c8_percentile_weighted"
    CATEGORY = "normalization"
    DESCRIPTION = "Percentile normalization with weighted fusion (70% BM25)"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight

    def percentile_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to percentile ranks (0-1)."""
        return stats.rankdata(scores, method='average') / len(scores)

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
            raise ValueError("C8 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C8 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Percentile normalization
        dense_norm = self.percentile_normalize(dense_scores)
        bm25_norm = self.percentile_normalize(bm25_scores)

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
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C9_LogScaleWeighted(FusionStrategy):
    """C9: Log-scale Normalization + Weighted Hybrid

    Compress large score ranges using log transform.
    Formula: log(1 + score) then normalize
    """

    STRATEGY_ID = "c9_logscale_weighted"
    CATEGORY = "normalization"
    DESCRIPTION = "Log-scale normalization with weighted fusion (70% BM25)"

    def __init__(
        self,
        top_k: int = 20,
        bm25_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = 1.0 - bm25_weight

    def logscale_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Log-transform then min-max normalize."""
        # Shift to positive if needed
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score

        # Log transform
        log_scores = np.log1p(scores)

        # Min-max normalize
        min_s, max_s = log_scores.min(), log_scores.max()
        if max_s - min_s > 0:
            return (log_scores - min_s) / (max_s - min_s)
        return np.zeros_like(log_scores)

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
            raise ValueError("C9 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C9 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Log-scale normalization
        dense_norm = self.logscale_normalize(dense_scores)
        bm25_norm = self.logscale_normalize(bm25_scores)

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
            metadata={"strategy": self.STRATEGY_ID},
        )
