"""
Simple Fusion Strategies (C7-C12)

Basic hybrid retrieval combining dense and sparse signals.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C7_RRFBasic(FusionStrategy):
    """C7: Reciprocal Rank Fusion (basic, k=60)."""

    STRATEGY_ID = "c7_rrf_basic"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "Reciprocal Rank Fusion with k=60"

    def __init__(self, top_k: int = 20, rrf_k: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k

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

        # Get dense scores
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Convert to ranks
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        # RRF
        rrf_scores = self.reciprocal_rank_fusion([dense_ranks, bm25_ranks], k=self.rrf_k)

        # Rank by RRF scores
        indices = np.argsort(rrf_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(rrf_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "rrf_k": self.rrf_k},
        )


@register_strategy
class C8_RRFLowK(FusionStrategy):
    """C8: RRF with lower k (k=20) for more rank sensitivity."""

    STRATEGY_ID = "c8_rrf_low_k"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "Reciprocal Rank Fusion with k=20 (more rank sensitive)"

    def __init__(self, top_k: int = 20, rrf_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k

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

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        rrf_scores = self.reciprocal_rank_fusion([dense_ranks, bm25_ranks], k=self.rrf_k)

        indices = np.argsort(rrf_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(rrf_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "rrf_k": self.rrf_k},
        )


@register_strategy
class C9_CombMNZ(FusionStrategy):
    """C9: CombMNZ fusion (sum * count)."""

    STRATEGY_ID = "c9_combmnz"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "CombMNZ: score = sum(scores) * systems_retrieving"

    def __init__(self, top_k: int = 20, **kwargs):
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
            raise ValueError("C9 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C9 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombMNZ with normalized scores
        fused = self.combmnz([dense_scores, bm25_scores], normalize=True)

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
class C10_CombSUM(FusionStrategy):
    """C10: CombSUM fusion (simple sum)."""

    STRATEGY_ID = "c10_combsum"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "CombSUM: simple sum of normalized scores"

    def __init__(self, top_k: int = 20, **kwargs):
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
            raise ValueError("C10 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C10 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombSUM with normalized scores (equal weights)
        fused = self.combsum([dense_scores, bm25_scores], normalize=True)

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
class C11_ScoreInterpolation(FusionStrategy):
    """C11: Linear score interpolation (0.5 dense + 0.5 BM25)."""

    STRATEGY_ID = "c11_interpolation"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "Linear interpolation: 0.5*dense + 0.5*BM25"

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
            raise ValueError("C11 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C11 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize both
        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        # Interpolate
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
            metadata={"strategy": self.STRATEGY_ID, "alpha": self.alpha},
        )


@register_strategy
class C12_MaxFusion(FusionStrategy):
    """C12: Max fusion (take max score from either system)."""

    STRATEGY_ID = "c12_max_fusion"
    CATEGORY = "simple_fusion"
    DESCRIPTION = "Max fusion: score = max(dense, BM25)"

    def __init__(self, top_k: int = 20, **kwargs):
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

        # Normalize both
        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        # Max fusion
        fused = np.maximum(dense_norm, bm25_norm)

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
