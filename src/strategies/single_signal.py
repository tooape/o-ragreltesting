"""
Single Signal Strategies (C1-C6)

Basic retrieval using individual signals without fusion.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    BaseStrategy,
    DenseRetrievalStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C1_DenseOnly(DenseRetrievalStrategy):
    """C1: Dense retrieval only (cosine similarity)."""

    STRATEGY_ID = "c1_dense_only"
    CATEGORY = "single_signal"
    DESCRIPTION = "Dense retrieval with cosine similarity, no fusion"

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
            raise ValueError("C1 requires query embeddings")

        scores = self.cosine_similarity(query_embedding, chunk_embeddings)
        ranked_ids, score_dict = self.rank_by_scores(scores, chunks, self.top_k)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C2_BM25Only(BaseStrategy):
    """C2: BM25 only (keyword matching)."""

    STRATEGY_ID = "c2_bm25_only"
    CATEGORY = "single_signal"
    DESCRIPTION = "BM25 keyword retrieval only"

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    @property
    def requires_embeddings(self) -> bool:
        return False

    @property
    def requires_bm25(self) -> bool:
        return True

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        if bm25_scores is None:
            raise ValueError("C2 requires BM25 scores")

        indices = np.argsort(bm25_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(bm25_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C3_Dense256(DenseRetrievalStrategy):
    """C3: Dense retrieval with 256-dim Matryoshka truncation."""

    STRATEGY_ID = "c3_dense_256"
    CATEGORY = "single_signal"
    DESCRIPTION = "Dense retrieval with 256-dim Matryoshka embeddings"

    def __init__(self, top_k: int = 20, truncate_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.truncate_dim = truncate_dim

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
            raise ValueError("C3 requires query embeddings")

        # Truncate to 256 dims
        q_truncated = query_embedding[:self.truncate_dim]
        c_truncated = chunk_embeddings[:, :self.truncate_dim]

        # Renormalize
        q_truncated = q_truncated / (np.linalg.norm(q_truncated) + 1e-8)
        c_norms = np.linalg.norm(c_truncated, axis=1, keepdims=True)
        c_truncated = c_truncated / (c_norms + 1e-8)

        scores = np.dot(c_truncated, q_truncated)
        ranked_ids, score_dict = self.rank_by_scores(scores, chunks, self.top_k)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "dim": self.truncate_dim},
        )


@register_strategy
class C4_Dense512(DenseRetrievalStrategy):
    """C4: Dense retrieval with 512-dim Matryoshka truncation."""

    STRATEGY_ID = "c4_dense_512"
    CATEGORY = "single_signal"
    DESCRIPTION = "Dense retrieval with 512-dim Matryoshka embeddings"

    def __init__(self, top_k: int = 20, truncate_dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.truncate_dim = truncate_dim

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
            raise ValueError("C4 requires query embeddings")

        # Truncate to 512 dims
        q_truncated = query_embedding[:self.truncate_dim]
        c_truncated = chunk_embeddings[:, :self.truncate_dim]

        # Renormalize
        q_truncated = q_truncated / (np.linalg.norm(q_truncated) + 1e-8)
        c_norms = np.linalg.norm(c_truncated, axis=1, keepdims=True)
        c_truncated = c_truncated / (c_norms + 1e-8)

        scores = np.dot(c_truncated, q_truncated)
        ranked_ids, score_dict = self.rank_by_scores(scores, chunks, self.top_k)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "dim": self.truncate_dim},
        )


@register_strategy
class C5_Dense768(DenseRetrievalStrategy):
    """C5: Dense retrieval with full 768-dim embeddings."""

    STRATEGY_ID = "c5_dense_768"
    CATEGORY = "single_signal"
    DESCRIPTION = "Dense retrieval with full 768-dim embeddings"

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
            raise ValueError("C5 requires query embeddings")

        scores = self.cosine_similarity(query_embedding, chunk_embeddings)
        ranked_ids, score_dict = self.rank_by_scores(scores, chunks, self.top_k)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C6_BM25Plus(BaseStrategy):
    """C6: BM25+ variant (improved handling of long documents)."""

    STRATEGY_ID = "c6_bm25_plus"
    CATEGORY = "single_signal"
    DESCRIPTION = "BM25+ variant for better long document handling"

    def __init__(self, top_k: int = 20, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.delta = delta

    @property
    def requires_embeddings(self) -> bool:
        return False

    @property
    def requires_bm25(self) -> bool:
        return True

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        # Note: BM25+ scores should be pre-computed by the BM25Searcher with variant="plus"
        # This strategy assumes the scores are already BM25+ scores
        if bm25_scores is None:
            raise ValueError("C6 requires BM25+ scores")

        indices = np.argsort(bm25_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(bm25_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "variant": "plus"},
        )
