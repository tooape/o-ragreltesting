"""
Two-Stage Retrieval Strategies (C25-C30)

Strategies that use a fast first stage followed by more accurate reranking.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import (
    TwoStageStrategy,
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C25_BM25ThenDense(TwoStageStrategy):
    """C25: BM25 first stage, dense reranking."""

    STRATEGY_ID = "c25_bm25_then_dense"
    CATEGORY = "two_stage"
    DESCRIPTION = "BM25 retrieval -> dense reranking"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 100,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if bm25_scores is None:
            raise ValueError("C25 requires BM25 scores")

        indices = np.argsort(bm25_scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        if query_embedding is None:
            raise ValueError("C25 requires query embeddings for reranking")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Score only candidates
        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(c_norms, q_norm)

        # Rank candidates
        ranked_order = np.argsort(scores)[::-1]
        reranked_indices = [candidate_indices[i] for i in ranked_order]
        reranked_scores = scores[ranked_order]

        return reranked_indices, reranked_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        # First stage
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Second stage
        reranked_indices, scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Get top_k
        final_indices = reranked_indices[:self.top_k]
        final_scores = scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "first_stage_k": self.first_stage_k},
        )


@register_strategy
class C26_DenseThenBM25(TwoStageStrategy):
    """C26: Dense first stage, BM25 reranking."""

    STRATEGY_ID = "c26_dense_then_bm25"
    CATEGORY = "two_stage"
    DESCRIPTION = "Dense retrieval -> BM25 reranking"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 100,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if query_embedding is None:
            raise ValueError("C26 requires query embeddings")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(c_norms, q_norm)

        indices = np.argsort(scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        if bm25_scores is None:
            raise ValueError("C26 requires BM25 scores for reranking")

        # Get BM25 scores for candidates
        candidate_scores = bm25_scores[candidate_indices]

        ranked_order = np.argsort(candidate_scores)[::-1]
        reranked_indices = [candidate_indices[i] for i in ranked_order]
        reranked_scores = candidate_scores[ranked_order]

        return reranked_indices, reranked_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )
        reranked_indices, scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices = reranked_indices[:self.top_k]
        final_scores = scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C27_HybridThenDense(TwoStageStrategy):
    """C27: Hybrid first stage, dense reranking."""

    STRATEGY_ID = "c27_hybrid_then_dense"
    CATEGORY = "two_stage"
    DESCRIPTION = "Hybrid retrieval -> dense reranking on top candidates"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 100,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if query_embedding is None or bm25_scores is None:
            raise ValueError("C27 requires both embeddings and BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # RRF fusion
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        n_docs = len(chunks)
        rrf_scores = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            rrf_scores[doc_idx] = (
                1.0 / (self.rrf_k + dense_ranks[doc_idx] + 1)
                + 1.0 / (self.rrf_k + bm25_ranks[doc_idx] + 1)
            )

        indices = np.argsort(rrf_scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(c_norms, q_norm)

        ranked_order = np.argsort(scores)[::-1]
        reranked_indices = [candidate_indices[i] for i in ranked_order]
        reranked_scores = scores[ranked_order]

        return reranked_indices, reranked_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )
        reranked_indices, scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices = reranked_indices[:self.top_k]
        final_scores = scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C28_BM25ThenHybrid(TwoStageStrategy):
    """C28: BM25 first stage, hybrid reranking."""

    STRATEGY_ID = "c28_bm25_then_hybrid"
    CATEGORY = "two_stage"
    DESCRIPTION = "BM25 retrieval -> hybrid reranking on candidates"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 100,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k
        self.alpha = alpha

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if bm25_scores is None:
            raise ValueError("C28 requires BM25 scores")

        indices = np.argsort(bm25_scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        candidate_bm25 = bm25_scores[candidate_indices]

        # Normalize within candidates
        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(candidate_bm25)

        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

        ranked_order = np.argsort(hybrid_scores)[::-1]
        reranked_indices = [candidate_indices[i] for i in ranked_order]
        reranked_scores = hybrid_scores[ranked_order]

        return reranked_indices, reranked_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )
        reranked_indices, scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices = reranked_indices[:self.top_k]
        final_scores = scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID},
        )


@register_strategy
class C29_ThreeStage(TwoStageStrategy):
    """C29: Three-stage retrieval (BM25 -> dense -> rerank top)."""

    STRATEGY_ID = "c29_three_stage"
    CATEGORY = "two_stage"
    DESCRIPTION = "Three-stage: BM25(200) -> dense(50) -> final(20)"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 200,
        second_stage_k: int = 50,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k
        self.second_stage_k = second_stage_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if bm25_scores is None:
            raise ValueError("C29 requires BM25 scores")
        indices = np.argsort(bm25_scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        # Dense reranking
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(c_norms, q_norm)

        ranked_order = np.argsort(scores)[::-1][:self.second_stage_k]
        stage2_indices = [candidate_indices[i] for i in ranked_order]
        stage2_scores = scores[ranked_order]

        return stage2_indices, stage2_scores

    def third_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        """Third stage: hybrid reranking on refined candidates."""
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        candidate_bm25 = bm25_scores[candidate_indices]

        # Normalize
        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        hybrid_scores = 0.6 * normalize(dense_scores) + 0.4 * normalize(candidate_bm25)

        ranked_order = np.argsort(hybrid_scores)[::-1]
        final_indices = [candidate_indices[i] for i in ranked_order]
        final_scores = hybrid_scores[ranked_order]

        return final_indices, final_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        # Stage 1: BM25
        stage1_candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Stage 2: Dense
        stage2_indices, _ = self.second_stage_rerank(
            query, stage1_candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Stage 3: Hybrid
        final_indices, final_scores = self.third_stage_rerank(
            query, stage2_indices, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices = final_indices[:self.top_k]
        final_scores = final_scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "stages": 3},
        )


@register_strategy
class C30_LargeFirstStage(TwoStageStrategy):
    """C30: Very large first stage with aggressive reranking."""

    STRATEGY_ID = "c30_large_first_stage"
    CATEGORY = "two_stage"
    DESCRIPTION = "Large first stage (300) -> aggressive reranking"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 300,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k
        self.rrf_k = rrf_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if query_embedding is None or bm25_scores is None:
            raise ValueError("C30 requires both embeddings and BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # RRF
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        n_docs = len(chunks)
        rrf_scores = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            rrf_scores[doc_idx] = (
                1.0 / (self.rrf_k + dense_ranks[doc_idx] + 1)
                + 1.0 / (self.rrf_k + bm25_ranks[doc_idx] + 1)
            )

        indices = np.argsort(rrf_scores)[::-1][:self.first_stage_k]
        return indices.tolist()

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        # Dense-heavy reranking
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_embeddings = chunk_embeddings[candidate_indices]
        c_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        candidate_bm25 = bm25_scores[candidate_indices]

        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        # Dense-heavy for reranking
        hybrid_scores = 0.7 * normalize(dense_scores) + 0.3 * normalize(candidate_bm25)

        ranked_order = np.argsort(hybrid_scores)[::-1]
        reranked_indices = [candidate_indices[i] for i in ranked_order]
        reranked_scores = hybrid_scores[ranked_order]

        return reranked_indices, reranked_scores

    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )
        reranked_indices, scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices = reranked_indices[:self.top_k]
        final_scores = scores[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx, score in zip(final_indices, final_scores):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(score)

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "first_stage_k": self.first_stage_k},
        )
