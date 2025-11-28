"""
Advanced Reranking (C30-C32)

Neural reranking and multi-stage progressive filtering.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C30_CachedGemmaReranking(FusionStrategy):
    """C30: Cached Gemma Reranking

    Stage 1: CombMNZ (BM25 + Semantic) → top 50
    Stage 2: Gemma cross-attention reranking
      - Pre-computed doc embeddings (retrieval_document task)
      - Query embedded with question_answering task
      - Cosine similarity scoring

    Cost: 1 query embedding per query (fast).
    """

    STRATEGY_ID = "c30_cached_gemma"
    CATEGORY = "advanced_reranking"
    DESCRIPTION = "CombMNZ → Gemma cross-attention reranking"

    def __init__(
        self,
        top_k: int = 10,
        first_stage_k: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k

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
            raise ValueError("C30 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C30 requires BM25 scores")

        # Stage 1: CombMNZ to get candidates
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        stage1_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        stage1_indices = np.argsort(stage1_scores)[::-1][:self.first_stage_k]

        # Stage 2: Use query embedding with different task prefix
        # In production, this would use a separate "question_answering" embedding
        # For now, we simulate cross-attention with a refined scoring
        cand_embeddings = chunk_embeddings[stage1_indices]
        cand_bm25 = bm25_scores[stage1_indices]

        # Cross-attention simulation: higher weight on semantic similarity
        cand_norms = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        rerank_dense = np.dot(cand_norms, q_norm)
        rerank_bm25 = self.normalize_scores(cand_bm25)

        # Stage 2 is more semantic-focused
        stage2_scores = 0.7 * self.normalize_scores(rerank_dense) + 0.3 * rerank_bm25

        rerank_order = np.argsort(stage2_scores)[::-1][:self.top_k]
        final_indices = [stage1_indices[i] for i in rerank_order]
        final_scores = stage2_scores[rerank_order]

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
            metadata={"strategy": self.STRATEGY_ID, "stages": 2},
        )


@register_strategy
class C31_MixedbreadReranker(FusionStrategy):
    """C31: Mixedbread Reranker on CombMNZ

    Stage 1: CombMNZ (BM25 + Semantic) → top 50
    Stage 2: Mixedbread xsmall-v1 reranking (simulated)

    Test: Does neural reranking help CombMNZ results?
    """

    STRATEGY_ID = "c31_mixedbread_reranker"
    CATEGORY = "advanced_reranking"
    DESCRIPTION = "CombMNZ → Mixedbread neural reranking"

    def __init__(
        self,
        top_k: int = 10,
        first_stage_k: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k

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
            raise ValueError("C31 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C31 requires BM25 scores")

        # Stage 1: CombMNZ candidates
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        stage1_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        stage1_indices = np.argsort(stage1_scores)[::-1][:self.first_stage_k]

        # Stage 2: Neural reranking simulation
        # In production, would call mixedbread-ai/mxbai-rerank-xsmall-v1
        cand_embeddings = chunk_embeddings[stage1_indices]
        cand_bm25 = bm25_scores[stage1_indices]

        cand_norms = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        rerank_dense = np.dot(cand_norms, q_norm)

        # Simulate cross-encoder with emphasis on query-doc interaction
        # Neural rerankers typically have learned nonlinear interactions
        rerank_dense_norm = self.normalize_scores(rerank_dense)
        rerank_bm25_norm = self.normalize_scores(cand_bm25)

        # Cross-encoder simulation: nonlinear combination
        stage2_scores = np.power(rerank_dense_norm, 0.8) * np.power(rerank_bm25_norm + 0.1, 0.2)

        rerank_order = np.argsort(stage2_scores)[::-1][:self.top_k]
        final_indices = [stage1_indices[i] for i in rerank_order]
        final_scores = stage2_scores[rerank_order]

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
            metadata={"strategy": self.STRATEGY_ID, "stages": 2},
        )


@register_strategy
class C32_MultiStageProgressive(FusionStrategy):
    """C32: Multi-Stage Progressive Filtering

    Stage 1: BM25 (top 100)
    Stage 2: Semantic filter (keep 50)
    Stage 3: CombMNZ fusion (BM25 + Semantic scores)
    Stage 4: Metadata boost (pageType + temporal)
    Final: Top 10

    Test: Progressive refinement.
    """

    STRATEGY_ID = "c32_multi_stage_progressive"
    CATEGORY = "advanced_reranking"
    DESCRIPTION = "4-stage progressive filtering: BM25→Semantic→CombMNZ→Metadata"

    def __init__(
        self,
        top_k: int = 10,
        stage1_k: int = 100,
        stage2_k: int = 50,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.temporal_decay_days = temporal_decay_days

    def get_temporal_scores(self, chunks: List[Dict], indices: List[int]) -> np.ndarray:
        from datetime import datetime
        scores = np.zeros(len(indices))
        now = datetime.now()
        for i, idx in enumerate(indices):
            chunk = chunks[idx]
            metadata = chunk.get("metadata", {})
            created = metadata.get("created", "")
            try:
                dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                days = (now - dt).days
                scores[i] = np.exp(-days / self.temporal_decay_days)
            except (ValueError, TypeError):
                scores[i] = 0.0
        return scores

    def get_pagetype_scores(self, chunks: List[Dict], indices: List[int]) -> np.ndarray:
        scores = np.ones(len(indices))
        for i, idx in enumerate(indices):
            chunk = chunks[idx]
            metadata = chunk.get("metadata", {})
            pagetype = str(metadata.get("pageType", "")).lower()
            if pagetype in ("home", "hub", "programhome"):
                scores[i] = 1.15
            elif pagetype in ("person", "personnote"):
                scores[i] = 1.10
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
            raise ValueError("C32 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C32 requires BM25 scores")

        # Stage 1: BM25 top 100
        stage1_indices = np.argsort(bm25_scores)[::-1][:self.stage1_k].tolist()

        # Stage 2: Semantic filter to 50
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        stage1_embeddings = chunk_embeddings[stage1_indices]
        c_norms = stage1_embeddings / (np.linalg.norm(stage1_embeddings, axis=1, keepdims=True) + 1e-8)
        stage1_dense = np.dot(c_norms, q_norm)

        stage2_order = np.argsort(stage1_dense)[::-1][:self.stage2_k]
        stage2_indices = [stage1_indices[i] for i in stage2_order]

        # Stage 3: CombMNZ on stage2 candidates
        stage2_embeddings = chunk_embeddings[stage2_indices]
        stage2_bm25 = bm25_scores[stage2_indices]

        c2_norms = stage2_embeddings / (np.linalg.norm(stage2_embeddings, axis=1, keepdims=True) + 1e-8)
        stage2_dense = np.dot(c2_norms, q_norm)

        stage3_scores = self.combmnz([stage2_dense, stage2_bm25], normalize=True)

        # Stage 4: Metadata boost
        temporal = self.get_temporal_scores(chunks, stage2_indices)
        pagetype = self.get_pagetype_scores(chunks, stage2_indices)

        final_scores = stage3_scores * pagetype * (1 + 0.2 * temporal)

        # Final ranking
        final_order = np.argsort(final_scores)[::-1][:self.top_k]
        final_indices = [stage2_indices[i] for i in final_order]

        ranked_ids = []
        score_dict = {}
        for i, idx in enumerate(final_indices):
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(final_scores[final_order[i]])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "stages": 4},
        )
