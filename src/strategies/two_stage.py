"""
Two-Stage Approaches (C24-C27)

Multi-stage retrieval strategies for balancing speed and accuracy.
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
class C24_BM25ThenCombMNZ(TwoStageStrategy, FusionStrategy):
    """C24: Tier1 BM25 → Tier2 CombMNZ

    Stage 1: BM25 only (top 50)
    Stage 2: CombMNZ(BM25 + Semantic) on Stage 1 results
    Final: Top 10

    Rationale: Fast BM25 candidate generation, then fusion.
    """

    STRATEGY_ID = "c24_bm25_then_combmnz"
    CATEGORY = "two_stage"
    DESCRIPTION = "BM25 first stage → CombMNZ reranking"

    def __init__(
        self,
        first_stage_k: int = 50,
        top_k: int = 10,
        **kwargs,
    ):
        # Initialize both parent classes properly
        TwoStageStrategy.__init__(self, first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        """First stage: BM25 top-k."""
        if bm25_scores is None:
            raise ValueError("C24 requires BM25 scores")

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
        """Second stage: CombMNZ on candidates."""
        if query_embedding is None:
            raise ValueError("C24 requires query embeddings for second stage")

        # Get candidate embeddings and scores
        cand_embeddings = chunk_embeddings[candidate_indices]
        cand_bm25 = bm25_scores[candidate_indices]

        # Dense similarity for candidates
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombMNZ fusion
        fused = self.combmnz([dense_scores, cand_bm25], normalize=True)

        # Rerank
        rerank_order = np.argsort(fused)[::-1][:self.top_k]
        final_indices = [candidate_indices[i] for i in rerank_order]
        final_scores = fused[rerank_order]

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
        # Stage 1
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Stage 2
        final_indices, final_scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

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
class C25_SemanticThenCombMNZ(TwoStageStrategy, FusionStrategy):
    """C25: Tier1 Semantic → Tier2 CombMNZ

    Stage 1: Semantic only (top 50)
    Stage 2: CombMNZ(BM25 + Semantic) on Stage 1 results
    Final: Top 10

    Test: Semantic candidate generation.
    """

    STRATEGY_ID = "c25_semantic_then_combmnz"
    CATEGORY = "two_stage"
    DESCRIPTION = "Semantic first stage → CombMNZ reranking"

    def __init__(
        self,
        first_stage_k: int = 50,
        top_k: int = 10,
        **kwargs,
    ):
        TwoStageStrategy.__init__(self, first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        """First stage: Semantic top-k."""
        if query_embedding is None:
            raise ValueError("C25 requires query embeddings")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        indices = np.argsort(dense_scores)[::-1][:self.first_stage_k]
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
        """Second stage: CombMNZ on candidates."""
        cand_embeddings = chunk_embeddings[candidate_indices]
        cand_bm25 = bm25_scores[candidate_indices] if bm25_scores is not None else np.zeros(len(candidate_indices))

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        fused = self.combmnz([dense_scores, cand_bm25], normalize=True)

        rerank_order = np.argsort(fused)[::-1][:self.top_k]
        final_indices = [candidate_indices[i] for i in rerank_order]
        final_scores = fused[rerank_order]

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
        candidates = self.first_stage_retrieve(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        final_indices, final_scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

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
class C26_UnionThenMetadata(TwoStageStrategy, FusionStrategy):
    """C26: Tier1 Union → Tier2 Metadata Rerank

    Stage 1: Union of BM25(50) + Semantic(50) → ~75 unique docs
    Stage 2: Metadata reranking (pageType + tags + links + temporal)
    Final: Top 10

    Test: Broad retrieval, smart reranking.
    """

    STRATEGY_ID = "c26_union_then_metadata"
    CATEGORY = "two_stage"
    DESCRIPTION = "Union first stage → Metadata-based reranking"

    def __init__(
        self,
        first_stage_k: int = 50,
        top_k: int = 10,
        temporal_decay_days: float = 45,
        **kwargs,
    ):
        TwoStageStrategy.__init__(self, first_stage_k=first_stage_k, **kwargs)
        self.top_k = top_k
        self.temporal_decay_days = temporal_decay_days

    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        """First stage: Union of BM25 and Semantic."""
        candidates = set()

        # BM25 candidates
        if bm25_scores is not None:
            bm25_top = np.argsort(bm25_scores)[::-1][:self.first_stage_k]
            candidates.update(bm25_top.tolist())

        # Semantic candidates
        if query_embedding is not None:
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
            dense_scores = np.dot(c_norms, q_norm)
            dense_top = np.argsort(dense_scores)[::-1][:self.first_stage_k]
            candidates.update(dense_top.tolist())

        return list(candidates)

    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Second stage: Metadata-based reranking."""
        from datetime import datetime

        cand_embeddings = chunk_embeddings[candidate_indices]
        cand_bm25 = bm25_scores[candidate_indices] if bm25_scores is not None else np.zeros(len(candidate_indices))
        cand_chunks = [chunks[i] for i in candidate_indices]

        # Base CombMNZ scores
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)
        base_scores = self.combmnz([dense_scores, cand_bm25], normalize=True)

        # Metadata boosts
        query_terms = set(query.lower().split())
        now = datetime.now()
        metadata_boost = np.ones(len(candidate_indices))

        for i, chunk in enumerate(cand_chunks):
            meta = chunk.get("metadata", {})
            boost = 1.0

            # PageType
            pagetype = str(meta.get("pageType", "")).lower()
            if pagetype in ("home", "hub"):
                boost *= 1.15

            # Tags
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_terms.update(tag.lower().replace("#", "").split("/"))
            overlap = len(query_terms & tag_terms) / max(len(query_terms), 1)
            boost *= (1 + 0.15 * overlap)

            # Links
            link_count = meta.get("link_count", 0)
            boost *= min(1 + 0.05 * np.log1p(link_count), 1.15)

            # Temporal
            created = meta.get("created", "")
            try:
                dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                days = (now - dt).days
                boost *= (1 + 0.1 * np.exp(-days / self.temporal_decay_days))
            except (ValueError, TypeError):
                pass

            metadata_boost[i] = min(boost, 2.0)

        final_scores = base_scores * metadata_boost
        rerank_order = np.argsort(final_scores)[::-1][:self.top_k]
        final_indices = [candidate_indices[i] for i in rerank_order]

        return final_indices, final_scores[rerank_order]

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

        final_indices, final_scores = self.second_stage_rerank(
            query, candidates, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

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
class C27_ConfidenceRouting(FusionStrategy):
    """C27: Confidence-Based Routing

    If BM25 top score > threshold AND gap to #2 > gap_threshold:
        Return BM25 results (fast path)
    Else:
        Use CombMNZ (BM25 + Semantic + Graph)

    Test: Short-circuit on confident matches.
    """

    STRATEGY_ID = "c27_confidence_routing"
    CATEGORY = "two_stage"
    DESCRIPTION = "Confidence-based routing: fast BM25 or full CombMNZ"

    def __init__(
        self,
        top_k: int = 10,
        bm25_threshold: float = 20.0,
        gap_threshold: float = 8.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.bm25_threshold = bm25_threshold
        self.gap_threshold = gap_threshold

    def is_confident_match(self, bm25_scores: np.ndarray) -> bool:
        """Check if BM25 has a confident top match."""
        sorted_scores = np.sort(bm25_scores)[::-1]
        if len(sorted_scores) < 2:
            return False

        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        gap = top_score - second_score

        return top_score > self.bm25_threshold and gap > self.gap_threshold

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
        if bm25_scores is None:
            raise ValueError("C27 requires BM25 scores")

        # Check confidence
        if self.is_confident_match(bm25_scores):
            # Fast path: BM25 only
            indices = np.argsort(bm25_scores)[::-1][:self.top_k]
            route = "fast_bm25"
            final_scores = bm25_scores
        else:
            # Full path: CombMNZ with graph
            if query_embedding is None:
                raise ValueError("C27 requires query embeddings for full path")

            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
            dense_scores = np.dot(c_norms, q_norm)

            graph_scores = self.get_graph_scores(chunks)
            fused = self.combmnz([dense_scores, bm25_scores, graph_scores], normalize=True)

            indices = np.argsort(fused)[::-1][:self.top_k]
            route = "full_combmnz"
            final_scores = fused

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(final_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={"strategy": self.STRATEGY_ID, "route": route},
        )
