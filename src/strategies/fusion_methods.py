"""
Fusion Methods (C1-C5) - Replacing RRF

CombMNZ and CombSUM variants for score-based fusion.

Key insight: RRF loses score magnitude information.
CombMNZ preserves scores while rewarding consensus across retrievers.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C1_CombMNZ_Basic(FusionStrategy):
    """C1: CombMNZ (Basic) - BM25 + Semantic

    Formula: score = SUM(normalized_scores) × COUNT(systems_retrieving)

    Expected improvement: Better than RRF on consensus docs.
    """

    STRATEGY_ID = "c1_combmnz_basic"
    CATEGORY = "fusion_methods"
    DESCRIPTION = "CombMNZ: BM25 + Semantic with min-max normalization"

    def __init__(
        self,
        top_k: int = 20,
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
            raise ValueError("C1 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C1 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombMNZ with min-max normalization
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
class C2_CombMNZ_Graph(FusionStrategy):
    """C2: CombMNZ with Graph - BM25 + Semantic + PageRank

    Test: Does graph signal help with CombMNZ?
    """

    STRATEGY_ID = "c2_combmnz_graph"
    CATEGORY = "fusion_methods"
    DESCRIPTION = "CombMNZ: BM25 + Semantic + Graph (PageRank)"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k

    def get_graph_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Extract PageRank or link-based scores from chunk metadata."""
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            # Try various graph score fields
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            # Normalize link count to score
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
            raise ValueError("C2 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C2 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Graph scores
        graph_scores = self.get_graph_scores(chunks)

        # CombMNZ with 3 signals
        fused = self.combmnz([dense_scores, bm25_scores, graph_scores], normalize=True)

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
class C3_CombMNZ_All(FusionStrategy):
    """C3: CombMNZ with All Signals - BM25 + Semantic + Graph + Temporal

    Test: Full multi-signal with CombMNZ.
    """

    STRATEGY_ID = "c3_combmnz_all"
    CATEGORY = "fusion_methods"
    DESCRIPTION = "CombMNZ: BM25 + Semantic + Graph + Temporal"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 50,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k
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
            raise ValueError("C3 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C3 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Additional signals
        graph_scores = self.get_graph_scores(chunks)
        temporal_scores = self.get_temporal_scores(chunks)

        # CombMNZ with 4 signals
        fused = self.combmnz(
            [dense_scores, bm25_scores, graph_scores, temporal_scores],
            normalize=True
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
class C4_CombSUM_Basic(FusionStrategy):
    """C4: CombSUM (Basic) - BM25 + Semantic

    Test: CombSUM vs CombMNZ (no count multiplier).
    """

    STRATEGY_ID = "c4_combsum_basic"
    CATEGORY = "fusion_methods"
    DESCRIPTION = "CombSUM: simple sum of BM25 + Semantic"

    def __init__(
        self,
        top_k: int = 20,
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
            raise ValueError("C4 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C4 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombSUM (equal weights)
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
class C5_CombSUM_All(FusionStrategy):
    """C5: CombSUM with All Signals - BM25 + Semantic + Graph + Temporal

    Test: Does dropping count penalty help?
    """

    STRATEGY_ID = "c5_combsum_all"
    CATEGORY = "fusion_methods"
    DESCRIPTION = "CombSUM: BM25 + Semantic + Graph + Temporal"

    def __init__(
        self,
        top_k: int = 20,
        first_stage_k: int = 50,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k
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
            raise ValueError("C5 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C5 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Additional signals
        graph_scores = self.get_graph_scores(chunks)
        temporal_scores = self.get_temporal_scores(chunks)

        # CombSUM with 4 signals
        fused = self.combsum(
            [dense_scores, bm25_scores, graph_scores, temporal_scores],
            normalize=True
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
