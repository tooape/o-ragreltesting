"""
Temporal and Metadata Boost Strategies (C19-C24)

Strategies that incorporate temporal signals and metadata for ranking.
"""

import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue

    return None


def days_since(dt: Optional[datetime]) -> float:
    """Get days since a datetime."""
    if dt is None:
        return float("inf")
    return (datetime.now() - dt).days


@register_strategy
class C19_RecencyBoost(FusionStrategy):
    """C19: Recency boost on dense+BM25 fusion."""

    STRATEGY_ID = "c19_recency_boost"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Boost recent documents in hybrid fusion"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        recency_weight: float = 0.2,
        decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.decay_days = decay_days

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute recency scores with exponential decay."""
        scores = np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt)

            # Exponential decay
            if days < float("inf"):
                scores[i] = np.exp(-days / self.decay_days)
            else:
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
            raise ValueError("C19 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C19 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        recency_scores = self.compute_recency_scores(chunks)

        # Combine: (1-r)*hybrid + r*recency
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = (1 - self.recency_weight) * hybrid + self.recency_weight * recency_scores

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
            metadata={"strategy": self.STRATEGY_ID, "recency_weight": self.recency_weight},
        )


@register_strategy
class C20_RecencyDecay(FusionStrategy):
    """C20: Recency with configurable decay rate."""

    STRATEGY_ID = "c20_recency_decay"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Recency boost with tunable decay rate"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        recency_weight: float = 0.15,
        decay_days: float = 14,  # Faster decay
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.decay_days = decay_days

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute recency scores with exponential decay."""
        scores = np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt)

            if days < float("inf"):
                scores[i] = np.exp(-days / self.decay_days)
            else:
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
            raise ValueError("C20 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C20 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        recency_scores = self.compute_recency_scores(chunks)

        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = (1 - self.recency_weight) * hybrid + self.recency_weight * recency_scores

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
            metadata={"strategy": self.STRATEGY_ID, "decay_days": self.decay_days},
        )


@register_strategy
class C21_TagBoost(FusionStrategy):
    """C21: Boost documents with matching tags."""

    STRATEGY_ID = "c21_tag_boost"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Boost documents with query-relevant tags"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        tag_weight: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.tag_weight = tag_weight

    def compute_tag_scores(self, query: str, chunks: List[Dict]) -> np.ndarray:
        """Compute tag match scores."""
        query_terms = set(query.lower().split())
        scores = np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            tag_terms = set()
            for tag in tags:
                tag_terms.update(tag.lower().replace("#", "").split("/"))

            overlap = len(query_terms & tag_terms)
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
            raise ValueError("C21 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C21 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        tag_scores = self.compute_tag_scores(query, chunks)

        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = (1 - self.tag_weight) * hybrid + self.tag_weight * tag_scores

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
            metadata={"strategy": self.STRATEGY_ID, "tag_weight": self.tag_weight},
        )


@register_strategy
class C22_PageTypeBoost(FusionStrategy):
    """C22: Boost by pageType metadata."""

    STRATEGY_ID = "c22_pagetype_boost"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Boost documents based on pageType (hub pages ranked higher)"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        pagetype_weight: float = 0.1,
        pagetype_boosts: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.pagetype_weight = pagetype_weight
        # Default boosts for different page types
        self.pagetype_boosts = pagetype_boosts or {
            "home": 1.0,
            "hub": 0.9,
            "program": 0.8,
            "daily": 0.5,
            "misc": 0.3,
        }

    def compute_pagetype_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute pageType boost scores."""
        scores = np.zeros(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagetype = str(metadata.get("pageType", "")).lower()
            scores[i] = self.pagetype_boosts.get(pagetype, 0.5)

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
            raise ValueError("C22 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C22 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        pagetype_scores = self.compute_pagetype_scores(chunks)

        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = (1 - self.pagetype_weight) * hybrid + self.pagetype_weight * pagetype_scores

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
class C23_RecencyTagCombo(FusionStrategy):
    """C23: Combined recency and tag boosting."""

    STRATEGY_ID = "c23_recency_tag_combo"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Combined recency and tag boosting on hybrid"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        recency_weight: float = 0.1,
        tag_weight: float = 0.1,
        decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.tag_weight = tag_weight
        self.decay_days = decay_days

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt)
            if days < float("inf"):
                scores[i] = np.exp(-days / self.decay_days)
        return scores

    def compute_tag_scores(self, query: str, chunks: List[Dict]) -> np.ndarray:
        query_terms = set(query.lower().split())
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_terms.update(tag.lower().replace("#", "").split("/"))
            overlap = len(query_terms & tag_terms)
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
            raise ValueError("C23 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C23 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        recency_scores = self.compute_recency_scores(chunks)
        tag_scores = self.compute_tag_scores(query, chunks)

        hybrid_weight = 1 - self.recency_weight - self.tag_weight
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = hybrid_weight * hybrid + self.recency_weight * recency_scores + self.tag_weight * tag_scores

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
class C24_FullMetadataBoost(FusionStrategy):
    """C24: Full metadata signal fusion (recency + tags + pageType)."""

    STRATEGY_ID = "c24_full_metadata"
    CATEGORY = "temporal_metadata"
    DESCRIPTION = "Full metadata boost: recency + tags + pageType"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.5,
        recency_weight: float = 0.1,
        tag_weight: float = 0.05,
        pagetype_weight: float = 0.05,
        decay_days: float = 30,
        pagetype_boosts: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.tag_weight = tag_weight
        self.pagetype_weight = pagetype_weight
        self.decay_days = decay_days
        self.pagetype_boosts = pagetype_boosts or {
            "home": 1.0, "hub": 0.9, "program": 0.8, "daily": 0.5, "misc": 0.3,
        }

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt)
            if days < float("inf"):
                scores[i] = np.exp(-days / self.decay_days)
        return scores

    def compute_tag_scores(self, query: str, chunks: List[Dict]) -> np.ndarray:
        query_terms = set(query.lower().split())
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_terms.update(tag.lower().replace("#", "").split("/"))
            overlap = len(query_terms & tag_terms)
            scores[i] = overlap / max(len(query_terms), 1)
        return scores

    def compute_pagetype_scores(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagetype = str(metadata.get("pageType", "")).lower()
            scores[i] = self.pagetype_boosts.get(pagetype, 0.5)
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
            raise ValueError("C24 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C24 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)
        recency_scores = self.compute_recency_scores(chunks)
        tag_scores = self.compute_tag_scores(query, chunks)
        pagetype_scores = self.compute_pagetype_scores(chunks)

        hybrid_weight = 1 - self.recency_weight - self.tag_weight - self.pagetype_weight
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        fused = (
            hybrid_weight * hybrid
            + self.recency_weight * recency_scores
            + self.tag_weight * tag_scores
            + self.pagetype_weight * pagetype_scores
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
