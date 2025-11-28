"""
Temporal Boosting Variants (C15-C19)

Different temporal decay functions applied to CombMNZ base.
"""

import numpy as np
from datetime import datetime
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
    formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str)[:19], fmt)
        except ValueError:
            continue
    return None


@register_strategy
class C15_ExpDecay30d(FusionStrategy):
    """C15: Exponential Decay (30-day half-life)

    Base: CombMNZ (BM25 + Semantic)
    Temporal boost: exp(-days / 30)
    Application: Multiply final score

    Test: Gentle recency preference.
    """

    STRATEGY_ID = "c15_exp_decay_30d"
    CATEGORY = "temporal_boosting"
    DESCRIPTION = "Exponential decay (30-day half-life) on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        decay_days: float = 30,
        temporal_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.decay_days = decay_days
        self.temporal_weight = temporal_weight

    def compute_temporal_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            if dt:
                days = (now - dt).days
                scores[i] = np.exp(-days / self.decay_days)
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
            raise ValueError("C15 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C15 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # CombMNZ base
        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)

        # Temporal boost (multiplicative)
        temporal = self.compute_temporal_boost(chunks)
        fused = base_scores * (1 + self.temporal_weight * temporal)

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
class C16_ExpDecay60d(FusionStrategy):
    """C16: Exponential Decay (60-day half-life)

    Test: Slower temporal decay.
    """

    STRATEGY_ID = "c16_exp_decay_60d"
    CATEGORY = "temporal_boosting"
    DESCRIPTION = "Exponential decay (60-day half-life) on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        decay_days: float = 60,
        temporal_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.decay_days = decay_days
        self.temporal_weight = temporal_weight

    def compute_temporal_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            if dt:
                days = (now - dt).days
                scores[i] = np.exp(-days / self.decay_days)
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
            raise ValueError("C16 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C16 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        temporal = self.compute_temporal_boost(chunks)
        fused = base_scores * (1 + self.temporal_weight * temporal)

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
class C17_LinearDecay90d(FusionStrategy):
    """C17: Linear Decay (90-day window)

    Formula: 1.0 - (days / 90) if days < 90 else 0.1
    Test: Hard cutoff at 90 days.
    """

    STRATEGY_ID = "c17_linear_decay_90d"
    CATEGORY = "temporal_boosting"
    DESCRIPTION = "Linear decay (90-day window) on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        window_days: int = 90,
        min_score: float = 0.1,
        temporal_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.window_days = window_days
        self.min_score = min_score
        self.temporal_weight = temporal_weight

    def compute_temporal_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            if dt:
                days = (now - dt).days
                if days < self.window_days:
                    scores[i] = 1.0 - (days / self.window_days)
                else:
                    scores[i] = self.min_score
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
            raise ValueError("C17 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C17 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        temporal = self.compute_temporal_boost(chunks)
        fused = base_scores * (1 + self.temporal_weight * temporal)

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
            metadata={"strategy": self.STRATEGY_ID, "window_days": self.window_days},
        )


@register_strategy
class C18_StepFunction(FusionStrategy):
    """C18: Step Function Boost

    Temporal boost:
    - 0-7 days: 1.5x
    - 8-30 days: 1.2x
    - 31-90 days: 1.0x
    - 90+ days: 0.9x

    Test: Strong boost for recent, penalty for old.
    """

    STRATEGY_ID = "c18_step_function"
    CATEGORY = "temporal_boosting"
    DESCRIPTION = "Step function temporal boost on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        boost_7d: float = 1.5,
        boost_30d: float = 1.2,
        boost_90d: float = 1.0,
        boost_old: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.boost_7d = boost_7d
        self.boost_30d = boost_30d
        self.boost_90d = boost_90d
        self.boost_old = boost_old

    def compute_temporal_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.ones(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            if dt:
                days = (now - dt).days
                if days <= 7:
                    scores[i] = self.boost_7d
                elif days <= 30:
                    scores[i] = self.boost_30d
                elif days <= 90:
                    scores[i] = self.boost_90d
                else:
                    scores[i] = self.boost_old
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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        temporal = self.compute_temporal_boost(chunks)
        fused = base_scores * temporal  # Direct multiplier for step function

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
class C19_SigmoidBoost(FusionStrategy):
    """C19: Sigmoid Temporal Boost

    Formula: 1 / (1 + exp((days - 45) / 15))
    Test: Smooth transition around 45 days.
    """

    STRATEGY_ID = "c19_sigmoid_boost"
    CATEGORY = "temporal_boosting"
    DESCRIPTION = "Sigmoid temporal boost on CombMNZ (midpoint 45 days)"

    def __init__(
        self,
        top_k: int = 20,
        midpoint_days: float = 45,
        steepness: float = 15,
        temporal_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.midpoint_days = midpoint_days
        self.steepness = steepness
        self.temporal_weight = temporal_weight

    def compute_temporal_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.zeros(len(chunks))
        now = datetime.now()
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            if dt:
                days = (now - dt).days
                scores[i] = 1.0 / (1.0 + np.exp((days - self.midpoint_days) / self.steepness))
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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        temporal = self.compute_temporal_boost(chunks)
        fused = base_scores * (1 + self.temporal_weight * temporal)

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
            metadata={"strategy": self.STRATEGY_ID, "midpoint": self.midpoint_days},
        )
