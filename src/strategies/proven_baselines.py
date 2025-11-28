"""
Proven Baselines (P1-P3) - November 2025 Benchmark Winners

These strategies achieved the best results in comprehensive benchmarking
(69 queries, 2,413 chunks, EmbeddingGemma-300M FP32).

Key finding: All top performers use LINEAR INTERPOLATION between
hybrid (dense+BM25) and recency signals, NOT multiplicative boosting.

Target to beat: MRR@5 > 0.66
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


def days_since(dt: Optional[datetime], reference: Optional[datetime] = None) -> float:
    """Get days since a datetime."""
    if dt is None:
        return float("inf")
    ref = reference or datetime.now()
    return (ref - dt).days


@register_strategy
class P1_RecencyBoost(FusionStrategy):
    """P1: Recency Boost (Best Overall) - MRR@5: 0.6623

    Hybrid dense+BM25 fusion with exponential recency boost via linear interpolation.

    Algorithm:
    1. Compute dense similarity: dense_score = cosine(query_emb, chunk_emb)
    2. Compute BM25 score: bm25_score = BM25(query, chunk)
    3. Normalize both to [0, 1] via min-max scaling
    4. Interpolate: hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm
    5. Compute recency: days_old = (today - chunk_date).days
    6. Compute recency scores: recency = exp(-days_old / decay_days)
    7. Final: score = (1 - recency_weight) * hybrid + recency_weight * recency
    """

    STRATEGY_ID = "p1_recency_boost"
    CATEGORY = "proven_baselines"
    DESCRIPTION = "Best overall: hybrid + recency via linear interpolation (MRR@5: 0.6623)"

    # Dimension-specific optimal configurations from Optuna optimization
    OPTIMAL_CONFIGS = {
        768: {"alpha": 0.624, "recency_weight": 0.340, "decay_days": 10.25},
        512: {"alpha": 0.456, "recency_weight": 0.355, "decay_days": 12.0},
        # Alternative 512d configs for reference:
        # "512_8day": {"alpha": 0.52, "recency_weight": 0.40, "decay_days": 8},
        # "512_30day": {"alpha": 0.80, "recency_weight": 0.20, "decay_days": 30},
    }

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.624,          # 768d default: 62% dense
        recency_weight: float = 0.340,  # 768d default: 34% recency
        decay_days: float = 10.25,      # 768d default: ~10 day half-life
        embedding_dim: Optional[int] = None,  # Auto-select config if provided
        reference_date: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.reference_date = reference_date

        # Auto-select optimal config based on embedding dimension
        if embedding_dim and embedding_dim in self.OPTIMAL_CONFIGS:
            config = self.OPTIMAL_CONFIGS[embedding_dim]
            self.alpha = config["alpha"]
            self.recency_weight = config["recency_weight"]
            self.decay_days = config["decay_days"]
        else:
            self.alpha = alpha
            self.recency_weight = recency_weight
            self.decay_days = decay_days

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute recency scores with exponential decay."""
        scores = np.zeros(len(chunks))
        ref_date = self.reference_date or datetime.now()

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt, ref_date)

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
            raise ValueError("P1 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("P1 requires BM25 scores")

        # Dense similarity (cosine)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize to [0, 1] via min-max
        dense_norm = self.normalize_scores(dense_scores, method="minmax")
        bm25_norm = self.normalize_scores(bm25_scores, method="minmax")

        # Hybrid fusion (dense + BM25)
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

        # Recency scores (exponential decay)
        recency_scores = self.compute_recency_scores(chunks)

        # LINEAR INTERPOLATION (not multiplicative!)
        final_scores = (1 - self.recency_weight) * hybrid + self.recency_weight * recency_scores

        # Rank by final scores
        indices = np.argsort(final_scores)[::-1][:self.top_k]

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
            metadata={
                "strategy": self.STRATEGY_ID,
                "alpha": self.alpha,
                "recency_weight": self.recency_weight,
                "decay_days": self.decay_days,
            },
        )


@register_strategy
class P2_RecencyDecay(FusionStrategy):
    """P2: Recency Decay (Higher Semantic Weight) - MRR@5: 0.6498

    Same algorithm as P1, but optimized for higher dense weight and longer decay.
    More "semantic" and less aggressive on recency.

    When to use: When semantic matching should dominate over recency
    (research queries, conceptual lookups).
    """

    STRATEGY_ID = "p2_recency_decay"
    CATEGORY = "proven_baselines"
    DESCRIPTION = "Higher semantic weight (68% dense), slower decay (MRR@5: 0.6498)"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.682,          # 68% dense (higher semantic weight)
        recency_weight: float = 0.320,  # 32% recency (less aggressive)
        decay_days: float = 11.71,      # ~12 day half-life (slower decay)
        reference_date: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.decay_days = decay_days
        self.reference_date = reference_date

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute recency scores with exponential decay."""
        scores = np.zeros(len(chunks))
        ref_date = self.reference_date or datetime.now()

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt, ref_date)

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
            raise ValueError("P2 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("P2 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize
        dense_norm = self.normalize_scores(dense_scores, method="minmax")
        bm25_norm = self.normalize_scores(bm25_scores, method="minmax")

        # Hybrid + recency
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm
        recency_scores = self.compute_recency_scores(chunks)
        final_scores = (1 - self.recency_weight) * hybrid + self.recency_weight * recency_scores

        indices = np.argsort(final_scores)[::-1][:self.top_k]

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
            metadata={
                "strategy": self.STRATEGY_ID,
                "alpha": self.alpha,
                "recency_weight": self.recency_weight,
                "decay_days": self.decay_days,
            },
        )


@register_strategy
class P3_RecencyTagCombo(FusionStrategy):
    """P3: Recency + Tag Combo - MRR@5: 0.6471

    Combines recency boosting with tag-based relevance signals.
    Documents with matching tags receive additional score boost.

    Algorithm:
    1. Compute base hybrid score (same as P1)
    2. Apply recency via linear interpolation (same as P1)
    3. Compute tag overlap between query and chunk
    4. Apply tag boost: tag_boost = tag_overlap_score * tag_weight
    5. Final: score = recency_boosted_score * (1 + tag_boost)

    When to use: When chunks have meaningful, consistent tags.
    Adds complexity for marginal gain (+3ms latency).
    """

    STRATEGY_ID = "p3_recency_tag_combo"
    CATEGORY = "proven_baselines"
    DESCRIPTION = "Recency + tag overlap boost (MRR@5: 0.6471, +3ms latency)"

    def __init__(
        self,
        top_k: int = 20,
        alpha: float = 0.636,          # 64% dense
        recency_weight: float = 0.219,  # 22% recency
        tag_weight: float = 0.272,      # 27% tag boost factor
        decay_days: float = 13.32,      # ~13 day half-life
        reference_date: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.alpha = alpha
        self.recency_weight = recency_weight
        self.tag_weight = tag_weight
        self.decay_days = decay_days
        self.reference_date = reference_date

    def compute_recency_scores(self, chunks: List[Dict]) -> np.ndarray:
        """Compute recency scores with exponential decay."""
        scores = np.zeros(len(chunks))
        ref_date = self.reference_date or datetime.now()

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            created = metadata.get("created") or metadata.get("dateLink")
            dt = parse_date(str(created) if created else "")
            days = days_since(dt, ref_date)

            if days < float("inf"):
                scores[i] = np.exp(-days / self.decay_days)
            else:
                scores[i] = 0.0

        return scores

    def compute_tag_overlap(
        self,
        query: str,
        chunks: List[Dict],
        query_tags: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute tag overlap scores.

        If query_tags provided, use those. Otherwise extract from query text.
        """
        scores = np.zeros(len(chunks))

        # Extract query terms for tag matching
        if query_tags:
            query_terms = set(t.lower().replace("#", "") for t in query_tags)
        else:
            # Simple: use query words as potential tag matches
            query_terms = set(query.lower().split())

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            # Extract tag terms (handle #tag and tag/subtag formats)
            tag_terms = set()
            for tag in tags:
                tag_clean = tag.lower().replace("#", "")
                tag_terms.add(tag_clean)
                tag_terms.update(tag_clean.split("/"))

            # Compute overlap ratio
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
            raise ValueError("P3 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("P3 requires BM25 scores")

        # Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize
        dense_norm = self.normalize_scores(dense_scores, method="minmax")
        bm25_norm = self.normalize_scores(bm25_scores, method="minmax")

        # Hybrid fusion
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm

        # Recency boost via linear interpolation
        recency_scores = self.compute_recency_scores(chunks)
        recency_boosted = (1 - self.recency_weight) * hybrid + self.recency_weight * recency_scores

        # Tag overlap boost (multiplicative)
        query_tags = metadata.get("query_tags") if metadata else None
        tag_overlap = self.compute_tag_overlap(query, chunks, query_tags)
        final_scores = recency_boosted * (1 + self.tag_weight * tag_overlap)

        indices = np.argsort(final_scores)[::-1][:self.top_k]

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
            metadata={
                "strategy": self.STRATEGY_ID,
                "alpha": self.alpha,
                "recency_weight": self.recency_weight,
                "tag_weight": self.tag_weight,
                "decay_days": self.decay_days,
            },
        )
