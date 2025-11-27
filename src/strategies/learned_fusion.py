"""
Learned Fusion Strategies (C34-C37)

Strategies that use learned parameters or optimization for fusion.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .base import (
    BaseStrategy,
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@dataclass
class OptimizedWeights:
    """Container for learned/optimized weights."""
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    recency_weight: float = 0.0
    tag_weight: float = 0.0
    pagetype_weight: float = 0.0
    rrf_k: int = 60

    @classmethod
    def from_dict(cls, d: Dict) -> "OptimizedWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@register_strategy
class C34_GridSearchOptimal(FusionStrategy):
    """C34: Grid-search optimized weights (placeholder for learned values)."""

    STRATEGY_ID = "c34_grid_search"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "Grid-search optimized fusion weights"

    def __init__(
        self,
        top_k: int = 20,
        weights: Optional[OptimizedWeights] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        # Default weights - will be replaced by grid search results
        self.weights = weights or OptimizedWeights(
            dense_weight=0.55,
            bm25_weight=0.35,
            recency_weight=0.1,
        )

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
            raise ValueError("C34 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C34 requires BM25 scores")

        w = self.weights

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        # Recency
        from datetime import datetime

        def parse_date(d):
            if not d:
                return None
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(str(d), fmt)
                except:
                    pass
            return None

        recency_scores = np.zeros(len(chunks))
        if w.recency_weight > 0:
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                created = meta.get("created") or meta.get("dateLink")
                dt = parse_date(created)
                if dt:
                    days = (datetime.now() - dt).days
                    recency_scores[i] = np.exp(-days / 30)

        # Combine with optimized weights
        total_weight = w.dense_weight + w.bm25_weight + w.recency_weight
        fused = (
            (w.dense_weight / total_weight) * dense_norm
            + (w.bm25_weight / total_weight) * bm25_norm
            + (w.recency_weight / total_weight) * recency_scores
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
            metadata={
                "strategy": self.STRATEGY_ID,
                "weights": {
                    "dense": w.dense_weight,
                    "bm25": w.bm25_weight,
                    "recency": w.recency_weight,
                },
            },
        )


@register_strategy
class C35_BayesianOptimal(FusionStrategy):
    """C35: Bayesian-optimized weights (placeholder for Optuna results)."""

    STRATEGY_ID = "c35_bayesian"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "Bayesian-optimized fusion weights (Optuna)"

    def __init__(
        self,
        top_k: int = 20,
        weights: Optional[OptimizedWeights] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        # Placeholder - will be set by Optuna optimization
        self.weights = weights or OptimizedWeights(
            dense_weight=0.6,
            bm25_weight=0.3,
            recency_weight=0.1,
            rrf_k=45,
        )

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
            raise ValueError("C35 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C35 requires BM25 scores")

        w = self.weights

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Weighted RRF with optimized k and weights
        dense_ranks = np.argsort(np.argsort(-dense_scores))
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))

        n_docs = len(chunks)
        rrf_scores = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            rrf_scores[doc_idx] = (
                w.dense_weight / (w.rrf_k + dense_ranks[doc_idx] + 1)
                + w.bm25_weight / (w.rrf_k + bm25_ranks[doc_idx] + 1)
            )

        # Add recency boost
        if w.recency_weight > 0:
            from datetime import datetime

            def parse_date(d):
                if not d:
                    return None
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        return datetime.strptime(str(d), fmt)
                    except:
                        pass
                return None

            recency_scores = np.zeros(n_docs)
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                created = meta.get("created") or meta.get("dateLink")
                dt = parse_date(created)
                if dt:
                    days = (datetime.now() - dt).days
                    recency_scores[i] = np.exp(-days / 30)

            rrf_norm = self.normalize_scores(rrf_scores)
            fused = (1 - w.recency_weight) * rrf_norm + w.recency_weight * recency_scores
        else:
            fused = rrf_scores

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
            metadata={
                "strategy": self.STRATEGY_ID,
                "rrf_k": w.rrf_k,
                "weights": {
                    "dense": w.dense_weight,
                    "bm25": w.bm25_weight,
                    "recency": w.recency_weight,
                },
            },
        )


@register_strategy
class C36_PerQueryOptimal(FusionStrategy):
    """C36: Per-query type optimized weights."""

    STRATEGY_ID = "c36_per_query"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "Per-query-type optimized fusion weights"

    def __init__(
        self,
        top_k: int = 20,
        type_weights: Optional[Dict[str, OptimizedWeights]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        # Default per-type weights - will be learned
        self.type_weights = type_weights or {
            "entity": OptimizedWeights(dense_weight=0.6, bm25_weight=0.2, recency_weight=0.2),
            "status": OptimizedWeights(dense_weight=0.4, bm25_weight=0.3, recency_weight=0.3),
            "lookup": OptimizedWeights(dense_weight=0.5, bm25_weight=0.4, recency_weight=0.1),
            "conceptual": OptimizedWeights(dense_weight=0.7, bm25_weight=0.3, recency_weight=0.0),
        }

    def classify_query(self, query: str) -> str:
        """Simple query classification."""
        import re

        query_lower = query.lower()
        words = set(re.findall(r'\w+', query_lower))

        entity_kw = {"who", "person", "pm", "manager", "engineer", "lead"}
        status_kw = {"status", "current", "latest", "recent", "update", "progress"}
        conceptual_kw = {"how", "why", "explain", "understand", "overview"}

        if words & entity_kw or "1x1" in query_lower:
            return "entity"
        if words & status_kw:
            return "status"
        if words & conceptual_kw:
            return "conceptual"
        return "lookup"

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
            raise ValueError("C36 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C36 requires BM25 scores")

        query_type = self.classify_query(query)
        w = self.type_weights.get(query_type, self.type_weights["lookup"])

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        dense_norm = self.normalize_scores(dense_scores)
        bm25_norm = self.normalize_scores(bm25_scores)

        # Recency
        from datetime import datetime

        def parse_date(d):
            if not d:
                return None
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(str(d), fmt)
                except:
                    pass
            return None

        recency_scores = np.zeros(len(chunks))
        if w.recency_weight > 0:
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                created = meta.get("created") or meta.get("dateLink")
                dt = parse_date(created)
                if dt:
                    days = (datetime.now() - dt).days
                    recency_scores[i] = np.exp(-days / 30)

        total_weight = w.dense_weight + w.bm25_weight + w.recency_weight
        fused = (
            (w.dense_weight / total_weight) * dense_norm
            + (w.bm25_weight / total_weight) * bm25_norm
            + (w.recency_weight / total_weight) * recency_scores
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
            metadata={
                "strategy": self.STRATEGY_ID,
                "query_type": query_type,
                "weights": {
                    "dense": w.dense_weight,
                    "bm25": w.bm25_weight,
                    "recency": w.recency_weight,
                },
            },
        )


@register_strategy
class C37_EnsembleVoting(FusionStrategy):
    """C37: Ensemble voting from multiple strategies."""

    STRATEGY_ID = "c37_ensemble"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "Ensemble voting from top strategies"

    def __init__(
        self,
        top_k: int = 20,
        ensemble_size: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.ensemble_size = ensemble_size

    def run_substrategy(
        self,
        strategy_type: str,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        bm25_scores: np.ndarray,
        chunks: List[Dict],
    ) -> List[int]:
        """Run a sub-strategy and return top-k indices."""
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores)

        if strategy_type == "dense":
            scores = dense_norm
        elif strategy_type == "bm25":
            scores = bm25_norm
        elif strategy_type == "balanced":
            scores = 0.5 * dense_norm + 0.5 * bm25_norm
        elif strategy_type == "dense_heavy":
            scores = 0.7 * dense_norm + 0.3 * bm25_norm
        elif strategy_type == "bm25_heavy":
            scores = 0.3 * dense_norm + 0.7 * bm25_norm
        else:
            scores = dense_norm

        return np.argsort(scores)[::-1][:self.top_k * 2].tolist()

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
            raise ValueError("C37 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C37 requires BM25 scores")

        # Run ensemble of strategies
        strategies = ["dense", "bm25", "balanced", "dense_heavy", "bm25_heavy"]

        # Collect votes
        vote_counts = {}
        vote_positions = {}

        for strat in strategies[:self.ensemble_size]:
            top_indices = self.run_substrategy(
                strat, query_embedding, chunk_embeddings, bm25_scores, chunks
            )
            for pos, idx in enumerate(top_indices):
                if idx not in vote_counts:
                    vote_counts[idx] = 0
                    vote_positions[idx] = []
                vote_counts[idx] += 1
                vote_positions[idx].append(pos)

        # Score by: votes * (1 / avg_position)
        final_scores = {}
        for idx, votes in vote_counts.items():
            avg_pos = np.mean(vote_positions[idx])
            final_scores[idx] = votes / (avg_pos + 1)

        # Rank by final scores
        sorted_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        top_indices = sorted_indices[:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in top_indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(final_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={
                "strategy": self.STRATEGY_ID,
                "ensemble_size": self.ensemble_size,
            },
        )
