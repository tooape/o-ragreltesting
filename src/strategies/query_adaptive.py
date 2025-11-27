"""
Query-Adaptive Strategies (C31-C33)

Strategies that adapt their behavior based on query characteristics.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    QueryAdaptiveStrategy,
    BaseStrategy,
    FusionStrategy,
    RankingResult,
    register_strategy,
)


class QueryClassifier:
    """Classify queries by type for adaptive retrieval."""

    # Keywords indicating different query types
    ENTITY_KEYWORDS = {
        "who", "person", "pm", "manager", "engineer", "lead", "contact",
    }
    STATUS_KEYWORDS = {
        "status", "current", "latest", "recent", "update", "progress",
        "where", "state", "now",
    }
    LOOKUP_KEYWORDS = {
        "what", "when", "date", "time", "link", "url", "page", "find",
        "where", "meeting", "last",
    }
    CONCEPTUAL_KEYWORDS = {
        "how", "why", "explain", "understand", "overview", "summary",
        "difference", "compare",
    }

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify a query into a type.

        Types:
        - entity: Looking for a person/team
        - status: Looking for current state of something
        - lookup: Looking for specific information
        - conceptual: Looking for understanding/explanation

        Returns:
            Query type string
        """
        query_lower = query.lower()
        words = set(re.findall(r'\w+', query_lower))

        # Score each type
        entity_score = len(words & cls.ENTITY_KEYWORDS)
        status_score = len(words & cls.STATUS_KEYWORDS)
        lookup_score = len(words & cls.LOOKUP_KEYWORDS)
        conceptual_score = len(words & cls.CONCEPTUAL_KEYWORDS)

        # Add heuristics
        if any(word in query_lower for word in ["1x1", "meeting with"]):
            entity_score += 2
        if "?" in query:
            lookup_score += 1
        if len(words) <= 3:
            lookup_score += 1  # Short queries are often lookups

        scores = {
            "entity": entity_score,
            "status": status_score,
            "lookup": lookup_score,
            "conceptual": conceptual_score,
        }

        # Return highest scoring type, default to lookup
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "lookup"
        return best_type


@register_strategy
class C31_QueryTypeAdaptive(QueryAdaptiveStrategy):
    """C31: Adapt strategy based on query type classification."""

    STRATEGY_ID = "c31_query_type_adaptive"
    CATEGORY = "query_adaptive"
    DESCRIPTION = "Adapt ranking weights based on query type"

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.classifier = QueryClassifier()

        # Strategy parameters per query type
        self.type_configs = {
            "entity": {"alpha": 0.6, "recency_weight": 0.2},  # Dense + recency
            "status": {"alpha": 0.4, "recency_weight": 0.3},  # More BM25 + strong recency
            "lookup": {"alpha": 0.5, "recency_weight": 0.1},  # Balanced
            "conceptual": {"alpha": 0.7, "recency_weight": 0.0},  # Dense-heavy
        }

    @property
    def requires_bm25(self) -> bool:
        return True

    def classify_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        return self.classifier.classify(query)

    def get_strategy_for_query_type(self, query_type: str) -> BaseStrategy:
        # Not used directly - we inline the logic
        pass

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

        # Classify query
        query_type = self.classify_query(query, query_embedding)
        config = self.type_configs.get(query_type, self.type_configs["lookup"])

        alpha = config["alpha"]
        recency_weight = config["recency_weight"]

        # Compute scores
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Normalize
        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores)

        # Recency scores
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
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            created = meta.get("created") or meta.get("dateLink")
            dt = parse_date(created)
            if dt:
                days = (datetime.now() - dt).days
                recency_scores[i] = np.exp(-days / 30)

        # Combine
        hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm
        fused = (1 - recency_weight) * hybrid + recency_weight * recency_scores

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
                "alpha": alpha,
                "recency_weight": recency_weight,
            },
        )


@register_strategy
class C32_LengthAdaptive(QueryAdaptiveStrategy):
    """C32: Adapt based on query length."""

    STRATEGY_ID = "c32_length_adaptive"
    CATEGORY = "query_adaptive"
    DESCRIPTION = "Adapt strategy based on query length"

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def classify_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        words = len(query.split())
        if words <= 2:
            return "short"
        elif words <= 5:
            return "medium"
        else:
            return "long"

    def get_strategy_for_query_type(self, query_type: str) -> BaseStrategy:
        pass

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

        length_type = self.classify_query(query, query_embedding)

        # Short queries: favor BM25 (more precise keyword match)
        # Long queries: favor dense (better semantic understanding)
        if length_type == "short":
            alpha = 0.3  # BM25-heavy
        elif length_type == "medium":
            alpha = 0.5  # Balanced
        else:
            alpha = 0.7  # Dense-heavy

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

        fused = alpha * dense_norm + (1 - alpha) * bm25_norm

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
                "length_type": length_type,
                "alpha": alpha,
            },
        )


@register_strategy
class C33_SignalConfidenceAdaptive(QueryAdaptiveStrategy):
    """C33: Adapt based on signal confidence/agreement."""

    STRATEGY_ID = "c33_confidence_adaptive"
    CATEGORY = "query_adaptive"
    DESCRIPTION = "Adapt fusion weights based on signal agreement"

    def __init__(self, top_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    def classify_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        # Classification is done dynamically based on signal agreement
        return "adaptive"

    def get_strategy_for_query_type(self, query_type: str) -> BaseStrategy:
        pass

    def compute_signal_agreement(
        self,
        dense_scores: np.ndarray,
        bm25_scores: np.ndarray,
        top_k: int = 10,
    ) -> float:
        """Compute how much dense and BM25 agree on top results.

        Returns value in [0, 1] where 1 = perfect agreement.
        """
        dense_top = set(np.argsort(dense_scores)[::-1][:top_k])
        bm25_top = set(np.argsort(bm25_scores)[::-1][:top_k])

        overlap = len(dense_top & bm25_top)
        return overlap / top_k

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
            raise ValueError("C33 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C33 requires BM25 scores")

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        # Compute agreement
        agreement = self.compute_signal_agreement(dense_scores, bm25_scores)

        # If signals agree, use equal weighting
        # If they disagree, lean toward dense (more semantic understanding)
        if agreement > 0.5:
            alpha = 0.5  # High agreement - equal weights
        elif agreement > 0.3:
            alpha = 0.6  # Medium agreement - slight dense preference
        else:
            alpha = 0.7  # Low agreement - trust dense more

        def normalize(s):
            min_s, max_s = s.min(), s.max()
            if max_s - min_s > 0:
                return (s - min_s) / (max_s - min_s)
            return np.zeros_like(s)

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores)

        fused = alpha * dense_norm + (1 - alpha) * bm25_norm

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
                "agreement": agreement,
                "alpha": alpha,
            },
        )
