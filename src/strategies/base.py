"""
Base strategy class for O-RAG ranking strategies.

All 37 ranking strategies inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time


@dataclass
class StrategyConfig:
    """Configuration for a ranking strategy."""
    name: str
    category: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RankingResult:
    """Result from a ranking strategy."""
    query_id: str
    ranked_chunk_ids: List[str]
    scores: Dict[str, float]  # chunk_id -> score
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all ranking strategies.

    All strategies must implement:
    - rank(): Rank chunks for a single query
    - get_config(): Return strategy configuration
    """

    # Strategy identifiers (override in subclasses)
    STRATEGY_ID: str = "base"
    CATEGORY: str = "base"
    DESCRIPTION: str = "Base strategy"

    def __init__(self, **kwargs):
        """Initialize strategy with parameters.

        Args:
            **kwargs: Strategy-specific parameters
        """
        self.params = kwargs
        self._initialized = False

    @abstractmethod
    def rank(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RankingResult:
        """Rank chunks for a query.

        Args:
            query: Query text
            query_embedding: Query embedding vector (may be None for BM25-only)
            chunk_embeddings: Matrix of chunk embeddings (n_chunks x dim)
            chunks: List of chunk metadata dicts
            bm25_scores: Optional pre-computed BM25 scores
            metadata: Optional additional metadata

        Returns:
            RankingResult with ranked chunk IDs and scores
        """
        pass

    def rank_batch(
        self,
        queries: Dict[str, str],
        query_embeddings: Dict[str, np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RankingResult]:
        """Rank chunks for multiple queries.

        Default implementation calls rank() for each query.
        Override for batch-optimized strategies.

        Args:
            queries: Dict of query_id -> query_text
            query_embeddings: Dict of query_id -> embedding
            chunk_embeddings: Matrix of chunk embeddings
            chunks: List of chunk metadata dicts
            bm25_scores: Optional dict of query_id -> BM25 scores
            metadata: Optional additional metadata

        Returns:
            Dict of query_id -> RankingResult
        """
        results = {}
        for qid, query_text in queries.items():
            q_emb = query_embeddings.get(qid)
            bm25 = bm25_scores.get(qid) if bm25_scores else None

            results[qid] = self.rank(
                query=query_text,
                query_embedding=q_emb,
                chunk_embeddings=chunk_embeddings,
                chunks=chunks,
                bm25_scores=bm25,
                metadata=metadata,
            )
            results[qid].query_id = qid

        return results

    def get_config(self) -> StrategyConfig:
        """Get strategy configuration.

        Returns:
            StrategyConfig with name, category, and parameters
        """
        return StrategyConfig(
            name=self.STRATEGY_ID,
            category=self.CATEGORY,
            description=self.DESCRIPTION,
            parameters=self.params.copy(),
        )

    def initialize(self, **kwargs) -> None:
        """Optional initialization hook.

        Called before ranking to allow lazy initialization.
        Override for strategies that need setup.
        """
        self._initialized = True

    @property
    def requires_embeddings(self) -> bool:
        """Whether strategy requires dense embeddings.

        Override to return False for BM25-only strategies.
        """
        return True

    @property
    def requires_bm25(self) -> bool:
        """Whether strategy requires BM25 scores.

        Override to return True for strategies using BM25.
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.STRATEGY_ID}, {self.params})"


class DenseRetrievalStrategy(BaseStrategy):
    """Base class for dense (embedding-based) retrieval strategies."""

    CATEGORY = "dense"

    def cosine_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and all chunks.

        Args:
            query_embedding: Query embedding (dim,)
            chunk_embeddings: Chunk embeddings (n_chunks, dim)

        Returns:
            Similarity scores (n_chunks,)
        """
        # Normalize (in case not already normalized)
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)

        return np.dot(c_norms, q_norm)

    def rank_by_scores(
        self,
        scores: np.ndarray,
        chunks: List[Dict],
        top_k: int = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Rank chunks by scores.

        Args:
            scores: Score array (n_chunks,)
            chunks: List of chunk dicts (must have 'id' or index used)
            top_k: Number of results (None = all)

        Returns:
            Tuple of (ranked_ids, score_dict)
        """
        indices = np.argsort(scores)[::-1]
        if top_k:
            indices = indices[:top_k]

        ranked_ids = []
        score_dict = {}

        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(scores[idx])

        return ranked_ids, score_dict


class FusionStrategy(BaseStrategy):
    """Base class for fusion (hybrid) retrieval strategies."""

    CATEGORY = "fusion"

    @property
    def requires_bm25(self) -> bool:
        return True

    def normalize_scores(
        self,
        scores: np.ndarray,
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize scores to [0, 1].

        Args:
            scores: Raw scores
            method: 'minmax', 'zscore', or 'rank'

        Returns:
            Normalized scores
        """
        if method == "minmax":
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s > 0:
                return (scores - min_s) / (max_s - min_s)
            return np.zeros_like(scores)

        elif method == "zscore":
            mean, std = scores.mean(), scores.std()
            if std > 0:
                return (scores - mean) / std
            return np.zeros_like(scores)

        elif method == "rank":
            # Convert to rank-based scores (higher rank = higher score)
            ranks = np.argsort(np.argsort(scores))
            return ranks / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(scores)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def reciprocal_rank_fusion(
        self,
        rankings: List[np.ndarray],
        k: int = 60,
    ) -> np.ndarray:
        """Reciprocal Rank Fusion.

        Args:
            rankings: List of rank arrays (lower = better)
            k: RRF parameter (default 60)

        Returns:
            Fused scores
        """
        n_docs = rankings[0].shape[0]
        fused = np.zeros(n_docs)

        for ranks in rankings:
            for doc_idx in range(n_docs):
                fused[doc_idx] += 1.0 / (k + ranks[doc_idx] + 1)

        return fused

    def combmnz(
        self,
        score_lists: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """CombMNZ fusion.

        Score = sum(scores) * num_systems_retrieving_doc

        Args:
            score_lists: List of score arrays (normalized recommended)
            normalize: Normalize inputs before fusion

        Returns:
            Fused scores
        """
        if normalize:
            score_lists = [self.normalize_scores(s) for s in score_lists]

        n_docs = score_lists[0].shape[0]
        combined = np.zeros(n_docs)
        counts = np.zeros(n_docs)

        for scores in score_lists:
            combined += scores
            counts += (scores > 0).astype(float)

        return combined * counts

    def combsum(
        self,
        score_lists: List[np.ndarray],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """CombSUM fusion (optionally weighted).

        Args:
            score_lists: List of score arrays
            weights: Optional weights for each system
            normalize: Normalize inputs before fusion

        Returns:
            Fused scores
        """
        if normalize:
            score_lists = [self.normalize_scores(s) for s in score_lists]

        if weights is None:
            weights = [1.0] * len(score_lists)

        combined = np.zeros(score_lists[0].shape[0])
        for scores, weight in zip(score_lists, weights):
            combined += weight * scores

        return combined


class TwoStageStrategy(BaseStrategy):
    """Base class for two-stage retrieval strategies."""

    CATEGORY = "two_stage"

    def __init__(
        self,
        first_stage_k: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.first_stage_k = first_stage_k

    @abstractmethod
    def first_stage_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        """First stage retrieval (fast, broad).

        Args:
            query: Query text
            query_embedding: Query embedding
            chunk_embeddings: All chunk embeddings
            chunks: All chunks
            bm25_scores: Optional BM25 scores

        Returns:
            List of candidate chunk indices
        """
        pass

    @abstractmethod
    def second_stage_rerank(
        self,
        query: str,
        candidate_indices: List[int],
        query_embedding: Optional[np.ndarray],
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Second stage reranking (accurate, on candidates).

        Args:
            query: Query text
            candidate_indices: Indices from first stage
            query_embedding: Query embedding
            chunk_embeddings: All chunk embeddings
            chunks: All chunks
            bm25_scores: Optional BM25 scores

        Returns:
            Tuple of (reranked_indices, scores)
        """
        pass


class QueryAdaptiveStrategy(BaseStrategy):
    """Base class for query-adaptive strategies."""

    CATEGORY = "query_adaptive"

    @abstractmethod
    def classify_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Classify query type.

        Args:
            query: Query text
            query_embedding: Optional query embedding

        Returns:
            Query type identifier
        """
        pass

    @abstractmethod
    def get_strategy_for_query_type(
        self,
        query_type: str,
    ) -> BaseStrategy:
        """Get appropriate strategy for query type.

        Args:
            query_type: Query type identifier

        Returns:
            Strategy instance to use
        """
        pass


# Strategy registry
STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(cls: type) -> type:
    """Decorator to register a strategy class.

    Usage:
        @register_strategy
        class MyStrategy(BaseStrategy):
            STRATEGY_ID = "my_strategy"
            ...
    """
    STRATEGY_REGISTRY[cls.STRATEGY_ID] = cls
    return cls


def get_strategy(strategy_id: str, **kwargs) -> BaseStrategy:
    """Get a strategy instance by ID.

    Args:
        strategy_id: Strategy identifier
        **kwargs: Strategy parameters

    Returns:
        Strategy instance
    """
    if strategy_id not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy_id}. Available: {available}")

    return STRATEGY_REGISTRY[strategy_id](**kwargs)


def list_strategies() -> List[Dict[str, str]]:
    """List all registered strategies.

    Returns:
        List of strategy info dicts
    """
    return [
        {
            "id": cls.STRATEGY_ID,
            "category": cls.CATEGORY,
            "description": cls.DESCRIPTION,
        }
        for cls in STRATEGY_REGISTRY.values()
    ]
