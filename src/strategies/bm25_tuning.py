"""
BM25 Parameter Tuning (C33-C35)

Different BM25 configurations for various document types.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    BaseStrategy,
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C33_BM25ShortDocs(BaseStrategy):
    """C33: BM25 Tuned for Short Docs

    Parameters:
      k1: 1.2 (less term freq saturation)
      b: 0.5 (less length normalization)

    Test: Optimized for note-length documents.
    Note: Assumes BM25 scores are pre-computed with these params.
    """

    STRATEGY_ID = "c33_bm25_short_docs"
    CATEGORY = "bm25_tuning"
    DESCRIPTION = "BM25 tuned for short docs (k1=1.2, b=0.5)"

    # Recommended BM25 params (for reference - actual BM25 computed externally)
    RECOMMENDED_K1 = 1.2
    RECOMMENDED_B = 0.5

    def __init__(
        self,
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    @property
    def requires_embeddings(self) -> bool:
        return False

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
            raise ValueError("C33 requires BM25 scores")

        # Pure BM25 ranking (scores assumed computed with k1=1.2, b=0.5)
        indices = np.argsort(bm25_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(bm25_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={
                "strategy": self.STRATEGY_ID,
                "bm25_params": {"k1": self.RECOMMENDED_K1, "b": self.RECOMMENDED_B},
            },
        )


@register_strategy
class C34_BM25LongDocs(BaseStrategy):
    """C34: BM25 Tuned for Long Docs

    Parameters:
      k1: 1.8 (more term freq saturation)
      b: 0.85 (more length normalization)

    Test: Penalize length less.
    Note: Assumes BM25 scores are pre-computed with these params.
    """

    STRATEGY_ID = "c34_bm25_long_docs"
    CATEGORY = "bm25_tuning"
    DESCRIPTION = "BM25 tuned for long docs (k1=1.8, b=0.85)"

    RECOMMENDED_K1 = 1.8
    RECOMMENDED_B = 0.85

    def __init__(
        self,
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k

    @property
    def requires_bm25(self) -> bool:
        return True

    @property
    def requires_embeddings(self) -> bool:
        return False

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
            raise ValueError("C34 requires BM25 scores")

        indices = np.argsort(bm25_scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(bm25_scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={
                "strategy": self.STRATEGY_ID,
                "bm25_params": {"k1": self.RECOMMENDED_K1, "b": self.RECOMMENDED_B},
            },
        )


@register_strategy
class C35_BM25PlusExpansion(FusionStrategy):
    """C35: BM25+ with Query Expansion

    Preprocessing:
      - Expand acronyms (CKG → "CKG creative knowledge graph")
      - Add synonyms from aliases in frontmatter
      - Stemming (simulated)

    BM25 Parameters: k1=1.2, b=0.5

    Test: Better term coverage.
    """

    STRATEGY_ID = "c35_bm25_plus_expansion"
    CATEGORY = "bm25_tuning"
    DESCRIPTION = "BM25+ with query expansion (acronyms, aliases)"

    # Acronym expansion map
    ACRONYM_EXPANSIONS = {
        "ckg": "CKG creative knowledge graph",
        "psw": "PsW photoshop web",
        "lr": "Lr lightroom",
        "ai": "AI artificial intelligence",
        "ml": "ML machine learning",
        "srl": "SRL semantic role labeling",
        "ner": "NER named entity recognition",
        "rag": "RAG retrieval augmented generation",
        "llm": "LLM large language model",
    }

    def __init__(
        self,
        top_k: int = 10,
        expansion_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.expansion_weight = expansion_weight

    def expand_query(self, query: str) -> str:
        """Expand query with acronyms and synonyms."""
        words = query.lower().split()
        expanded = [query]

        for word in words:
            if word in self.ACRONYM_EXPANSIONS:
                expanded.append(self.ACRONYM_EXPANSIONS[word])

        return " ".join(expanded)

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
            raise ValueError("C35 requires BM25 scores")

        # Query expansion (note: actual BM25 recomputation would happen externally)
        expanded_query = self.expand_query(query)

        # For now, use provided BM25 scores with optional alias boosting
        # In production, would recompute BM25 with expanded query
        scores = bm25_scores.copy()

        # Boost chunks whose aliases match expanded terms
        expanded_terms = set(expanded_query.lower().split())
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            aliases = meta.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]

            alias_terms = set()
            for alias in aliases:
                alias_terms.update(alias.lower().split())

            overlap = len(expanded_terms & alias_terms)
            if overlap > 0:
                scores[i] *= (1 + 0.1 * overlap)

        indices = np.argsort(scores)[::-1][:self.top_k]

        ranked_ids = []
        score_dict = {}
        for idx in indices:
            chunk_id = chunks[idx].get("id", str(idx))
            ranked_ids.append(chunk_id)
            score_dict[chunk_id] = float(scores[idx])

        return RankingResult(
            query_id="",
            ranked_chunk_ids=ranked_ids,
            scores=score_dict,
            metadata={
                "strategy": self.STRATEGY_ID,
                "expanded_query": expanded_query,
            },
        )
