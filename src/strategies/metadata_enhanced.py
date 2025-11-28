"""
Metadata-Enhanced Reranking (C20-C23)

Strategies that leverage document metadata for ranking.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C20_PageTypeAware(FusionStrategy):
    """C20: PageType-Aware Reranking

    Metadata boosts:
    - programHome: 1.15x (hub documents)
    - personNote: 1.10x
    - dailyNote: 1.05x if query has temporal words
    - No boost: others
    """

    STRATEGY_ID = "c20_pagetype_aware"
    CATEGORY = "metadata_enhanced"
    DESCRIPTION = "PageType-aware boosting on CombMNZ"

    TEMPORAL_KEYWORDS = {"recent", "latest", "today", "yesterday", "last", "this week", "new"}

    def __init__(
        self,
        top_k: int = 20,
        home_boost: float = 1.15,
        person_boost: float = 1.10,
        daily_boost: float = 1.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.home_boost = home_boost
        self.person_boost = person_boost
        self.daily_boost = daily_boost

    def has_temporal_words(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)

    def compute_pagetype_boost(self, query: str, chunks: List[Dict]) -> np.ndarray:
        scores = np.ones(len(chunks))
        temporal_query = self.has_temporal_words(query)

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            pagetype = str(metadata.get("pageType", "")).lower()

            if pagetype in ("home", "programhome", "hub"):
                scores[i] = self.home_boost
            elif pagetype in ("person", "personnote"):
                scores[i] = self.person_boost
            elif pagetype in ("daily", "dailynote") and temporal_query:
                scores[i] = self.daily_boost

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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        pagetype_boost = self.compute_pagetype_boost(query, chunks)
        fused = base_scores * pagetype_boost

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
class C21_TagOverlap(FusionStrategy):
    """C21: Tag Overlap Boosting

    Tag boost: 1.0 + (0.15 × tag_overlap_ratio)
    Where: tag_overlap = |query_tags ∩ doc_tags| / |query_tags|
    """

    STRATEGY_ID = "c21_tag_overlap"
    CATEGORY = "metadata_enhanced"
    DESCRIPTION = "Tag overlap boosting on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        tag_boost_factor: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.tag_boost_factor = tag_boost_factor

    def compute_tag_boost(self, query: str, chunks: List[Dict]) -> np.ndarray:
        query_terms = set(query.lower().split())
        scores = np.ones(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            tag_terms = set()
            for tag in tags:
                tag_clean = tag.lower().replace("#", "")
                tag_terms.add(tag_clean)
                tag_terms.update(tag_clean.split("/"))

            overlap = len(query_terms & tag_terms)
            overlap_ratio = overlap / max(len(query_terms), 1)
            scores[i] = 1.0 + (self.tag_boost_factor * overlap_ratio)

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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        tag_boost = self.compute_tag_boost(query, chunks)
        fused = base_scores * tag_boost

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
class C22_LinkDensity(FusionStrategy):
    """C22: Link Density Boosting

    Link boost: 1.0 + (0.05 × log(1 + link_count))
    Cap: 1.3x maximum boost

    Hub documents get visibility.
    """

    STRATEGY_ID = "c22_link_density"
    CATEGORY = "metadata_enhanced"
    DESCRIPTION = "Link density boosting on CombMNZ"

    def __init__(
        self,
        top_k: int = 20,
        link_boost_factor: float = 0.05,
        max_boost: float = 1.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.link_boost_factor = link_boost_factor
        self.max_boost = max_boost

    def compute_link_boost(self, chunks: List[Dict]) -> np.ndarray:
        scores = np.ones(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            link_count = metadata.get("link_count", 0)
            # Also check for wikilinks in content if available
            content = chunk.get("content", "")
            if link_count == 0 and content:
                link_count = content.count("[[")

            boost = 1.0 + (self.link_boost_factor * np.log1p(link_count))
            scores[i] = min(boost, self.max_boost)

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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        link_boost = self.compute_link_boost(chunks)
        fused = base_scores * link_boost

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
class C23_CombinedMetadata(FusionStrategy):
    """C23: Combined Metadata Signals

    Boosts (multiplicative):
    - PageType: 1.0-1.15x
    - Tag overlap: 1.0-1.15x
    - Link density: 1.0-1.15x
    - Temporal: exp(-days / 45)

    Max combined boost: 2.0x
    """

    STRATEGY_ID = "c23_combined_metadata"
    CATEGORY = "metadata_enhanced"
    DESCRIPTION = "Combined metadata signals (pageType + tags + links + temporal)"

    TEMPORAL_KEYWORDS = {"recent", "latest", "today", "yesterday", "last", "this week", "new"}

    def __init__(
        self,
        top_k: int = 20,
        pagetype_boost: float = 1.15,
        tag_boost_factor: float = 0.15,
        link_boost_factor: float = 0.05,
        temporal_decay_days: float = 45,
        max_combined_boost: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.pagetype_boost = pagetype_boost
        self.tag_boost_factor = tag_boost_factor
        self.link_boost_factor = link_boost_factor
        self.temporal_decay_days = temporal_decay_days
        self.max_combined_boost = max_combined_boost

    def has_temporal_words(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)

    def compute_all_boosts(self, query: str, chunks: List[Dict]) -> np.ndarray:
        query_terms = set(query.lower().split())
        temporal_query = self.has_temporal_words(query)
        now = datetime.now()

        combined = np.ones(len(chunks))

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            boost = 1.0

            # PageType boost
            pagetype = str(metadata.get("pageType", "")).lower()
            if pagetype in ("home", "programhome", "hub"):
                boost *= self.pagetype_boost
            elif pagetype in ("person", "personnote"):
                boost *= 1.10
            elif pagetype in ("daily", "dailynote") and temporal_query:
                boost *= 1.05

            # Tag overlap boost
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_clean = tag.lower().replace("#", "")
                tag_terms.add(tag_clean)
                tag_terms.update(tag_clean.split("/"))
            overlap = len(query_terms & tag_terms)
            overlap_ratio = overlap / max(len(query_terms), 1)
            boost *= (1.0 + self.tag_boost_factor * overlap_ratio)

            # Link density boost
            link_count = metadata.get("link_count", 0)
            content = chunk.get("content", "")
            if link_count == 0 and content:
                link_count = content.count("[[")
            link_boost = min(1.0 + self.link_boost_factor * np.log1p(link_count), 1.15)
            boost *= link_boost

            # Temporal boost
            created = metadata.get("created") or metadata.get("dateLink")
            if created:
                try:
                    dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                    days = (now - dt).days
                    temporal_boost = np.exp(-days / self.temporal_decay_days)
                    boost *= (1.0 + 0.15 * temporal_boost)
                except (ValueError, TypeError):
                    pass

            combined[i] = min(boost, self.max_combined_boost)

        return combined

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

        base_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        metadata_boost = self.compute_all_boosts(query, chunks)
        fused = base_scores * metadata_boost

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
