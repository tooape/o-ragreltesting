"""
Learned Fusion (C36-C37)

Strategies using learned/optimized models for fusion.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    FusionStrategy,
    RankingResult,
    register_strategy,
)


@register_strategy
class C36_LogisticRegression(FusionStrategy):
    """C36: Logistic Regression Fusion

    Features (7 total):
      1. Dense similarity score
      2. BM25 score
      3. PageRank / link count
      4. Temporal recency
      5. Tag overlap
      6. PageType match
      7. Title match score

    Combines features with pre-trained logistic regression weights.
    Note: This is a simulation - actual weights would be learned from labeled data.
    """

    STRATEGY_ID = "c36_logistic_regression"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "Logistic regression fusion with 7 features"

    # Pre-trained weights (simulated - would be learned from labeled data)
    # Positive weights indicate features that increase relevance
    FEATURE_WEIGHTS = {
        "dense": 0.45,
        "bm25": 0.30,
        "pagerank": 0.08,
        "recency": 0.07,
        "tag_overlap": 0.04,
        "pagetype": 0.03,
        "title_match": 0.03,
    }
    BIAS = -0.1

    def __init__(
        self,
        top_k: int = 10,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.temporal_decay_days = temporal_decay_days

    def extract_features(
        self,
        query: str,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: np.ndarray,
    ) -> np.ndarray:
        """Extract 7 features for each document."""
        from datetime import datetime

        n_docs = len(chunks)
        features = np.zeros((n_docs, 7))
        query_terms = set(query.lower().split())
        now = datetime.now()

        # Feature 1: Dense similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        features[:, 0] = np.dot(c_norms, q_norm)

        # Feature 2: BM25
        features[:, 1] = bm25_scores

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})

            # Feature 3: PageRank / link density
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            content = chunk.get("content", "")
            if link_count == 0 and content:
                link_count = content.count("[[")
            features[i, 2] = pagerank if pagerank > 0 else np.log1p(link_count) / 10.0

            # Feature 4: Temporal recency
            created = metadata.get("created") or metadata.get("dateLink")
            if created:
                try:
                    dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                    days = (now - dt).days
                    features[i, 3] = np.exp(-days / self.temporal_decay_days)
                except (ValueError, TypeError):
                    pass

            # Feature 5: Tag overlap
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_clean = tag.lower().replace("#", "")
                tag_terms.add(tag_clean)
                tag_terms.update(tag_clean.split("/"))
            overlap = len(query_terms & tag_terms)
            features[i, 4] = overlap / max(len(query_terms), 1)

            # Feature 6: PageType match
            pagetype = str(metadata.get("pageType", "")).lower()
            if pagetype in ("home", "programhome", "hub"):
                features[i, 5] = 1.0
            elif pagetype in ("person", "personnote"):
                features[i, 5] = 0.8
            elif pagetype == "daily":
                features[i, 5] = 0.5

            # Feature 7: Title match
            title = metadata.get("title", "").lower()
            title_terms = set(title.split())
            title_overlap = len(query_terms & title_terms)
            features[i, 6] = title_overlap / max(len(query_terms), 1)

        # Normalize each feature to [0, 1]
        for col in range(features.shape[1]):
            col_min, col_max = features[:, col].min(), features[:, col].max()
            if col_max - col_min > 0:
                features[:, col] = (features[:, col] - col_min) / (col_max - col_min)

        return features

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

        # Extract features
        features = self.extract_features(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores
        )

        # Apply logistic regression weights
        weights = np.array([
            self.FEATURE_WEIGHTS["dense"],
            self.FEATURE_WEIGHTS["bm25"],
            self.FEATURE_WEIGHTS["pagerank"],
            self.FEATURE_WEIGHTS["recency"],
            self.FEATURE_WEIGHTS["tag_overlap"],
            self.FEATURE_WEIGHTS["pagetype"],
            self.FEATURE_WEIGHTS["title_match"],
        ])

        # Compute logits and apply sigmoid
        logits = np.dot(features, weights) + self.BIAS
        scores = 1 / (1 + np.exp(-logits))  # Sigmoid

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
                "n_features": 7,
            },
        )


@register_strategy
class C37_LambdaMART(FusionStrategy):
    """C37: LambdaMART Reranking (Simulated)

    Two-stage approach:
      Stage 1: CombMNZ to get top 100 candidates
      Stage 2: Gradient-boosted tree reranking (simulated)

    Uses same 7 features as C36 but with nonlinear combination
    to simulate decision tree ensemble behavior.

    Note: In production, would use XGBoost/LightGBM with actual trained model.
    """

    STRATEGY_ID = "c37_lambdamart"
    CATEGORY = "learned_fusion"
    DESCRIPTION = "LambdaMART-style gradient boosting reranking"

    def __init__(
        self,
        top_k: int = 10,
        first_stage_k: int = 100,
        temporal_decay_days: float = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.first_stage_k = first_stage_k
        self.temporal_decay_days = temporal_decay_days

    def extract_features(
        self,
        query: str,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: List[Dict],
        bm25_scores: np.ndarray,
        indices: List[int],
    ) -> np.ndarray:
        """Extract features for candidate documents."""
        from datetime import datetime

        n_cands = len(indices)
        features = np.zeros((n_cands, 7))
        query_terms = set(query.lower().split())
        now = datetime.now()

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        for i, idx in enumerate(indices):
            chunk = chunks[idx]
            metadata = chunk.get("metadata", {})

            # Feature 1: Dense similarity
            c_emb = chunk_embeddings[idx]
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-8)
            features[i, 0] = np.dot(c_norm, q_norm)

            # Feature 2: BM25
            features[i, 1] = bm25_scores[idx]

            # Feature 3: PageRank / link density
            pagerank = metadata.get("pagerank", 0.0)
            link_count = metadata.get("link_count", 0)
            content = chunk.get("content", "")
            if link_count == 0 and content:
                link_count = content.count("[[")
            features[i, 2] = pagerank if pagerank > 0 else np.log1p(link_count) / 10.0

            # Feature 4: Temporal recency
            created = metadata.get("created") or metadata.get("dateLink")
            if created:
                try:
                    dt = datetime.strptime(str(created)[:10], "%Y-%m-%d")
                    days = (now - dt).days
                    features[i, 3] = np.exp(-days / self.temporal_decay_days)
                except (ValueError, TypeError):
                    pass

            # Feature 5: Tag overlap
            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            tag_terms = set()
            for tag in tags:
                tag_clean = tag.lower().replace("#", "")
                tag_terms.add(tag_clean)
                tag_terms.update(tag_clean.split("/"))
            overlap = len(query_terms & tag_terms)
            features[i, 4] = overlap / max(len(query_terms), 1)

            # Feature 6: PageType match
            pagetype = str(metadata.get("pageType", "")).lower()
            if pagetype in ("home", "programhome", "hub"):
                features[i, 5] = 1.0
            elif pagetype in ("person", "personnote"):
                features[i, 5] = 0.8
            elif pagetype == "daily":
                features[i, 5] = 0.5

            # Feature 7: Title match
            title = metadata.get("title", "").lower()
            title_terms = set(title.split())
            title_overlap = len(query_terms & title_terms)
            features[i, 6] = title_overlap / max(len(query_terms), 1)

        return features

    def simulate_gbdt_score(self, features: np.ndarray) -> np.ndarray:
        """Simulate gradient boosted decision tree scoring.

        This approximates what a trained LambdaMART model would do:
        - Nonlinear feature interactions
        - Threshold-based splits
        - Ensemble of weak learners

        In production, would use xgboost.Booster.predict()
        """
        n_docs = features.shape[0]
        scores = np.zeros(n_docs)

        # Simulate tree 1: Dense + BM25 interaction
        dense, bm25 = features[:, 0], features[:, 1]
        tree1 = np.where(
            dense > 0.5,
            0.3 + 0.2 * dense,  # High dense: use dense
            0.2 * dense + 0.3 * bm25  # Low dense: rely more on BM25
        )

        # Simulate tree 2: Recency gate
        recency = features[:, 3]
        tree2 = np.where(
            recency > 0.3,
            0.15 * recency,  # Recent: add recency bonus
            0.0  # Old: no bonus
        )

        # Simulate tree 3: PageType + PageRank interaction
        pagetype, pagerank = features[:, 5], features[:, 2]
        tree3 = np.where(
            pagetype > 0.7,
            0.1 + 0.1 * pagerank,  # Hub page: boost based on links
            0.05 * pagetype + 0.05 * pagerank
        )

        # Simulate tree 4: Title match boost
        title_match = features[:, 6]
        tree4 = np.where(
            title_match > 0.5,
            0.15 * title_match,  # Strong title match: big boost
            0.05 * title_match
        )

        # Simulate tree 5: Tag relevance
        tag_overlap = features[:, 4]
        tree5 = 0.05 * tag_overlap

        # Combine trees (LambdaMART uses additive model)
        scores = tree1 + tree2 + tree3 + tree4 + tree5

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
            raise ValueError("C37 requires query embeddings")
        if bm25_scores is None:
            raise ValueError("C37 requires BM25 scores")

        # Stage 1: CombMNZ to get candidates
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        c_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        dense_scores = np.dot(c_norms, q_norm)

        stage1_scores = self.combmnz([dense_scores, bm25_scores], normalize=True)
        stage1_indices = np.argsort(stage1_scores)[::-1][:self.first_stage_k].tolist()

        # Stage 2: GBDT reranking on candidates
        features = self.extract_features(
            query, query_embedding, chunk_embeddings, chunks, bm25_scores, stage1_indices
        )

        # Normalize features
        for col in range(features.shape[1]):
            col_min, col_max = features[:, col].min(), features[:, col].max()
            if col_max - col_min > 0:
                features[:, col] = (features[:, col] - col_min) / (col_max - col_min)

        # Apply simulated GBDT scoring
        stage2_scores = self.simulate_gbdt_score(features)

        # Get final ranking
        rerank_order = np.argsort(stage2_scores)[::-1][:self.top_k]
        final_indices = [stage1_indices[i] for i in rerank_order]
        final_scores = stage2_scores[rerank_order]

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
            metadata={
                "strategy": self.STRATEGY_ID,
                "stages": 2,
                "first_stage_k": self.first_stage_k,
            },
        )
