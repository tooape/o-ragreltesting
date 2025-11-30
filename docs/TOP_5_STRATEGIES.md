# Top 5 Ranking Strategies - O-RAG Relevance Testing

Validated benchmark results from 69 queries and 2,413 chunks using EmbeddingGemma-300M at 768 dimensions.

**Benchmark Date**: November 28, 2025  
**Last Validated**: November 30, 2025

---

## Leaderboard Summary

| Rank | Strategy                | MRR@5      | NDCG@5     | Category         |
|------|-------------------------|------------|------------|------------------|
| 1    | p1_recency_boost        | **0.6374** | 0.5988     | Proven Baselines |
| 2    | p2_recency_decay        | 0.6316     | 0.5887     | Proven Baselines |
| 3    | c37_lambdamart          | 0.6027     | 0.5517     | Learned Fusion   |
| 4    | c23_combined_metadata   | 0.5966     | 0.5503     | Metadata Enhanced|
| 5    | c36_logistic_regression | 0.5848     | 0.5494     | Learned Fusion   |

---

## 1. P1: Recency Boost

**Strategy ID**: `p1_recency_boost`  
**Category**: Proven Baselines  
**MRR@5**: 0.6374 | **NDCG@5**: 0.5988  
**Source**: `src/strategies/proven_baselines.py`

### Overview

Hybrid dense+BM25 fusion with exponential recency boost via linear interpolation. Best overall strategy for mixed query types.

### Parameters (768d Optimized)

```python
{
    "alpha": 0.624,           # 62% dense, 38% BM25
    "recency_weight": 0.340,  # 34% recency in final blend
    "decay_days": 10.25       # ~10 day exponential decay half-life
}
```

### Algorithm

1. Compute dense similarity: `dense_score = cosine(query_emb, chunk_emb)`
2. Normalize both dense and BM25 to [0, 1] via min-max scaling
3. Hybrid fusion: `hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm`
4. Recency score: `recency = exp(-days_old / decay_days)`
5. Final score: `score = (1 - recency_weight) * hybrid + recency_weight * recency`

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "p1_recency_boost",
    top_k=20,
    alpha=0.624,
    recency_weight=0.340,
    decay_days=10.25
)
```

### Full Metrics

| Metric | @5 | @10 | @20 |
|--------|----|----|-----|
| MRR | 0.6374 | 0.6407 | 0.6455 |
| NDCG | 0.5988 | 0.6249 | 0.6499 |
| Precision | 0.2174 | 0.1261 | 0.0732 |
| Recall | 0.6635 | 0.7237 | 0.7964 |
| MAP | 0.5394 | 0.5563 | 0.5661 |

---

## 2. P2: Recency Decay

**Strategy ID**: `p2_recency_decay`  
**Category**: Proven Baselines  
**MRR@5**: 0.6316 | **NDCG@5**: 0.5887  
**Source**: `src/strategies/proven_baselines.py`

### Overview

Same algorithm as P1 but with higher semantic weight and slower decay. Better for conceptual/research queries where recency matters less.

### Parameters

```python
{
    "alpha": 0.682,           # 68% dense (higher semantic weight)
    "recency_weight": 0.320,  # 32% recency (less aggressive)
    "decay_days": 11.71       # ~12 day half-life (slower decay)
}
```

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "p2_recency_decay",
    top_k=20,
    alpha=0.682,
    recency_weight=0.320,
    decay_days=11.71
)
```

### Full Metrics

| Metric | @5 | @10 | @20 |
|--------|----|----|-----|
| MRR | 0.6316 | 0.6404 | 0.6453 |
| NDCG | 0.5887 | 0.6222 | 0.6508 |
| Precision | 0.2087 | 0.1246 | 0.0739 |
| Recall | 0.6413 | 0.7188 | 0.8012 |
| MAP | 0.5344 | 0.5536 | 0.5654 |

---

## 3. C37: LambdaMART Reranking

**Strategy ID**: `c37_lambdamart`  
**Category**: Learned Fusion  
**MRR@5**: 0.6027 | **NDCG@5**: 0.5517  
**Source**: `src/strategies/learned_fusion.py`

### Overview

Two-stage approach: CombMNZ for candidate retrieval, then gradient-boosted tree reranking (simulated). Uses nonlinear feature interactions.

### Parameters

```python
{
    "first_stage_k": 100,       # Candidates from CombMNZ
    "temporal_decay_days": 30   # Temporal feature decay
}
```

### Features (7 total)

1. Dense similarity score
2. BM25 score
3. PageRank / link density
4. Temporal recency
5. Tag overlap
6. PageType match (hub/home pages)
7. Title match score

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "c37_lambdamart",
    top_k=20,
    first_stage_k=100,
    temporal_decay_days=30
)
```

### Full Metrics

| Metric | @5 | @10 | @20 |
|--------|----|----|-----|
| MRR | 0.6027 | 0.6065 | 0.6143 |
| NDCG | 0.5517 | 0.5837 | 0.6115 |
| Precision | 0.2000 | 0.1203 | 0.0710 |
| Recall | 0.6229 | 0.6966 | 0.7785 |
| MAP | 0.4843 | 0.5047 | 0.5144 |

---

## 4. C23: Combined Metadata

**Strategy ID**: `c23_combined_metadata`  
**Category**: Metadata Enhanced  
**MRR@5**: 0.5966 | **NDCG@5**: 0.5503  
**Source**: `src/strategies/metadata_enhanced.py`

### Overview

Multiplicative boosting using multiple metadata signals: pageType, tag overlap, link density, and temporal recency.

### Parameters

```python
{
    "pagetype_boost": 1.15,        # Hub/home pages: 1.0-1.15x
    "tag_boost_factor": 0.15,      # Tag overlap: 1.0-1.15x
    "link_boost_factor": 0.05,     # Link density: 1.0-1.05x
    "temporal_decay_days": 45,     # Slower decay for metadata strategy
    "max_combined_boost": 2.0      # Cap on total boost
}
```

### Algorithm

1. Base CombMNZ score from dense + BM25
2. Apply multiplicative boosts for metadata signals
3. Cap combined boost at 2.0x

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "c23_combined_metadata",
    top_k=20,
    pagetype_boost=1.15,
    tag_boost_factor=0.15,
    link_boost_factor=0.05,
    temporal_decay_days=45
)
```

### Full Metrics

| Metric | @5 | @10 | @20 |
|--------|----|----|-----|
| MRR | 0.5966 | 0.6141 | 0.6192 |
| NDCG | 0.5503 | 0.5922 | 0.6198 |
| Precision | 0.2000 | 0.1217 | 0.0703 |
| Recall | 0.6106 | 0.7089 | 0.7964 |
| MAP | 0.4912 | 0.5138 | 0.5237 |

---

## 5. C36: Logistic Regression Fusion

**Strategy ID**: `c36_logistic_regression`  
**Category**: Learned Fusion  
**MRR@5**: 0.5848 | **NDCG@5**: 0.5494  
**Source**: `src/strategies/learned_fusion.py`

### Overview

Linear combination of 7 features with pre-defined weights (simulated logistic regression - actual weights would be learned from labeled data).

### Feature Weights

```python
FEATURE_WEIGHTS = {
    "dense": 0.45,         # Dense similarity (dominant)
    "bm25": 0.30,          # BM25 lexical score
    "pagerank": 0.08,      # Graph centrality
    "recency": 0.07,       # Temporal signal
    "tag_overlap": 0.04,   # Tag matching
    "pagetype": 0.03,      # PageType indicator
    "title_match": 0.03,   # Title relevance
}
```

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "c36_logistic_regression",
    top_k=20
)
```

### Full Metrics

| Metric | @5 | @10 | @20 |
|--------|----|----|-----|
| MRR | 0.5848 | 0.6061 | 0.6107 |
| NDCG | 0.5494 | 0.5964 | 0.6170 |
| Precision | 0.1884 | 0.1232 | 0.0688 |
| Recall | 0.6029 | 0.7118 | 0.7751 |
| MAP | 0.5002 | 0.5235 | 0.5301 |

---

## Strategy Selection Guide

| Query Type | Recommended | Why |
|------------|-------------|-----|
| **General search** | P1 or P2 | Best overall MRR, balanced signals |
| **Status queries** ("latest on X") | P1 | Stronger recency signal |
| **Conceptual lookup** ("how does X work") | P2 | Higher semantic weight |
| **Navigational** ("find the X page") | C23 | PageType awareness |
| **Complex multi-signal** | C37 | Nonlinear feature interactions |

---

## Notable Exclusions

The following strategies from the original document did **not** make the validated top 5:

| Strategy | Actual Rank | Actual MRR@5 | Notes |
|----------|-------------|--------------|-------|
| p3_recency_tag_combo | #6 | 0.5758 | Tag boosting adds latency with marginal gain |
| c13_multi_signal_optimized | #11 | 0.5601 | Four-signal fusion underperformed |

---

## Production Recommendation

For O-RAG production deployment:

```python
# Recommended: P1 at 768d
strategy = create_strategy(
    "p1_recency_boost",
    top_k=20,
    alpha=0.624,
    recency_weight=0.340,
    decay_days=10.25
)

embedder = EmbeddingGemmaEmbedder(
    truncate_dim=768,
    use_bf16=True
)
```

**Alternative**: Use P2 for research-heavy queries where semantic matching should dominate over recency.
