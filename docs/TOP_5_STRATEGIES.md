# Top 5 Ranking Strategies - O-RAG Relevance Testing

This document provides detailed documentation for the top 5 ranking strategies identified through comprehensive evaluation (69 queries, 2,413 chunks, EmbeddingGemma-300M).

*Last Updated: November 29, 2025*

---

## Leaderboard Summary

| Rank | Strategy | MRR@5 | NDCG@5 | Category |
|------|----------|-------|--------|----------|
| 1 | c37_lambdamart | **0.6575** | 0.5977 | Learned Fusion |
| 2 | p2_recency_decay | 0.6418 | 0.6009 | Proven Baselines |
| 3 | p1_recency_boost | 0.6418 | 0.6009 | Proven Baselines |
| 4 | c13_multi_signal_optimized | 0.6396 | - | Custom Weights |
| 5 | p3_recency_tag_combo | 0.6370 | - | Proven Baselines |

---

## 1. C37: LambdaMART Reranking

**Strategy ID**: `c37_lambdamart`
**Category**: Learned Fusion
**MRR@5**: 0.6575 | **NDCG@5**: 0.5977
**Source**: `src/strategies/learned_fusion.py:196-415`

### Overview

A two-stage learning-to-rank approach that simulates gradient-boosted decision tree (GBDT) reranking. Uses CombMNZ for initial candidate retrieval, then applies nonlinear feature interactions via simulated decision tree ensembles.

### Optimal Hyperparameters

```python
{
    "first_stage_k": 200,        # Candidates from CombMNZ
    "temporal_decay_days": 13    # Exponential decay half-life
}
```

### Algorithm

**Stage 1: Candidate Retrieval (CombMNZ)**
1. Compute dense similarity: `dense_score = cosine(query_emb, chunk_emb)`
2. Normalize both dense and BM25 scores to [0, 1]
3. Apply CombMNZ: `score = (dense_norm + bm25_norm) * count_nonzero`
4. Select top `first_stage_k` candidates (200)

**Stage 2: GBDT Reranking**
Extract 7 features for each candidate:
1. **Dense similarity** - Cosine similarity with query embedding
2. **BM25 score** - Lexical relevance
3. **PageRank/link density** - Graph centrality signal
4. **Temporal recency** - `exp(-days / decay_days)`
5. **Tag overlap** - Query term overlap with chunk tags
6. **PageType match** - Hub/home pages get boost
7. **Title match** - Query overlap with chunk title

Apply simulated GBDT scoring (5 trees with nonlinear interactions):
- Tree 1: Dense + BM25 interaction (high dense → use dense; low dense → rely on BM25)
- Tree 2: Recency gate (recent → add bonus; old → no bonus)
- Tree 3: PageType + PageRank interaction
- Tree 4: Title match boost
- Tree 5: Tag relevance

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "c37_lambdamart",
    top_k=20,
    first_stage_k=200,
    temporal_decay_days=13
)

result = strategy.rank(
    query="Intent AI 2026 strategy",
    query_embedding=q_emb,
    chunk_embeddings=chunk_embs,
    chunks=chunks,
    bm25_scores=bm25_scores
)

# result.ranked_chunk_ids contains ordered chunk IDs
# result.scores contains {chunk_id: score} mapping
```

### Why It Works

- **Two-stage design**: CombMNZ provides diverse candidates; GBDT refines ranking
- **Feature interactions**: Nonlinear combinations capture complex relevance patterns
- **Recency gating**: Only recent documents get temporal boost (avoids old document penalty)
- **PageType awareness**: Hub pages naturally rank higher for navigational queries

### Tradeoffs

- **Pros**: Highest MRR@5, handles diverse query types well
- **Cons**: More complex than linear fusion, 6.2% quality drop at 512d embeddings

---

## 2. P2: Recency Decay

**Strategy ID**: `p2_recency_decay`
**Category**: Proven Baselines
**MRR@5**: 0.6418 | **NDCG@5**: 0.6009
**Source**: `src/strategies/proven_baselines.py:183-279`

### Overview

Linear interpolation between hybrid dense+BM25 scores and exponential recency decay. Optimized for higher semantic weight and slower decay than P1.

### Optimal Hyperparameters

```python
{
    "alpha": 0.65,           # 65% dense, 35% BM25
    "recency_weight": 0.35,  # 35% recency in final blend
    "decay_days": 10         # Exponential decay half-life
}
```

### Algorithm

1. **Dense similarity**: `dense_score = cosine(query_emb, chunk_emb)`
2. **Normalize**: Min-max scale both dense and BM25 to [0, 1]
3. **Hybrid fusion**: `hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm`
4. **Recency score**: `recency = exp(-days_old / decay_days)`
5. **Final score**: `score = (1 - recency_weight) * hybrid + recency_weight * recency`

**Key insight**: Uses LINEAR INTERPOLATION, not multiplicative boosting.

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "p2_recency_decay",
    top_k=20,
    alpha=0.65,
    recency_weight=0.35,
    decay_days=10
)

result = strategy.rank(
    query="recent updates on CKG",
    query_embedding=q_emb,
    chunk_embeddings=chunk_embs,
    chunks=chunks,
    bm25_scores=bm25_scores
)
```

### Why It Works

- **Linear interpolation**: Smooth blending avoids extreme score swings
- **Balanced signals**: 65% semantic ensures conceptual matching dominates
- **10-day decay**: Recent notes surface for status queries, but relevance still matters

### Tradeoffs

- **Pros**: Simple, interpretable, only 1.4% quality drop at 512d (best dimension robustness)
- **Cons**: Slightly lower MRR than LambdaMART

### Production Recommendation

**Best choice for production** when paired with 512d Matryoshka embeddings:
- Only 1.4% MRR drop (0.6329 at 512d vs 0.6418 at 768d)
- 33% storage reduction (4.9MB vs 7.35MB)
- Simpler deployment than LambdaMART

---

## 3. P1: Recency Boost

**Strategy ID**: `p1_recency_boost`
**Category**: Proven Baselines
**MRR@5**: 0.6418 | **NDCG@5**: 0.6009
**Source**: `src/strategies/proven_baselines.py:55-180`

### Overview

Same algorithm as P2 but with dimension-aware configuration. Includes precomputed optimal configs for 768d and 512d embeddings.

### Optimal Hyperparameters

```python
# 768d configuration (default)
{
    "alpha": 0.65,           # 65% dense
    "recency_weight": 0.35,  # 35% recency
    "decay_days": 10
}

# 512d configuration (auto-selected when embedding_dim=512)
{
    "alpha": 0.456,          # 46% dense (more BM25 reliance)
    "recency_weight": 0.355,
    "decay_days": 12.0       # Slower decay
}
```

### Algorithm

Identical to P2. The differentiation is in configuration management:

```python
# Auto-selects config based on embedding dimension
strategy = create_strategy(
    "p1_recency_boost",
    top_k=20,
    embedding_dim=512  # Auto-selects 512d-optimized config
)
```

### Implementation

```python
from strategies import create_strategy

# Manual config
strategy = create_strategy(
    "p1_recency_boost",
    top_k=20,
    alpha=0.65,
    recency_weight=0.35,
    decay_days=10
)

# Or auto-config based on embedding dimension
strategy = create_strategy(
    "p1_recency_boost",
    top_k=20,
    embedding_dim=768  # Uses OPTIMAL_CONFIGS[768]
)
```

### Why P1 vs P2?

P1 and P2 converged to identical optimal parameters after fine-tuning. P1 offers dimension-aware auto-configuration, while P2 is a simpler explicit-config version.

---

## 4. C13: Multi-Signal Optimized

**Strategy ID**: `c13_multi_signal_optimized`
**Category**: Custom Weights
**MRR@5**: 0.6396
**Source**: `src/strategies/custom_weights.py:206-307`

### Overview

Four-signal weighted fusion combining BM25, semantic, graph, and temporal signals with optimized weights.

### Optimal Hyperparameters

```python
{
    "bm25_weight": 0.25,         # 25% BM25
    "semantic_weight": 0.45,     # 45% Dense (dominant)
    "graph_weight": 0.02,        # 2% PageRank/links
    "temporal_weight": 0.20,     # 20% Recency
    "temporal_decay_days": 15    # 15-day half-life
}
```

### Algorithm

1. **Compute signals**:
   - Dense: `cosine(query_emb, chunk_emb)`
   - BM25: Lexical score
   - Graph: `pagerank if pagerank > 0 else log1p(link_count) / 10`
   - Temporal: `exp(-days / decay_days)`

2. **Normalize**: Min-max scale each signal to [0, 1]

3. **Weighted fusion**:
   ```
   score = 0.25 * bm25 + 0.45 * dense + 0.02 * graph + 0.20 * temporal
   ```

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "c13_multi_signal_optimized",
    top_k=20,
    bm25_weight=0.25,
    semantic_weight=0.45,
    graph_weight=0.02,
    temporal_weight=0.20,
    temporal_decay_days=15
)

result = strategy.rank(
    query="CKG architecture overview",
    query_embedding=q_emb,
    chunk_embeddings=chunk_embs,
    chunks=chunks,
    bm25_scores=bm25_scores
)
```

### Why It Works

- **Semantic dominance (45%)**: Conceptual matching is primary signal
- **Low graph weight (2%)**: PageRank helps but shouldn't dominate
- **Moderate recency (20%)**: Less aggressive than P1/P2 for evergreen queries

### Tradeoffs

- **Pros**: Explicit control over each signal weight
- **Cons**: More hyperparameters to tune, slightly lower MRR than P1/P2

---

## 5. P3: Recency + Tag Combo

**Strategy ID**: `p3_recency_tag_combo`
**Category**: Proven Baselines
**MRR@5**: 0.6370
**Source**: `src/strategies/proven_baselines.py:282-433`

### Overview

Extends P1/P2 with tag-based relevance boosting. Documents with tags matching query terms receive a multiplicative boost.

### Optimal Hyperparameters

```python
{
    "alpha": 0.65,           # 65% dense
    "recency_weight": 0.35,  # 35% recency
    "tag_weight": 0.10,      # 10% tag boost factor
    "decay_days": 10
}
```

### Algorithm

1. **Base scoring** (same as P1/P2):
   ```
   hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm
   recency_boosted = (1 - recency_weight) * hybrid + recency_weight * recency
   ```

2. **Tag overlap scoring**:
   - Extract query terms: `{"intent", "ai", "strategy"}`
   - Extract chunk tags: `{"#work", "#intent-ai"}` → `{"work", "intent-ai", "intent", "ai"}`
   - Compute overlap: `overlap = len(query_terms & tag_terms) / len(query_terms)`

3. **Final score** (multiplicative boost):
   ```
   score = recency_boosted * (1 + tag_weight * tag_overlap)
   ```

### Implementation

```python
from strategies import create_strategy

strategy = create_strategy(
    "p3_recency_tag_combo",
    top_k=20,
    alpha=0.65,
    recency_weight=0.35,
    tag_weight=0.10,
    decay_days=10
)

result = strategy.rank(
    query="Intent AI meetings",
    query_embedding=q_emb,
    chunk_embeddings=chunk_embs,
    chunks=chunks,
    bm25_scores=bm25_scores
)
```

### Why It Works

- **Tag signal**: Explicit metadata matching complements semantic similarity
- **Multiplicative boost**: Tags enhance existing relevance rather than override it
- **Handles structured content**: Works well when chunks have consistent tagging

### Tradeoffs

- **Pros**: Leverages vault metadata, good for tagged content
- **Cons**: +3ms latency, requires consistent tagging, marginal gain over P2

---

## 30-Day Decay Configuration (Long Time Horizon)

For use cases requiring a longer time horizon (e.g., evergreen content, research queries, historical lookups), the strategies were re-optimized with `decay_days=30` fixed.

### 30-Day Leaderboard

| Rank | Strategy | MRR@5 (30d) | MRR@5 (10d) | Change | Best Params (30d) |
|------|----------|-------------|-------------|--------|-------------------|
| 1 | **c13_multi_signal_optimized** | **0.6242** | 0.6396 | -2.4% | BM25=0.35, semantic=0.40, graph=0.0, temporal=0.20 |
| 2 | p2_recency_decay | 0.6220 | 0.6418 | -3.1% | alpha=0.55, recency=0.25 |
| 3 | p1_recency_boost | 0.6220 | 0.6418 | -3.1% | alpha=0.55, recency=0.25 |
| 4 | c37_lambdamart | 0.6193 | 0.6575 | -5.8% | first_stage_k=250 |
| 5 | p3_recency_tag_combo | 0.6147 | 0.6370 | -3.5% | alpha=0.55, recency=0.25, tag=0.05 |

### Key Observations

1. **C13 becomes the winner**: With 30-day decay, c13_multi_signal_optimized surpasses the recency-focused strategies
2. **LambdaMART drops most**: The largest degradation (-5.8%) since its recency gating is tuned for short time horizons
3. **Alpha shifts from 0.65 → 0.55**: Lower dense weight compensates for weaker recency signal
4. **Recency weight drops from 0.35 → 0.25**: Less reliance on temporal signal with longer decay
5. **Graph weight goes to 0**: With longer horizons, graph centrality adds noise

### 30-Day Optimal Configurations

**C13 Multi-Signal (Best for 30-day)**:
```python
strategy = create_strategy(
    "c13_multi_signal_optimized",
    top_k=20,
    bm25_weight=0.35,
    semantic_weight=0.40,
    graph_weight=0.0,        # Disabled for 30-day
    temporal_weight=0.20,
    temporal_decay_days=30
)
```

**P2 Recency Decay (30-day config)**:
```python
strategy = create_strategy(
    "p2_recency_decay",
    top_k=20,
    alpha=0.55,              # Reduced from 0.65
    recency_weight=0.25,     # Reduced from 0.35
    decay_days=30
)
```

**C37 LambdaMART (30-day config)**:
```python
strategy = create_strategy(
    "c37_lambdamart",
    top_k=20,
    first_stage_k=250,       # Increased from 200
    temporal_decay_days=30
)
```

### When to Use 30-Day Configuration

| Use Case | Recommended Config |
|----------|-------------------|
| Daily work notes, status updates | **10-day** (default) |
| Research queries, conceptual lookups | **30-day** |
| Historical searches ("what did we decide about X") | **30-day** |
| Evergreen documentation | **30-day** |
| Recent project context | **10-day** |

---

## Strategy Selection Guide

| Query Type | Recommended Strategy | Why |
|------------|---------------------|-----|
| **General search** | p2_recency_decay | Best balance of quality and simplicity |
| **Status queries** ("what's the latest on X") | p1_recency_boost or p2 | Strong recency signal |
| **Conceptual lookup** ("how does X work") | c13_multi_signal | Semantic-dominant (45%) |
| **Navigational** ("find the CKG home page") | c37_lambdamart | PageType awareness |
| **Tagged content** | p3_recency_tag_combo | Explicit tag matching |

## Production Recommendation

For O-RAG production deployment:

```python
# Recommended: p2_recency_decay + 512d Matryoshka
strategy = create_strategy(
    "p2_recency_decay",
    top_k=20,
    alpha=0.65,
    recency_weight=0.35,
    decay_days=10
)

embedder = EmbeddingGemmaEmbedder(
    truncate_dim=512,  # 33% storage reduction
    use_fp16=True      # BF16 inference
)
```

**Why**:
- Only 1.4% MRR drop at 512d (best dimension robustness)
- Simple, interpretable, no complex reranking
- 33% storage reduction (4.9MB vs 7.35MB for 768d)
- Consistent performance across query types
