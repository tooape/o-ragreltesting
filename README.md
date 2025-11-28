# O-RAG Relevance Testing

Evaluation framework for O-RAG (Obsidian RAG) ranking strategies with chunk-level ground truth.

## Overview

This repo provides:
1. **Hierarchical Markdown Chunker** - Chunks vault by heading hierarchy (H1-H6)
2. **Query/QRels Schema** - Chunk-level relevance judgments for ranking evaluation
3. **Evaluation Harness** - MRR, NDCG@k, Recall@k metrics with ranking strategy comparison

## Global Leaderboard

| Rank | Strategy | MRR@5 | NDCG@5 | Best Params |
|------|----------|-------|--------|-------------|
| 1 | **c37_lambdamart** | **0.6575** | 0.5977 | first_stage_k=200, decay=13d |
| 2 | p2_recency_decay | 0.6418 | 0.6009 | alpha=0.65, recency=0.35, decay=10d |
| 3 | p1_recency_boost | 0.6418 | 0.6009 | alpha=0.65, recency=0.35, decay=10d |
| 4 | c13_multi_signal_optimized | 0.6396 | - | BM25=0.25, semantic=0.45, graph=0.02, temporal=0.2, decay=15d |
| 5 | p3_recency_tag_combo | 0.6370 | - | alpha=0.65, recency=0.35, tag=0.1, decay=10d |
| 6 | c17_linear_decay_90d | 0.6326 | - | window=30d, min=0.07, temporal=0.4 |
| 7 | c23_combined_metadata | 0.6198 | - | pagetype=1.2, tag=0.05, link=0.04, decay=35d |
| 8 | c15_exp_decay_30d | 0.6118 | - | decay=20d, temporal=0.5 |
| 9 | c19_sigmoid_boost | 0.6063 | - | midpoint=20d, steepness=13, temporal=0.4 |
| 10 | c36_logistic_regression | 0.5857 | - | decay=38d |

*Last updated: November 28, 2025*

### Dimension Reduction Analysis (Top 3 Strategies)

| Strategy | 768d MRR@5 | 512d MRR@5 (Δ%) | 256d MRR@5 (Δ%) | 256d Storage |
|----------|------------|-----------------|-----------------|--------------|
| c37_lambdamart | 0.6575 | 0.6164 (-6.2%) | 0.5913 (-10.1%) | 2.5MB |
| p2_recency_decay | 0.6418 | 0.6329 (-1.4%) | 0.6075 (-5.3%) | 2.5MB |
| p1_recency_boost | 0.6418 | 0.6329 (-1.4%) | 0.6075 (-5.3%) | 2.5MB |

**Recommendation**: For production, use **p2_recency_decay** with **512d Matryoshka** embeddings - only 1.4% MRR drop with 33% storage reduction.

### Leaderboard Notes
- **MRR@5**: Mean Reciprocal Rank at position 5 (higher is better, max 1.0)
- **NDCG@5**: Normalized Discounted Cumulative Gain at position 5 (higher is better, max 1.0)
- **E2E Latency**: End-to-end query latency including retrieval and ranking
- Evaluated on full test query set with chunk-level relevance judgments

## Implementation Phases

### Phase 1: Strategy Baseline Evaluation
**Status**: Complete

**Objective**: Establish baseline performance for all 40 ranking strategies using optimal model configuration.

**Configuration**:
- Model: EmbeddingGemma (768 dimensions, unquantized)
- All strategies evaluated with default hyperparameters
- Metrics: MRR@5, NDCG@5, Recall@5, Precision@5, end-to-end latency

**Deliverables**:
1. Performance metrics for all 40 strategies
2. Top 30 strategies identified for coarse optimization
3. Leaderboard updated with baseline results

### Phase 2: Coarse Hyperparameter Optimization
**Status**: Complete

**Objective**: Coarse grid search optimization for the top 30 strategies from Phase 1.

**Configuration**:
- Model: EmbeddingGemma (768 dimensions, unquantized)
- Coarse grid search over key hyperparameters per strategy
- Same metrics as Phase 1

**Deliverables**:
1. Coarse-optimized hyperparameters for each of the 30 strategies
2. Performance improvement deltas vs. baseline
3. Top 10 strategies selected for fine optimization
4. Leaderboard updated with coarse-optimized results

### Phase 3: Fine Hyperparameter Optimization
**Status**: Complete

**Objective**: Fine-grained grid search optimization for the top 10 strategies from Phase 2.

**Configuration**:
- Model: EmbeddingGemma (768 dimensions, unquantized)
- Fine grid search around coarse optima
- Same metrics as Phase 1 & 2

**Deliverables**:
1. Fine-tuned hyperparameters for each of the 10 strategies
2. Performance improvement deltas vs. coarse optimization
3. Top 3 strategies selected for model optimization phase
4. Leaderboard updated with fine-optimized results

### Phase 4: Model Optimization
**Status**: Complete

**Objective**: Evaluate quality/performance tradeoffs with reduced dimensionality and quantization.

**Configuration**:
- Models evaluated:
  - EmbeddingGemma (768d, unquantized) - baseline
  - EmbeddingGemma (512d, unquantized)
  - EmbeddingGemma (256d, unquantized)
  - EmbeddingGemma QAT (768d, quantized)
  - EmbeddingGemma QAT (512d, quantized)
  - EmbeddingGemma QAT (256d, quantized)
- Top 3 strategies from Phase 3 with fine-tuned hyperparameters
- Same metrics as previous phases

**Deliverables**:
1. Quality degradation analysis for dimensionality reduction
2. Quality degradation analysis for quantization (QAT)
3. Latency improvements from reduced dimensions and quantization
4. Final production recommendation (strategy + model config)
5. Leaderboard updated with final optimized configurations

## O-RAG Tools Being Evaluated

| Tool | Purpose | Query Type |
|------|---------|------------|
| `/smart-search` | Semantic discovery | Conceptual, multi-term |
| `/simple-search` | Lexical lookup | Exact names, tags |
| `/local-graph` | Relationship browsing | Seed note + hops |

## Chunking Strategy

Chunks are created at heading boundaries with hierarchical context:

```
title: "CKG 4.0 > Key Initiatives > Multi-Language Support"
content: "- Expanded language coverage..."
notePath: "Notes/Programs/Intent/CKG 4.0.md"
headingPath: ["Key Initiatives", "Multi-Language Support"]
```

Chunk IDs use format: `{notePath}::{headingPath joined by >}`

## Query Format

```json
{
  "id": "smart-001",
  "tool": "smart-search",
  "query": "Intent AI 2026 strategy",
  "expected_chunks": [
    {
      "chunk_id": "Notes/.../November 20, 2025.md::Meetings>Intent Strategy...",
      "relevance": 3,
      "reason": "Direct match - strategy whitepaper meeting"
    }
  ]
}
```

Relevance grades:
- **3**: Highly relevant - directly answers query
- **2**: Relevant - provides useful context
- **1**: Marginally relevant - tangentially related
- **0**: Not relevant

## Usage

### Chunk the vault

```bash
python -m src.chunker /path/to/vault -o data/vault_chunks.json
```

### Run evaluation (TODO)

```bash
python -m src.evaluate --queries data/queries.json --chunks data/vault_chunks.json
```

## Directory Structure

```
├── data/
│   ├── vault_chunks.json    # Chunked vault (not committed)
│   └── queries.json         # Test queries with ground truth
├── schemas/
│   ├── query_schema.json    # Smart/simple search query format
│   └── local_graph_query_schema.json
├── src/
│   └── chunker.py           # Hierarchical markdown chunker
└── requirements.txt
```

## Metrics

- **MRR** (Mean Reciprocal Rank) - Where does first relevant chunk appear?
- **NDCG@k** - Quality of top-k ranking considering relevance grades
- **Recall@k** - What fraction of relevant chunks appear in top-k?
- **Precision@k** - What fraction of top-k are relevant?

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/
```
