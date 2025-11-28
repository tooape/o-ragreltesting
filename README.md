# O-RAG Relevance Testing

Evaluation framework for O-RAG (Obsidian RAG) ranking strategies with chunk-level ground truth.

## Overview

This repo provides:
1. **Hierarchical Markdown Chunker** - Chunks vault by heading hierarchy (H1-H6)
2. **Query/QRels Schema** - Chunk-level relevance judgments for ranking evaluation
3. **Evaluation Harness** - MRR, NDCG@k, Recall@k metrics with ranking strategy comparison

## Global Leaderboard

| Rank | Strategy | MRR@5 | NDCG@5 | E2E Latency (ms) | Summary |
|------|----------|-------|--------|------------------|---------|
| - | TBD | - | - | - | Phase 1 in progress |

*Last updated: TBD*

### Leaderboard Notes
- **MRR@5**: Mean Reciprocal Rank at position 5 (higher is better, max 1.0)
- **NDCG@5**: Normalized Discounted Cumulative Gain at position 5 (higher is better, max 1.0)
- **E2E Latency**: End-to-end query latency including retrieval and ranking
- Evaluated on full test query set with chunk-level relevance judgments

## Implementation Phases

### Phase 1: Strategy Baseline Evaluation
**Status**: In Progress

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
**Status**: Not Started

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
**Status**: Not Started

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
**Status**: Not Started

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
