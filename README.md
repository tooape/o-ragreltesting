# O-RAG Relevance Testing

Evaluation framework for O-RAG (Obsidian RAG) tools with chunk-level ground truth.

## Overview

This repo provides:
1. **Hierarchical Markdown Chunker** - Chunks vault by heading hierarchy (H1-H6)
2. **Query/QRels Schema** - Chunk-level relevance judgments for three tools
3. **Evaluation Harness** - MRR, NDCG@k, Recall@k metrics (TODO)

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
