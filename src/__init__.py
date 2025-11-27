"""
O-RAG: Obsidian RAG Testing Framework

Components:
- embedders: EmbeddingGemma with task-specific prompts
- bm25: BM25 keyword search
- chunker: Hierarchical markdown chunking
- evaluator: IR metrics (MRR, NDCG, P@k)
- strategies: 37 ranking strategies
- utils: GPU management and progress tracking
"""

__version__ = "0.1.0"
