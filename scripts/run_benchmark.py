#!/usr/bin/env python3
"""
O-RAG Benchmark Runner

Main script for running ranking strategy benchmarks on Lambda Cloud.

Usage:
    python run_benchmark.py --config config.json
    python run_benchmark.py --strategies c1_dense_only c7_rrf_basic --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedders import EmbeddingGemmaEmbedder, create_embedder
from bm25 import BM25Searcher, BM25Ranker
from evaluator import Evaluator, ResultsAggregator, EvaluationResult
from strategies import (
    ALL_STRATEGIES,
    CATEGORIES,
    create_strategy,
    get_all_strategy_ids,
    print_strategy_summary,
)
from utils import (
    GPUManager,
    ProgressTracker,
    create_phase_definitions,
    print_gpu_status,
)


class BenchmarkRunner:
    """Run O-RAG ranking benchmarks."""

    def __init__(
        self,
        chunks_file: Path,
        qrels_file: Path,
        output_dir: Path,
        embeddings_file: Optional[Path] = None,
        device: Optional[str] = None,
        truncate_dim: int = 768,
    ):
        """Initialize benchmark runner.

        Args:
            chunks_file: Path to vault_chunks.json
            qrels_file: Path to qrels.json
            output_dir: Directory for results
            embeddings_file: Pre-computed embeddings (optional)
            device: GPU device ('cuda:0', 'cpu', etc.)
            truncate_dim: Matryoshka dimension (256, 512, 768)
        """
        self.chunks_file = Path(chunks_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.embeddings_file = Path(embeddings_file) if embeddings_file else None
        self.truncate_dim = truncate_dim

        # Initialize GPU manager
        self.gpu_manager = GPUManager()
        self.device = device or self.gpu_manager.get_best_device()

        # Will be loaded
        self.chunks: List[Dict] = []
        self.chunk_ids: List[str] = []
        self.evaluator: Optional[Evaluator] = None
        self.embedder: Optional[EmbeddingGemmaEmbedder] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.bm25_ranker: Optional[BM25Ranker] = None
        self.queries: Dict[str, str] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}

        # Results
        self.aggregator = ResultsAggregator()

        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.tracker = ProgressTracker(self.progress_file)

    def setup(self):
        """Load data and initialize components."""
        print(f"\n{'='*60}")
        print("O-RAG Benchmark Setup")
        print(f"{'='*60}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load chunks
        print(f"\nLoading chunks from {self.chunks_file}...")
        with open(self.chunks_file) as f:
            data = json.load(f)

        # Handle different formats
        if "chunk_index" in data:
            self.chunks = data["chunk_index"]
        elif "chunks" in data:
            self.chunks = data["chunks"]
        else:
            self.chunks = data

        # Ensure each chunk has an ID
        for i, chunk in enumerate(self.chunks):
            if "id" not in chunk:
                chunk["id"] = chunk.get("notePath", f"chunk_{i}") + "::" + ">".join(chunk.get("headingPath", ["root"]))

        self.chunk_ids = [c["id"] for c in self.chunks]
        print(f"Loaded {len(self.chunks)} chunks")

        # Load qrels
        print(f"\nLoading qrels from {self.qrels_file}...")
        with open(self.qrels_file) as f:
            qrels_data = json.load(f)

        # Parse qrels and queries
        qrels = {}

        # Handle O-RAG qrels format (list with judgments)
        if "qrels" in qrels_data and isinstance(qrels_data["qrels"], list):
            for entry in qrels_data["qrels"]:
                qid = entry.get("id", "")
                judgments = entry.get("judgments", [])
                relevant_chunks = [j["chunk_id"] for j in judgments if "chunk_id" in j]
                if qid and relevant_chunks:
                    qrels[qid] = relevant_chunks
                    # Use prompt as query text
                    if "prompt" in entry:
                        self.queries[qid] = entry["prompt"]
                    elif "initial_search_query" in entry:
                        self.queries[qid] = entry["initial_search_query"]
        elif "queries" in qrels_data:
            for qid, q in qrels_data["queries"].items():
                qrels[qid] = q.get("relevant_chunks", q.get("relevant", []))
                if "query_text" in q:
                    self.queries[qid] = q["query_text"]
                elif "query" in q:
                    self.queries[qid] = q["query"]
        else:
            # Try to extract queries from simple format
            for qid, v in qrels_data.items():
                if isinstance(v, dict) and "query" in v:
                    self.queries[qid] = v["query"]
                    qrels[qid] = v.get("relevant", [])
                elif isinstance(v, list):
                    qrels[qid] = v

        self.evaluator = Evaluator(qrels)
        print(f"Loaded {len(qrels)} queries with relevance judgments")
        print(f"Query texts extracted: {len(self.queries)}")

        # Load or compute embeddings
        if self.embeddings_file and self.embeddings_file.exists():
            print(f"\nLoading pre-computed embeddings from {self.embeddings_file}...")
            self.chunk_embeddings = np.load(self.embeddings_file)
            print(f"Loaded embeddings: {self.chunk_embeddings.shape}")
        else:
            print(f"\nInitializing embedder on {self.device}...")
            self.embedder = create_embedder(
                model="gemma",
                device=self.device,
                truncate_dim=self.truncate_dim,
            )

            print("Computing chunk embeddings...")
            self.chunk_embeddings = self.embedder.encode_documents(
                self.chunks,
                batch_size=256,
                show_progress=True,
            )
            print(f"Computed embeddings: {self.chunk_embeddings.shape}")

            # Save embeddings
            if self.embeddings_file:
                np.save(self.embeddings_file, self.chunk_embeddings)
                print(f"Saved embeddings to {self.embeddings_file}")

        # Compute query embeddings
        print("\nComputing query embeddings...")
        if self.embedder is None:
            self.embedder = create_embedder(
                model="gemma",
                device=self.device,
                truncate_dim=self.truncate_dim,
            )

        self.query_embeddings = self.embedder.encode_queries(
            self.queries,
            task="retrieval",
        )
        print(f"Computed {len(self.query_embeddings)} query embeddings")

        # Initialize BM25
        print("\nBuilding BM25 index...")
        self.bm25_ranker = BM25Ranker()
        self.bm25_ranker.fit(self.chunks)

        print(f"\n{'='*60}")
        print("Setup complete!")
        print(f"{'='*60}\n")

    def run_strategy(
        self,
        strategy_id: str,
        top_k: int = 20,
    ) -> EvaluationResult:
        """Run a single strategy and evaluate.

        Args:
            strategy_id: Strategy identifier
            top_k: Number of results to retrieve

        Returns:
            EvaluationResult
        """
        strategy = create_strategy(strategy_id, top_k=top_k)
        config = strategy.get_config()

        print(f"\n  Running {strategy_id}: {config.description}")

        # Get BM25 scores if needed
        bm25_scores = None
        if strategy.requires_bm25:
            bm25_scores = {
                qid: self.bm25_ranker.get_scores(query)
                for qid, query in self.queries.items()
            }

        # Run strategy on all queries
        results = {}
        scores = {}
        start_time = time.time()

        for qid, query_text in self.queries.items():
            q_emb = self.query_embeddings.get(qid)
            q_bm25 = bm25_scores.get(qid) if bm25_scores else None

            result = strategy.rank(
                query=query_text,
                query_embedding=q_emb,
                chunk_embeddings=self.chunk_embeddings,
                chunks=self.chunks,
                bm25_scores=q_bm25,
            )

            results[qid] = result.ranked_chunk_ids
            scores[qid] = result.scores

        elapsed = time.time() - start_time

        # Evaluate
        eval_result = self.evaluator.evaluate_strategy(
            strategy_name=strategy_id,
            strategy_config=config.to_dict(),
            results=results,
            query_texts=self.queries,
            scores=scores,
            metadata={
                "elapsed_seconds": elapsed,
                "queries_per_second": len(self.queries) / elapsed if elapsed > 0 else 0,
            },
        )

        # Print key metrics
        metrics = eval_result.metrics
        print(f"    MRR@5: {metrics.get('mrr@5', 0):.4f}  "
              f"NDCG@5: {metrics.get('ndcg@5', 0):.4f}  "
              f"P@5: {metrics.get('p@5', 0):.4f}  "
              f"({elapsed:.1f}s)")

        return eval_result

    def run_benchmark(
        self,
        strategies: List[str] = None,
        top_k: int = 20,
        save_per_strategy: bool = True,
    ) -> Dict[str, Any]:
        """Run full benchmark.

        Args:
            strategies: List of strategy IDs (None = all)
            top_k: Number of results
            save_per_strategy: Save results after each strategy

        Returns:
            Summary results dict
        """
        if strategies is None:
            strategies = ALL_STRATEGIES

        print(f"\n{'='*60}")
        print(f"Running O-RAG Benchmark")
        print(f"Strategies: {len(strategies)}")
        print(f"Queries: {len(self.queries)}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")

        # Create benchmark in tracker
        phases = create_phase_definitions(strategies, "Strategy Evaluation")
        self.tracker.create_benchmark(
            benchmark_id=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            benchmark_name="O-RAG Strategy Benchmark",
            phases=phases,
            config={
                "strategies": strategies,
                "top_k": top_k,
                "device": self.device,
                "truncate_dim": self.truncate_dim,
            },
        )

        self.tracker.start_benchmark()
        self.tracker.start_phase("strategy_eval")

        # Run each strategy
        for i, strategy_id in enumerate(strategies):
            print(f"\n[{i+1}/{len(strategies)}] {strategy_id}")

            self.tracker.start_task("strategy_eval", strategy_id)

            try:
                result = self.run_strategy(strategy_id, top_k)
                self.aggregator.add_result(result)

                self.tracker.complete_task(
                    "strategy_eval",
                    strategy_id,
                    {"mrr@5": result.metrics.get("mrr@5", 0)},
                )

                if save_per_strategy:
                    result.to_json(self.output_dir / f"{strategy_id}_result.json")

            except Exception as e:
                print(f"    ERROR: {e}")
                self.tracker.fail_task("strategy_eval", strategy_id, str(e))

            # Clear GPU memory periodically
            if (i + 1) % 10 == 0:
                self.gpu_manager.clear_gpu_memory()

        self.tracker.complete_phase("strategy_eval")

        # Generate leaderboard
        leaderboard = self.aggregator.get_leaderboard("mrr@5")
        best = leaderboard[0] if leaderboard else None

        # Complete benchmark
        self.tracker.complete_benchmark({
            "best_strategy": best["strategy"] if best else None,
            "best_mrr@5": best["mrr@5"] if best else 0,
            "strategies_completed": len(self.aggregator.results),
        })

        # Save aggregated results
        self.aggregator.to_json(self.output_dir / "all_results.json")

        # Save leaderboard
        with open(self.output_dir / "leaderboard.json", 'w') as f:
            json.dump(leaderboard, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"\nTop 10 Strategies by MRR@5:")
        print("-" * 50)
        for entry in leaderboard[:10]:
            print(f"  {entry['rank']:2}. {entry['strategy']:30} MRR@5: {entry['mrr@5']:.4f}")

        print(f"\nResults saved to: {self.output_dir}")

        return {
            "leaderboard": leaderboard,
            "best": best,
            "num_strategies": len(strategies),
            "num_completed": len(self.aggregator.results),
        }


def main():
    parser = argparse.ArgumentParser(
        description="O-RAG Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all strategies
  python run_benchmark.py --chunks data/vault_chunks.json --qrels data/qrels.json

  # Run specific strategies
  python run_benchmark.py --strategies c1_dense_only c7_rrf_basic c13_dense_heavy

  # Quick test (first 5 strategies)
  python run_benchmark.py --quick

  # Run a category
  python run_benchmark.py --category simple_fusion
        """,
    )

    parser.add_argument("--chunks", type=Path, default=Path("data/vault_chunks.json"),
                        help="Path to vault chunks JSON")
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"),
                        help="Path to qrels JSON")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Pre-computed embeddings file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--dim", type=int, default=768, choices=[256, 512, 768],
                        help="Embedding dimension (Matryoshka)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of results to retrieve")

    # Strategy selection
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Specific strategies to run")
    parser.add_argument("--category", type=str, default=None,
                        choices=list(CATEGORIES.keys()),
                        help="Run all strategies in a category")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (first 5 strategies)")
    parser.add_argument("--list", action="store_true",
                        help="List available strategies and exit")

    args = parser.parse_args()

    # List strategies and exit
    if args.list:
        print_strategy_summary()
        return

    # Print GPU status
    print_gpu_status()

    # Determine strategies to run
    if args.quick:
        strategies = ALL_STRATEGIES[:5]
    elif args.category:
        strategies = CATEGORIES[args.category]
    elif args.strategies:
        strategies = args.strategies
    else:
        strategies = ALL_STRATEGIES

    # Validate strategies
    valid_strategies = set(get_all_strategy_ids())
    for s in strategies:
        if s not in valid_strategies:
            print(f"Error: Unknown strategy '{s}'")
            print(f"Use --list to see available strategies")
            sys.exit(1)

    # Run benchmark
    runner = BenchmarkRunner(
        chunks_file=args.chunks,
        qrels_file=args.qrels,
        output_dir=args.output,
        embeddings_file=args.embeddings,
        device=args.device,
        truncate_dim=args.dim,
    )

    runner.setup()
    results = runner.run_benchmark(strategies, top_k=args.top_k)

    print(f"\nBest strategy: {results['best']['strategy']}")
    print(f"Best MRR@5: {results['best']['mrr@5']:.4f}")


if __name__ == "__main__":
    main()
