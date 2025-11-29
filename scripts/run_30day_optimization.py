#!/usr/bin/env python3
"""
30-Day Decay Optimization for Top 5 Strategies

Fixes temporal_decay_days=30 and optimizes remaining hyperparameters.
This provides optimal configurations for longer time horizon use cases.

Usage:
    python run_30day_optimization.py --chunks data/chunks.json --qrels data/qrels.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedders import create_embedder
from bm25 import BM25Ranker
from evaluator import Evaluator, EvaluationResult
from strategies import create_strategy


# Top 5 strategies to optimize with 30-day decay
TOP5_STRATEGIES = [
    "c37_lambdamart",
    "p2_recency_decay",
    "p1_recency_boost",
    "c13_multi_signal_optimized",
    "p3_recency_tag_combo",
]

# Fixed decay for all strategies
FIXED_DECAY_DAYS = 30

# Parameter grids (excluding decay_days which is fixed at 30)
PARAM_GRIDS = {
    "c37_lambdamart": {
        "first_stage_k": [100, 150, 200, 250, 300],
        "temporal_decay_days": [30],  # Fixed
    },
    "p2_recency_decay": {
        "alpha": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        "recency_weight": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        "decay_days": [30],  # Fixed
    },
    "p1_recency_boost": {
        "alpha": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        "recency_weight": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        "decay_days": [30],  # Fixed
    },
    "c13_multi_signal_optimized": {
        "bm25_weight": [0.2, 0.25, 0.3, 0.35],
        "semantic_weight": [0.35, 0.4, 0.45, 0.5],
        "graph_weight": [0.0, 0.02, 0.05],
        "temporal_weight": [0.15, 0.2, 0.25, 0.3],
        "temporal_decay_days": [30],  # Fixed
    },
    "p3_recency_tag_combo": {
        "alpha": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
        "recency_weight": [0.2, 0.25, 0.3, 0.35, 0.4],
        "tag_weight": [0.05, 0.1, 0.15, 0.2],
        "decay_days": [30],  # Fixed
    },
}


class ThirtyDayOptimizer:
    """Optimize top 5 strategies with fixed 30-day decay."""

    def __init__(
        self,
        chunks_file: Path,
        qrels_file: Path,
        output_dir: Path,
        device: Optional[str] = None,
    ):
        self.chunks_file = Path(chunks_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.device = device

        self.chunks: List[Dict] = []
        self.evaluator: Optional[Evaluator] = None
        self.bm25_ranker: Optional[BM25Ranker] = None
        self.queries: Dict[str, str] = {}
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.query_embeddings: Dict[str, np.ndarray] = {}

    def load_data(self):
        """Load chunks and qrels."""
        print(f"\nLoading chunks from {self.chunks_file}...")
        with open(self.chunks_file) as f:
            data = json.load(f)

        if "chunk_index" in data:
            self.chunks = data["chunk_index"]
        elif "chunks" in data:
            self.chunks = data["chunks"]
        else:
            self.chunks = data

        for i, chunk in enumerate(self.chunks):
            if "id" not in chunk:
                chunk["id"] = chunk.get("notePath", f"chunk_{i}") + "::" + ">".join(chunk.get("headingPath", ["root"]))

        print(f"Loaded {len(self.chunks)} chunks")

        # Load qrels
        print(f"\nLoading qrels from {self.qrels_file}...")
        with open(self.qrels_file) as f:
            qrels_data = json.load(f)

        qrels = {}
        if "qrels" in qrels_data and isinstance(qrels_data["qrels"], list):
            for entry in qrels_data["qrels"]:
                qid = entry.get("id", "")
                judgments = entry.get("judgments", [])
                relevant_chunks = [j["chunk_id"] for j in judgments if "chunk_id" in j]
                if qid and relevant_chunks:
                    qrels[qid] = relevant_chunks
                    if "prompt" in entry:
                        self.queries[qid] = entry["prompt"]
                    elif "initial_search_query" in entry:
                        self.queries[qid] = entry["initial_search_query"]

        self.evaluator = Evaluator(qrels)
        print(f"Loaded {len(qrels)} queries")

        # Initialize BM25
        print("\nBuilding BM25 index...")
        self.bm25_ranker = BM25Ranker()
        self.bm25_ranker.fit(self.chunks)

    def compute_embeddings(self):
        """Compute embeddings once (768d for max quality)."""
        print("\nComputing embeddings (768d)...")
        embedder = create_embedder(
            model="gemma",
            device=self.device,
            truncate_dim=768,
        )

        self.chunk_embeddings = embedder.encode_documents(
            self.chunks,
            batch_size=32,  # Reduced for A10 GPU (22GB)
            show_progress=True,
        )

        self.query_embeddings = embedder.encode_queries(
            self.queries,
            task="retrieval",
        )

        print(f"Computed embeddings: chunks={self.chunk_embeddings.shape}, queries={len(self.query_embeddings)}")

    def setup(self):
        """Load data and compute embeddings."""
        print(f"\n{'='*60}")
        print("30-Day Decay Optimization Setup")
        print(f"{'='*60}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        self.compute_embeddings()

        print(f"\n{'='*60}")
        print("Setup complete!")
        print(f"{'='*60}\n")

    def evaluate_config(
        self,
        strategy_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate a single configuration."""
        strategy = create_strategy(strategy_id, top_k=20, **params)

        # Get BM25 scores
        bm25_scores = {
            qid: self.bm25_ranker.get_scores(query)
            for qid, query in self.queries.items()
        }

        # Run evaluation
        results = {}
        for qid, query_text in self.queries.items():
            q_emb = self.query_embeddings.get(qid)
            q_bm25 = bm25_scores.get(qid)

            result = strategy.rank(
                query=query_text,
                query_embedding=q_emb,
                chunk_embeddings=self.chunk_embeddings,
                chunks=self.chunks,
                bm25_scores=q_bm25,
            )
            results[qid] = result.ranked_chunk_ids

        # Compute metrics
        config = strategy.get_config()
        eval_result = self.evaluator.evaluate_strategy(
            strategy_name=strategy_id,
            strategy_config=config.to_dict(),
            results=results,
            query_texts=self.queries,
        )

        return {
            "mrr@5": eval_result.metrics.get("mrr@5", 0),
            "ndcg@5": eval_result.metrics.get("ndcg@5", 0),
            "recall@5": eval_result.metrics.get("recall@5", 0),
        }

    def optimize_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Run grid search for a strategy."""
        param_grid = PARAM_GRIDS[strategy_id]
        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]

        all_configs = list(product(*param_values))
        print(f"\n  Testing {len(all_configs)} configurations...")

        best_mrr = 0
        best_params = {}
        best_metrics = {}
        all_results = []

        for i, config_values in enumerate(all_configs):
            params = dict(zip(param_names, config_values))

            try:
                metrics = self.evaluate_config(strategy_id, params)
                mrr = metrics["mrr@5"]

                all_results.append({
                    "params": params,
                    **metrics,
                })

                if mrr > best_mrr:
                    best_mrr = mrr
                    best_params = params.copy()
                    best_metrics = metrics.copy()

                if (i + 1) % 10 == 0:
                    print(f"    {i+1}/{len(all_configs)} - Current best MRR@5: {best_mrr:.4f}")

            except Exception as e:
                print(f"    ERROR with {params}: {e}")

        return {
            "strategy": strategy_id,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "num_configs": len(all_configs),
            "all_results": all_results,
        }

    def run(self) -> Dict[str, Any]:
        """Run optimization for all top 5 strategies."""
        print(f"\n{'='*60}")
        print("30-Day Decay Optimization")
        print(f"Strategies: {len(TOP5_STRATEGIES)}")
        print(f"Fixed decay_days: {FIXED_DECAY_DAYS}")
        print(f"{'='*60}")

        results = []
        start_time = time.time()

        for strategy_id in TOP5_STRATEGIES:
            print(f"\n[{strategy_id}]")
            result = self.optimize_strategy(strategy_id)
            results.append(result)

            print(f"  Best MRR@5: {result['best_metrics']['mrr@5']:.4f}")
            print(f"  Best params: {result['best_params']}")

            # Save incremental results
            self._save_results(results)

        elapsed = time.time() - start_time

        # Final save
        self._save_results(results)
        self._create_summary(results)

        print(f"\n{'='*60}")
        print("30-DAY OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal time: {elapsed/60:.1f} minutes")

        print(f"\nResults Summary (30-day decay):")
        print("-" * 70)
        for r in results:
            print(f"  {r['strategy']:30} MRR@5: {r['best_metrics']['mrr@5']:.4f}")
            print(f"    Params: {r['best_params']}")

        print(f"\nResults saved to: {self.output_dir}")

        return {"results": results, "elapsed_seconds": elapsed}

    def _save_results(self, results: List[Dict]):
        """Save results to JSON."""
        with open(self.output_dir / "30day_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    def _create_summary(self, results: List[Dict]):
        """Create summary report."""
        summary = {
            "optimization": "30-day decay optimization",
            "fixed_decay_days": FIXED_DECAY_DAYS,
            "strategies": len(results),
            "results": [
                {
                    "strategy": r["strategy"],
                    "best_params": r["best_params"],
                    "mrr@5": r["best_metrics"]["mrr@5"],
                    "ndcg@5": r["best_metrics"]["ndcg@5"],
                }
                for r in results
            ],
        }

        with open(self.output_dir / "30day_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="30-Day Decay Optimization")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument("--output", type=Path, default=Path("results/30day"))
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    optimizer = ThirtyDayOptimizer(
        chunks_file=args.chunks,
        qrels_file=args.qrels,
        output_dir=args.output,
        device=args.device,
    )

    optimizer.setup()
    optimizer.run()


if __name__ == "__main__":
    main()
