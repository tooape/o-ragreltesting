#!/usr/bin/env python3
"""
Phase 2: Coarse Hyperparameter Optimization

Runs coarse grid search on the top 30 strategies from Phase 1.
Each strategy has category-specific hyperparameter grids.

Usage:
    python run_phase2.py --chunks data/vault_chunks.json --qrels data/qrels.json
    python run_phase2.py --strategy p1_recency_boost  # Single strategy
"""

import argparse
import json
import sys
import time
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedders import create_embedder
from bm25 import BM25Ranker
from evaluator import Evaluator, EvaluationResult
from strategies import create_strategy, get_all_strategy_ids


# =============================================================================
# HYPERPARAMETER GRIDS BY STRATEGY
# =============================================================================

# Coarse grid definitions - wide exploration
HYPERPARAMETER_GRIDS = {
    # -------------------------------------------------------------------------
    # Proven Baselines (P1-P3) - Key params: alpha, recency_weight, decay_days
    # -------------------------------------------------------------------------
    "p1_recency_boost": {
        "alpha": [0.4, 0.5, 0.6, 0.7, 0.8],           # Dense vs BM25 balance
        "recency_weight": [0.2, 0.3, 0.4, 0.5],       # Recency contribution
        "decay_days": [7, 14, 21, 30],                 # Temporal decay
    },
    "p2_recency_decay": {
        "alpha": [0.5, 0.6, 0.7, 0.8],
        "recency_weight": [0.2, 0.3, 0.4],
        "decay_days": [10, 15, 20, 30],
    },
    "p3_recency_tag_combo": {
        "alpha": [0.5, 0.6, 0.7],
        "recency_weight": [0.15, 0.25, 0.35],
        "tag_weight": [0.15, 0.25, 0.35],
        "decay_days": [10, 15, 20],
    },

    # -------------------------------------------------------------------------
    # Learned Fusion (C36-C37)
    # -------------------------------------------------------------------------
    "c36_logistic_regression": {
        "temporal_decay_days": [15, 30, 45, 60],
    },
    "c37_lambdamart": {
        "first_stage_k": [50, 100, 150, 200],
        "temporal_decay_days": [15, 30, 45, 60],
    },

    # -------------------------------------------------------------------------
    # Metadata Enhanced (C20-C23)
    # -------------------------------------------------------------------------
    "c20_pagetype_aware": {
        "home_boost": [1.05, 1.15, 1.25, 1.35],
        "person_boost": [1.0, 1.1, 1.2],
        "daily_boost": [1.0, 1.05, 1.1],
    },
    "c21_tag_overlap": {
        "tag_boost_factor": [0.1, 0.2, 0.3, 0.4],
    },
    "c22_link_density": {
        "link_boost_factor": [0.03, 0.05, 0.08, 0.1],
        "max_boost": [1.2, 1.3, 1.5],
    },
    "c23_combined_metadata": {
        "pagetype_boost": [1.1, 1.2, 1.3],
        "tag_boost_factor": [0.1, 0.2, 0.3],
        "link_boost_factor": [0.03, 0.05, 0.08],
        "temporal_decay_days": [30, 45, 60],
    },

    # -------------------------------------------------------------------------
    # Temporal Boosting (C15-C19)
    # -------------------------------------------------------------------------
    "c15_exp_decay_30d": {
        "decay_days": [15, 30, 45, 60],
        "temporal_weight": [0.1, 0.2, 0.3, 0.4],
    },
    "c16_exp_decay_60d": {
        "decay_days": [30, 60, 90, 120],
        "temporal_weight": [0.1, 0.2, 0.3],
    },
    "c17_linear_decay_90d": {
        "window_days": [45, 60, 90, 120],
        "min_score": [0.05, 0.1, 0.2],
        "temporal_weight": [0.1, 0.2, 0.3],
    },
    "c18_step_function": {
        "boost_7d": [1.3, 1.5, 1.7, 2.0],
        "boost_30d": [1.1, 1.2, 1.3],
        "boost_90d": [0.9, 1.0, 1.1],
        "boost_old": [0.8, 0.9, 1.0],
    },
    "c19_sigmoid_boost": {
        "midpoint_days": [30, 45, 60, 90],
        "steepness": [10, 15, 20, 30],
        "temporal_weight": [0.1, 0.2, 0.3],
    },

    # -------------------------------------------------------------------------
    # Custom Weights (C10-C14)
    # -------------------------------------------------------------------------
    "c10_bm25_heavy": {
        "bm25_weight": [0.6, 0.7, 0.8, 0.9],
    },
    "c11_semantic_heavy": {
        "bm25_weight": [0.2, 0.3, 0.4],  # Semantic = 1 - bm25
    },
    "c12_equal_weight": {
        # Fixed 50/50 - no params
    },
    "c13_multi_signal_optimized": {
        "bm25_weight": [0.35, 0.45, 0.55],
        "semantic_weight": [0.25, 0.35, 0.45],
        "graph_weight": [0.05, 0.1, 0.15],
        "temporal_weight": [0.05, 0.1, 0.15],
        "temporal_decay_days": [20, 30, 45],
    },
    "c14_multi_signal_bm25_dominant": {
        "bm25_weight": [0.5, 0.6, 0.7],
        "semantic_weight": [0.2, 0.25, 0.3],
        "graph_weight": [0.05, 0.1, 0.15],
        "temporal_weight": [0.03, 0.05, 0.1],
    },

    # -------------------------------------------------------------------------
    # Fusion Methods (C1-C5) - Limited params
    # -------------------------------------------------------------------------
    "c1_combmnz_basic": {},  # No tunable params in current impl
    "c2_combmnz_graph": {},
    "c4_combsum_basic": {},
    "c5_combsum_all": {},

    # -------------------------------------------------------------------------
    # Normalization (C6, C9)
    # -------------------------------------------------------------------------
    "c9_logscale_weighted": {},  # Fixed normalization

    # -------------------------------------------------------------------------
    # Two-Stage (C24-C27)
    # -------------------------------------------------------------------------
    "c24_bm25_then_combmnz": {
        "first_stage_k": [50, 100, 150, 200],
    },
    "c25_semantic_then_combmnz": {
        "first_stage_k": [50, 100, 150, 200],
    },
    "c26_union_then_metadata": {
        "first_stage_k": [50, 100, 150, 200],
    },
    "c27_confidence_routing": {},  # Fixed routing

    # -------------------------------------------------------------------------
    # Query Adaptive (C28)
    # -------------------------------------------------------------------------
    "c28_query_type_router": {},  # Fixed routing logic

    # -------------------------------------------------------------------------
    # Advanced Reranking (C32)
    # -------------------------------------------------------------------------
    "c32_multi_stage_progressive": {
        "first_stage_k": [50, 100, 150, 200],
    },
}


def generate_grid_combinations(grid: Dict[str, List]) -> List[Dict]:
    """Generate all combinations from a parameter grid."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []

    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


class Phase2Runner:
    """Run Phase 2 coarse hyperparameter optimization."""

    def __init__(
        self,
        chunks_file: Path,
        qrels_file: Path,
        output_dir: Path,
        phase1_results: Path,
        device: Optional[str] = None,
        truncate_dim: int = 768,
    ):
        self.chunks_file = Path(chunks_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.phase1_results = Path(phase1_results)
        self.truncate_dim = truncate_dim
        self.device = device

        # Will be loaded
        self.chunks: List[Dict] = []
        self.evaluator: Optional[Evaluator] = None
        self.embedder = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.bm25_ranker: Optional[BM25Ranker] = None
        self.queries: Dict[str, str] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}

        # Top 30 strategies from Phase 1
        self.top30_strategies: List[str] = []
        self.phase1_baselines: Dict[str, float] = {}  # Strategy -> baseline MRR@5

    def load_phase1_results(self):
        """Load Phase 1 leaderboard and identify top 30."""
        print(f"\nLoading Phase 1 results from {self.phase1_results}...")

        with open(self.phase1_results) as f:
            leaderboard = json.load(f)

        # Top 30 strategies
        self.top30_strategies = [entry["strategy"] for entry in leaderboard[:30]]
        self.phase1_baselines = {
            entry["strategy"]: entry["mrr@5"]
            for entry in leaderboard[:30]
        }

        print(f"Loaded {len(self.top30_strategies)} strategies for Phase 2")
        print(f"Best Phase 1: {self.top30_strategies[0]} (MRR@5: {self.phase1_baselines[self.top30_strategies[0]]:.4f})")

    def setup(self):
        """Load data and initialize components."""
        print(f"\n{'='*60}")
        print("Phase 2: Coarse Hyperparameter Optimization Setup")
        print(f"{'='*60}")

        # Load Phase 1 results
        self.load_phase1_results()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load chunks
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

        # Initialize embedder and compute embeddings
        print(f"\nInitializing embedder on {self.device or 'auto'}...")
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
        print(f"Chunk embeddings: {self.chunk_embeddings.shape}")

        print("Computing query embeddings...")
        self.query_embeddings = self.embedder.encode_queries(
            self.queries,
            task="retrieval",
        )
        print(f"Query embeddings: {len(self.query_embeddings)}")

        # Initialize BM25
        print("\nBuilding BM25 index...")
        self.bm25_ranker = BM25Ranker()
        self.bm25_ranker.fit(self.chunks)

        print(f"\n{'='*60}")
        print("Setup complete!")
        print(f"{'='*60}\n")

    def evaluate_config(
        self,
        strategy_id: str,
        params: Dict[str, Any],
        top_k: int = 20,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a single hyperparameter configuration.

        Returns:
            Tuple of (MRR@5, full_metrics_dict)
        """
        strategy = create_strategy(strategy_id, top_k=top_k, **params)

        # Get BM25 scores if needed
        bm25_scores = None
        if strategy.requires_bm25:
            bm25_scores = {
                qid: self.bm25_ranker.get_scores(query)
                for qid, query in self.queries.items()
            }

        # Run strategy
        results = {}
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

        # Compute metrics
        config = strategy.get_config()
        eval_result = self.evaluator.evaluate_strategy(
            strategy_name=strategy_id,
            strategy_config=config.to_dict(),
            results=results,
            query_texts=self.queries,
        )

        return eval_result.metrics.get("mrr@5", 0), eval_result.metrics

    def optimize_strategy(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Run coarse grid search for a strategy.

        Returns:
            Dict with best params, metrics, and all results
        """
        grid = HYPERPARAMETER_GRIDS.get(strategy_id, {})
        combinations = generate_grid_combinations(grid)
        baseline = self.phase1_baselines.get(strategy_id, 0)

        print(f"\n  {strategy_id}: {len(combinations)} configurations")
        print(f"  Phase 1 baseline MRR@5: {baseline:.4f}")

        if len(combinations) == 1 and not combinations[0]:
            # No hyperparameters to tune
            print(f"  No tunable parameters, skipping")
            return {
                "strategy": strategy_id,
                "baseline_mrr@5": baseline,
                "best_mrr@5": baseline,
                "improvement": 0,
                "best_params": {},
                "all_results": [],
            }

        all_results = []
        best_mrr = 0
        best_params = {}
        best_metrics = {}

        for i, params in enumerate(combinations):
            try:
                mrr, metrics = self.evaluate_config(strategy_id, params)
                all_results.append({
                    "params": params,
                    "mrr@5": mrr,
                    "ndcg@5": metrics.get("ndcg@5", 0),
                })

                if mrr > best_mrr:
                    best_mrr = mrr
                    best_params = params
                    best_metrics = metrics

                # Progress indicator
                if (i + 1) % 10 == 0 or i == len(combinations) - 1:
                    print(f"    Progress: {i+1}/{len(combinations)} | Best: {best_mrr:.4f}")

            except Exception as e:
                print(f"    Error with params {params}: {e}")

        improvement = best_mrr - baseline
        print(f"  Best MRR@5: {best_mrr:.4f} (Δ {improvement:+.4f})")
        print(f"  Best params: {best_params}")

        return {
            "strategy": strategy_id,
            "baseline_mrr@5": baseline,
            "best_mrr@5": best_mrr,
            "best_ndcg@5": best_metrics.get("ndcg@5", 0),
            "improvement": improvement,
            "best_params": best_params,
            "num_configurations": len(combinations),
            "all_results": sorted(all_results, key=lambda x: x["mrr@5"], reverse=True),
        }

    def run_phase2(
        self,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run Phase 2 optimization on all strategies.

        Args:
            strategies: Optional list of strategies (default: top 30)

        Returns:
            Summary results
        """
        if strategies is None:
            strategies = self.top30_strategies

        # Filter to only valid strategies
        valid = set(get_all_strategy_ids())
        strategies = [s for s in strategies if s in valid]

        print(f"\n{'='*60}")
        print(f"Phase 2: Coarse Hyperparameter Optimization")
        print(f"Strategies: {len(strategies)}")
        print(f"{'='*60}")

        results = []
        start_time = time.time()

        for i, strategy_id in enumerate(strategies):
            print(f"\n[{i+1}/{len(strategies)}]")

            try:
                result = self.optimize_strategy(strategy_id)
                results.append(result)

                # Save incremental results
                self._save_results(results)

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "strategy": strategy_id,
                    "error": str(e),
                })

        elapsed = time.time() - start_time

        # Create Phase 2 leaderboard (by best MRR@5)
        leaderboard = sorted(
            [r for r in results if "error" not in r],
            key=lambda x: x["best_mrr@5"],
            reverse=True,
        )

        # Save final results
        self._save_results(results)
        self._save_leaderboard(leaderboard)

        # Print summary
        print(f"\n{'='*60}")
        print("PHASE 2 COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print(f"\nTop 10 Optimized Strategies:")
        print("-" * 70)
        for i, entry in enumerate(leaderboard[:10], 1):
            imp = entry.get("improvement", 0)
            print(f"  {i:2}. {entry['strategy']:35} "
                  f"MRR@5: {entry['best_mrr@5']:.4f} (Δ {imp:+.4f})")

        print(f"\nResults saved to: {self.output_dir}")

        return {
            "leaderboard": leaderboard,
            "num_strategies": len(strategies),
            "total_configurations": sum(r.get("num_configurations", 0) for r in results),
            "elapsed_seconds": elapsed,
        }

    def _save_results(self, results: List[Dict]):
        """Save all results to JSON."""
        with open(self.output_dir / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    def _save_leaderboard(self, leaderboard: List[Dict]):
        """Save leaderboard to JSON."""
        # Leaderboard with ranks
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i

        with open(self.output_dir / "leaderboard.json", 'w') as f:
            json.dump(leaderboard, f, indent=2)

        # Also save top 10 recommendations for Phase 3
        top10 = [
            {
                "strategy": e["strategy"],
                "best_params": e["best_params"],
                "mrr@5": e["best_mrr@5"],
            }
            for e in leaderboard[:10]
        ]

        with open(self.output_dir / "top10_for_phase3.json", 'w') as f:
            json.dump(top10, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Coarse Hyperparameter Optimization",
    )

    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2"))
    parser.add_argument("--phase1", type=Path, default=Path("results/phase1/leaderboard.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dim", type=int, default=768, choices=[256, 512, 768])
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run single strategy (for testing)")

    args = parser.parse_args()

    runner = Phase2Runner(
        chunks_file=args.chunks,
        qrels_file=args.qrels,
        output_dir=args.output,
        phase1_results=args.phase1,
        device=args.device,
        truncate_dim=args.dim,
    )

    runner.setup()

    strategies = [args.strategy] if args.strategy else None
    results = runner.run_phase2(strategies)

    if results["leaderboard"]:
        print(f"\nBest strategy: {results['leaderboard'][0]['strategy']}")
        print(f"Best MRR@5: {results['leaderboard'][0]['best_mrr@5']:.4f}")


if __name__ == "__main__":
    main()
