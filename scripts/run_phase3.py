#!/usr/bin/env python3
"""
Phase 3: Fine Hyperparameter Optimization

Runs fine-grained grid search around the coarse optima from Phase 2.
Only evaluates the top 10 strategies.

Usage:
    python run_phase3.py --chunks data/chunks.json --qrels data/qrels.json
    python run_phase3.py --strategy c37_lambdamart  # Single strategy
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


def generate_fine_grid(coarse_best: Dict[str, Any], strategy_id: str) -> Dict[str, List]:
    """Generate fine grid around coarse optimum.

    Creates smaller step sizes centered on the best values from Phase 2.
    """
    fine_grids = {}

    # Strategy-specific fine grids
    if strategy_id == "c37_lambdamart":
        # Best: first_stage_k=200, temporal_decay_days=15
        best_k = coarse_best.get("first_stage_k", 200)
        best_decay = coarse_best.get("temporal_decay_days", 15)
        fine_grids = {
            "first_stage_k": [max(50, best_k - 50), best_k - 25, best_k, best_k + 25, min(300, best_k + 50)],
            "temporal_decay_days": [max(5, best_decay - 5), best_decay - 2, best_decay, best_decay + 2, best_decay + 5],
        }

    elif strategy_id == "p2_recency_decay":
        # Best: alpha=0.7, recency_weight=0.3, decay_days=15
        best_alpha = coarse_best.get("alpha", 0.7)
        best_rec = coarse_best.get("recency_weight", 0.3)
        best_decay = coarse_best.get("decay_days", 15)
        fine_grids = {
            "alpha": [max(0.3, best_alpha - 0.1), best_alpha - 0.05, best_alpha, best_alpha + 0.05, min(0.9, best_alpha + 0.1)],
            "recency_weight": [max(0.1, best_rec - 0.1), best_rec - 0.05, best_rec, best_rec + 0.05, min(0.5, best_rec + 0.1)],
            "decay_days": [max(5, best_decay - 5), best_decay - 2, best_decay, best_decay + 2, best_decay + 5],
        }

    elif strategy_id == "p1_recency_boost":
        # Best: alpha=0.7, recency_weight=0.3, decay_days=14
        best_alpha = coarse_best.get("alpha", 0.7)
        best_rec = coarse_best.get("recency_weight", 0.3)
        best_decay = coarse_best.get("decay_days", 14)
        fine_grids = {
            "alpha": [max(0.3, best_alpha - 0.1), best_alpha - 0.05, best_alpha, best_alpha + 0.05, min(0.9, best_alpha + 0.1)],
            "recency_weight": [max(0.1, best_rec - 0.1), best_rec - 0.05, best_rec, best_rec + 0.05, min(0.5, best_rec + 0.1)],
            "decay_days": [max(5, best_decay - 4), best_decay - 2, best_decay, best_decay + 2, best_decay + 4],
        }

    elif strategy_id == "p3_recency_tag_combo":
        # Best: alpha=0.7, recency_weight=0.35, tag_weight=0.15, decay_days=10
        best_alpha = coarse_best.get("alpha", 0.7)
        best_rec = coarse_best.get("recency_weight", 0.35)
        best_tag = coarse_best.get("tag_weight", 0.15)
        best_decay = coarse_best.get("decay_days", 10)
        fine_grids = {
            "alpha": [max(0.4, best_alpha - 0.1), best_alpha - 0.05, best_alpha, best_alpha + 0.05, min(0.85, best_alpha + 0.1)],
            "recency_weight": [max(0.1, best_rec - 0.1), best_rec - 0.05, best_rec, best_rec + 0.05, min(0.5, best_rec + 0.1)],
            "tag_weight": [max(0.05, best_tag - 0.05), best_tag, best_tag + 0.05, min(0.3, best_tag + 0.1)],
            "decay_days": [max(5, best_decay - 3), best_decay - 1, best_decay, best_decay + 2, best_decay + 4],
        }

    elif strategy_id == "c23_combined_metadata":
        # Best: pagetype_boost=1.2, tag_boost=0.1, link_boost=0.05, decay=45
        best_pt = coarse_best.get("pagetype_boost", 1.2)
        best_tag = coarse_best.get("tag_boost_factor", 0.1)
        best_link = coarse_best.get("link_boost_factor", 0.05)
        best_decay = coarse_best.get("temporal_decay_days", 45)
        fine_grids = {
            "pagetype_boost": [max(1.0, best_pt - 0.1), best_pt - 0.05, best_pt, best_pt + 0.05, min(1.5, best_pt + 0.1)],
            "tag_boost_factor": [max(0.02, best_tag - 0.05), best_tag - 0.02, best_tag, best_tag + 0.02, min(0.3, best_tag + 0.05)],
            "link_boost_factor": [max(0.01, best_link - 0.02), best_link - 0.01, best_link, best_link + 0.01, min(0.15, best_link + 0.02)],
            "temporal_decay_days": [max(20, best_decay - 10), best_decay - 5, best_decay, best_decay + 5, best_decay + 10],
        }

    elif strategy_id == "c13_multi_signal_optimized":
        # Best: bm25=0.35, semantic=0.45, graph=0.05, temporal=0.15, decay=20
        best_bm25 = coarse_best.get("bm25_weight", 0.35)
        best_sem = coarse_best.get("semantic_weight", 0.45)
        best_graph = coarse_best.get("graph_weight", 0.05)
        best_temp = coarse_best.get("temporal_weight", 0.15)
        best_decay = coarse_best.get("temporal_decay_days", 20)
        fine_grids = {
            "bm25_weight": [max(0.2, best_bm25 - 0.1), best_bm25 - 0.05, best_bm25, best_bm25 + 0.05, min(0.6, best_bm25 + 0.1)],
            "semantic_weight": [max(0.2, best_sem - 0.1), best_sem - 0.05, best_sem, best_sem + 0.05, min(0.6, best_sem + 0.1)],
            "graph_weight": [max(0.0, best_graph - 0.03), best_graph, best_graph + 0.03, min(0.2, best_graph + 0.05)],
            "temporal_weight": [max(0.05, best_temp - 0.05), best_temp - 0.03, best_temp, best_temp + 0.03, min(0.25, best_temp + 0.05)],
            "temporal_decay_days": [max(10, best_decay - 5), best_decay - 2, best_decay, best_decay + 3, best_decay + 7],
        }

    elif strategy_id == "c15_exp_decay_30d":
        # Best: decay_days=30, temporal_weight=0.4
        best_decay = coarse_best.get("decay_days", 30)
        best_temp = coarse_best.get("temporal_weight", 0.4)
        fine_grids = {
            "decay_days": [max(10, best_decay - 10), best_decay - 5, best_decay, best_decay + 5, best_decay + 10],
            "temporal_weight": [max(0.1, best_temp - 0.15), best_temp - 0.05, best_temp, best_temp + 0.05, min(0.6, best_temp + 0.1)],
        }

    elif strategy_id == "c17_linear_decay_90d":
        # Best: window_days=45, min_score=0.05, temporal_weight=0.3
        best_window = coarse_best.get("window_days", 45)
        best_min = coarse_best.get("min_score", 0.05)
        best_temp = coarse_best.get("temporal_weight", 0.3)
        fine_grids = {
            "window_days": [max(20, best_window - 15), best_window - 7, best_window, best_window + 7, best_window + 15],
            "min_score": [max(0.01, best_min - 0.03), best_min - 0.01, best_min, best_min + 0.02, min(0.2, best_min + 0.05)],
            "temporal_weight": [max(0.1, best_temp - 0.1), best_temp - 0.05, best_temp, best_temp + 0.05, min(0.5, best_temp + 0.1)],
        }

    elif strategy_id == "c36_logistic_regression":
        # Best: temporal_decay_days=45
        best_decay = coarse_best.get("temporal_decay_days", 45)
        fine_grids = {
            "temporal_decay_days": [max(15, best_decay - 15), best_decay - 7, best_decay - 3, best_decay,
                                    best_decay + 5, best_decay + 10, best_decay + 15],
        }

    elif strategy_id == "c19_sigmoid_boost":
        # Best: midpoint_days=30, steepness=15, temporal_weight=0.3
        best_mid = coarse_best.get("midpoint_days", 30)
        best_steep = coarse_best.get("steepness", 15)
        best_temp = coarse_best.get("temporal_weight", 0.3)
        fine_grids = {
            "midpoint_days": [max(10, best_mid - 10), best_mid - 5, best_mid, best_mid + 5, best_mid + 10],
            "steepness": [max(5, best_steep - 5), best_steep - 2, best_steep, best_steep + 3, best_steep + 7],
            "temporal_weight": [max(0.1, best_temp - 0.1), best_temp - 0.05, best_temp, best_temp + 0.05, min(0.5, best_temp + 0.1)],
        }

    # Remove duplicates from each list
    for key in fine_grids:
        fine_grids[key] = sorted(list(set(fine_grids[key])))

    return fine_grids


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


class Phase3Runner:
    """Run Phase 3 fine hyperparameter optimization."""

    def __init__(
        self,
        chunks_file: Path,
        qrels_file: Path,
        output_dir: Path,
        phase2_results: Path,
        device: Optional[str] = None,
        truncate_dim: int = 768,
    ):
        self.chunks_file = Path(chunks_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.phase2_results = Path(phase2_results)
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

        # Top 10 strategies from Phase 2
        self.top10_strategies: List[Dict] = []
        self.phase2_baselines: Dict[str, float] = {}  # Strategy -> best MRR@5 from Phase 2

    def load_phase2_results(self):
        """Load Phase 2 top 10 for Phase 3."""
        print(f"\nLoading Phase 2 results from {self.phase2_results}...")

        with open(self.phase2_results) as f:
            self.top10_strategies = json.load(f)

        self.phase2_baselines = {
            entry["strategy"]: entry["mrr@5"]
            for entry in self.top10_strategies
        }

        print(f"Loaded {len(self.top10_strategies)} strategies for Phase 3")
        print(f"Best Phase 2: {self.top10_strategies[0]['strategy']} (MRR@5: {self.top10_strategies[0]['mrr@5']:.4f})")

    def setup(self):
        """Load data and initialize components."""
        print(f"\n{'='*60}")
        print("Phase 3: Fine Hyperparameter Optimization Setup")
        print(f"{'='*60}")

        # Load Phase 2 results
        self.load_phase2_results()

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
        coarse_best: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run fine grid search for a strategy around coarse optimum.

        Returns:
            Dict with best params, metrics, and all results
        """
        # Generate fine grid around coarse optimum
        fine_grid = generate_fine_grid(coarse_best, strategy_id)
        combinations = generate_grid_combinations(fine_grid)
        baseline = self.phase2_baselines.get(strategy_id, 0)

        print(f"\n  {strategy_id}: {len(combinations)} fine configurations")
        print(f"  Phase 2 baseline MRR@5: {baseline:.4f}")
        print(f"  Coarse best params: {coarse_best}")
        print(f"  Fine grid: {fine_grid}")

        if len(combinations) == 0:
            print(f"  No configurations to evaluate, keeping coarse result")
            return {
                "strategy": strategy_id,
                "baseline_mrr@5": baseline,
                "best_mrr@5": baseline,
                "improvement": 0,
                "best_params": coarse_best,
                "all_results": [],
            }

        all_results = []
        best_mrr = 0
        best_params = coarse_best.copy()
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
                if (i + 1) % 25 == 0 or i == len(combinations) - 1:
                    print(f"    Progress: {i+1}/{len(combinations)} | Best: {best_mrr:.4f}")

            except Exception as e:
                print(f"    Error with params {params}: {e}")

        improvement = best_mrr - baseline
        print(f"  Best MRR@5: {best_mrr:.4f} (Δ vs Phase 2: {improvement:+.4f})")
        print(f"  Best params: {best_params}")

        return {
            "strategy": strategy_id,
            "baseline_mrr@5": baseline,
            "best_mrr@5": best_mrr,
            "best_ndcg@5": best_metrics.get("ndcg@5", 0),
            "improvement_vs_phase2": improvement,
            "best_params": best_params,
            "num_configurations": len(combinations),
            "all_results": sorted(all_results, key=lambda x: x["mrr@5"], reverse=True),
        }

    def run_phase3(
        self,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run Phase 3 fine optimization on top 10 strategies.

        Args:
            strategies: Optional list of strategies (default: top 10)

        Returns:
            Summary results
        """
        # Build strategy list with coarse params
        if strategies:
            strategy_list = [
                s for s in self.top10_strategies
                if s["strategy"] in strategies
            ]
        else:
            strategy_list = self.top10_strategies

        print(f"\n{'='*60}")
        print(f"Phase 3: Fine Hyperparameter Optimization")
        print(f"Strategies: {len(strategy_list)}")
        print(f"{'='*60}")

        results = []
        start_time = time.time()

        for i, entry in enumerate(strategy_list):
            strategy_id = entry["strategy"]
            coarse_best = entry["best_params"]

            print(f"\n[{i+1}/{len(strategy_list)}]")

            try:
                result = self.optimize_strategy(strategy_id, coarse_best)
                results.append(result)

                # Save incremental results
                self._save_results(results)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "strategy": strategy_id,
                    "error": str(e),
                })

        elapsed = time.time() - start_time

        # Create Phase 3 leaderboard (by best MRR@5)
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
        print("PHASE 3 COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        print(f"\nTop 10 Fine-Optimized Strategies:")
        print("-" * 70)
        for i, entry in enumerate(leaderboard[:10], 1):
            imp = entry.get("improvement_vs_phase2", 0)
            print(f"  {i:2}. {entry['strategy']:35} "
                  f"MRR@5: {entry['best_mrr@5']:.4f} (Δ vs P2: {imp:+.4f})")

        print(f"\nResults saved to: {self.output_dir}")

        return {
            "leaderboard": leaderboard,
            "num_strategies": len(strategy_list),
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

        # Also save top 3 recommendations for Phase 4
        top3 = [
            {
                "strategy": e["strategy"],
                "best_params": e["best_params"],
                "mrr@5": e["best_mrr@5"],
                "ndcg@5": e.get("best_ndcg@5", 0),
            }
            for e in leaderboard[:3]
        ]

        with open(self.output_dir / "top3_for_phase4.json", 'w') as f:
            json.dump(top3, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Fine Hyperparameter Optimization",
    )

    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase3"))
    parser.add_argument("--phase2", type=Path, default=Path("results/phase2/top10_for_phase3.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dim", type=int, default=768, choices=[256, 512, 768])
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run single strategy (for testing)")

    args = parser.parse_args()

    runner = Phase3Runner(
        chunks_file=args.chunks,
        qrels_file=args.qrels,
        output_dir=args.output,
        phase2_results=args.phase2,
        device=args.device,
        truncate_dim=args.dim,
    )

    runner.setup()

    strategies = [args.strategy] if args.strategy else None
    results = runner.run_phase3(strategies)

    if results["leaderboard"]:
        print(f"\nBest strategy: {results['leaderboard'][0]['strategy']}")
        print(f"Best MRR@5: {results['leaderboard'][0]['best_mrr@5']:.4f}")


if __name__ == "__main__":
    main()
