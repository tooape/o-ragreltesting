#!/usr/bin/env python3
"""
Phase 4: Model Optimization (Dimension and Quantization Analysis)

Evaluates quality/performance tradeoffs with reduced dimensionality.
Uses top 3 strategies from Phase 3 with their fine-tuned parameters.

Usage:
    python run_phase4.py --chunks data/chunks.json --qrels data/qrels.json
"""

import argparse
import json
import sys
import time
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


# Dimension configurations to evaluate
DIMENSION_CONFIGS = [
    {"name": "768d (full)", "truncate_dim": 768},
    {"name": "512d (Matryoshka)", "truncate_dim": 512},
    {"name": "256d (Matryoshka)", "truncate_dim": 256},
]


class Phase4Runner:
    """Run Phase 4 dimension optimization analysis."""

    def __init__(
        self,
        chunks_file: Path,
        qrels_file: Path,
        output_dir: Path,
        phase3_results: Path,
        device: Optional[str] = None,
    ):
        self.chunks_file = Path(chunks_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.phase3_results = Path(phase3_results)
        self.device = device

        # Will be loaded
        self.chunks: List[Dict] = []
        self.evaluator: Optional[Evaluator] = None
        self.bm25_ranker: Optional[BM25Ranker] = None
        self.queries: Dict[str, str] = {}

        # Top 3 strategies from Phase 3
        self.top3_strategies: List[Dict] = []

    def load_phase3_results(self):
        """Load Phase 3 top 3 for Phase 4."""
        print(f"\nLoading Phase 3 results from {self.phase3_results}...")

        with open(self.phase3_results) as f:
            self.top3_strategies = json.load(f)

        print(f"Loaded {len(self.top3_strategies)} strategies for Phase 4")
        for s in self.top3_strategies:
            print(f"  - {s['strategy']}: MRR@5={s['mrr@5']:.4f}, NDCG@5={s['ndcg@5']:.4f}")

    def load_data(self):
        """Load chunks and qrels."""
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

        # Initialize BM25
        print("\nBuilding BM25 index...")
        self.bm25_ranker = BM25Ranker()
        self.bm25_ranker.fit(self.chunks)

    def setup(self):
        """Load data and initialize components."""
        print(f"\n{'='*60}")
        print("Phase 4: Model Optimization (Dimension Analysis) Setup")
        print(f"{'='*60}")

        # Load Phase 3 results
        self.load_phase3_results()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.load_data()

        print(f"\n{'='*60}")
        print("Setup complete!")
        print(f"{'='*60}\n")

    def evaluate_with_dimension(
        self,
        strategy_entry: Dict,
        truncate_dim: int,
    ) -> Dict[str, Any]:
        """Evaluate a strategy at a specific dimension.

        Returns:
            Dict with metrics and timing
        """
        strategy_id = strategy_entry["strategy"]
        params = strategy_entry["best_params"]

        print(f"\n  Evaluating {strategy_id} at {truncate_dim}d...")

        # Create embedder at this dimension
        embed_start = time.time()
        embedder = create_embedder(
            model="gemma",
            device=self.device,
            truncate_dim=truncate_dim,
        )

        # Compute embeddings
        chunk_embeddings = embedder.encode_documents(
            self.chunks,
            batch_size=256,
            show_progress=False,
        )
        embed_time = time.time() - embed_start

        query_embeddings = embedder.encode_queries(
            self.queries,
            task="retrieval",
        )

        # Create strategy
        strategy = create_strategy(strategy_id, top_k=20, **params)

        # Get BM25 scores if needed
        bm25_scores = None
        if strategy.requires_bm25:
            bm25_scores = {
                qid: self.bm25_ranker.get_scores(query)
                for qid, query in self.queries.items()
            }

        # Run evaluation
        eval_start = time.time()
        results = {}
        for qid, query_text in self.queries.items():
            q_emb = query_embeddings.get(qid)
            q_bm25 = bm25_scores.get(qid) if bm25_scores else None

            result = strategy.rank(
                query=query_text,
                query_embedding=q_emb,
                chunk_embeddings=chunk_embeddings,
                chunks=self.chunks,
                bm25_scores=q_bm25,
            )
            results[qid] = result.ranked_chunk_ids

        eval_time = time.time() - eval_start

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
            "precision@5": eval_result.metrics.get("precision@5", 0),
            "embed_time_sec": embed_time,
            "eval_time_sec": eval_time,
            "embedding_shape": chunk_embeddings.shape,
            "storage_mb": chunk_embeddings.nbytes / (1024 * 1024),
        }

    def run_phase4(self) -> Dict[str, Any]:
        """Run Phase 4 dimension analysis on top 3 strategies.

        Returns:
            Summary results
        """
        print(f"\n{'='*60}")
        print("Phase 4: Dimension Analysis")
        print(f"Strategies: {len(self.top3_strategies)}")
        print(f"Dimensions: {[c['truncate_dim'] for c in DIMENSION_CONFIGS]}")
        print(f"{'='*60}")

        results = []
        start_time = time.time()

        for strategy_entry in self.top3_strategies:
            strategy_id = strategy_entry["strategy"]
            baseline_mrr = strategy_entry["mrr@5"]
            baseline_ndcg = strategy_entry["ndcg@5"]

            print(f"\n[{strategy_id}]")
            print(f"  Phase 3 baseline: MRR@5={baseline_mrr:.4f}, NDCG@5={baseline_ndcg:.4f}")

            strategy_results = {
                "strategy": strategy_id,
                "best_params": strategy_entry["best_params"],
                "phase3_baseline": {
                    "mrr@5": baseline_mrr,
                    "ndcg@5": baseline_ndcg,
                },
                "dimension_results": [],
            }

            for dim_config in DIMENSION_CONFIGS:
                dim_name = dim_config["name"]
                truncate_dim = dim_config["truncate_dim"]

                try:
                    dim_result = self.evaluate_with_dimension(
                        strategy_entry,
                        truncate_dim,
                    )

                    # Calculate degradation
                    mrr_delta = dim_result["mrr@5"] - baseline_mrr
                    ndcg_delta = dim_result["ndcg@5"] - baseline_ndcg
                    mrr_pct = (mrr_delta / baseline_mrr) * 100 if baseline_mrr > 0 else 0
                    ndcg_pct = (ndcg_delta / baseline_ndcg) * 100 if baseline_ndcg > 0 else 0

                    dim_entry = {
                        "dimension": truncate_dim,
                        "name": dim_name,
                        **dim_result,
                        "mrr_delta": mrr_delta,
                        "ndcg_delta": ndcg_delta,
                        "mrr_pct_change": mrr_pct,
                        "ndcg_pct_change": ndcg_pct,
                    }

                    strategy_results["dimension_results"].append(dim_entry)

                    print(f"    {dim_name}: MRR@5={dim_result['mrr@5']:.4f} ({mrr_delta:+.4f}, {mrr_pct:+.1f}%)")
                    print(f"             NDCG@5={dim_result['ndcg@5']:.4f} ({ndcg_delta:+.4f}, {ndcg_pct:+.1f}%)")
                    print(f"             Storage: {dim_result['storage_mb']:.1f}MB")

                except Exception as e:
                    print(f"    ERROR at {dim_name}: {e}")
                    import traceback
                    traceback.print_exc()

            results.append(strategy_results)

            # Save incremental results
            self._save_results(results)

        elapsed = time.time() - start_time

        # Save final results
        self._save_results(results)
        self._create_summary(results)

        # Print summary
        print(f"\n{'='*60}")
        print("PHASE 4 COMPLETE")
        print(f"{'='*60}")
        print(f"\nTotal time: {elapsed/60:.1f} minutes")

        print(f"\nDimension Analysis Summary:")
        print("-" * 70)
        for strategy_result in results:
            print(f"\n{strategy_result['strategy']}:")
            for dim_entry in strategy_result["dimension_results"]:
                print(f"  {dim_entry['name']:20} MRR@5: {dim_entry['mrr@5']:.4f} ({dim_entry['mrr_pct_change']:+.1f}%)  "
                      f"Storage: {dim_entry['storage_mb']:.1f}MB")

        print(f"\nResults saved to: {self.output_dir}")

        return {
            "results": results,
            "elapsed_seconds": elapsed,
        }

    def _save_results(self, results: List[Dict]):
        """Save all results to JSON."""
        with open(self.output_dir / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    def _create_summary(self, results: List[Dict]):
        """Create summary report."""
        summary = {
            "phase": "Phase 4: Model Optimization",
            "strategies_evaluated": len(results),
            "dimensions_tested": [c["truncate_dim"] for c in DIMENSION_CONFIGS],
            "strategies": [],
            "recommendation": None,
        }

        for strategy_result in results:
            strategy_summary = {
                "strategy": strategy_result["strategy"],
                "best_params": strategy_result["best_params"],
                "baseline_mrr@5": strategy_result["phase3_baseline"]["mrr@5"],
                "baseline_ndcg@5": strategy_result["phase3_baseline"]["ndcg@5"],
                "dimension_comparison": [],
            }

            for dim_entry in strategy_result["dimension_results"]:
                strategy_summary["dimension_comparison"].append({
                    "dimension": dim_entry["dimension"],
                    "mrr@5": dim_entry["mrr@5"],
                    "ndcg@5": dim_entry["ndcg@5"],
                    "mrr_change_pct": dim_entry["mrr_pct_change"],
                    "storage_mb": dim_entry["storage_mb"],
                })

            summary["strategies"].append(strategy_summary)

        # Find best tradeoff
        best_256 = results[0]["dimension_results"][2] if len(results[0]["dimension_results"]) > 2 else None
        if best_256:
            summary["recommendation"] = {
                "message": f"256d Matryoshka offers {best_256['storage_mb']:.1f}MB storage with "
                          f"{best_256['mrr_pct_change']:.1f}% MRR change",
                "production_config": {
                    "strategy": results[0]["strategy"],
                    "dimension": 256,
                    "params": results[0]["best_params"],
                }
            }

        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Model Optimization (Dimension Analysis)",
    )

    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--qrels", type=Path, default=Path("data/qrels.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase4"))
    parser.add_argument("--phase3", type=Path, default=Path("results/phase3/top3_for_phase4.json"))
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    runner = Phase4Runner(
        chunks_file=args.chunks,
        qrels_file=args.qrels,
        output_dir=args.output,
        phase3_results=args.phase3,
        device=args.device,
    )

    runner.setup()
    results = runner.run_phase4()


if __name__ == "__main__":
    main()
