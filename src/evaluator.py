"""
Evaluation metrics for O-RAG retrieval quality.

Implements standard IR metrics:
- MRR@k: Mean Reciprocal Rank
- NDCG@k: Normalized Discounted Cumulative Gain
- P@k: Precision at k
- Recall@k: Recall at k
- MAP@k: Mean Average Precision
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: str
    query_text: str
    retrieved: List[str]  # Retrieved chunk IDs in ranked order
    relevant: List[str]  # Relevant chunk IDs from qrels
    scores: Dict[str, float] = field(default_factory=dict)  # Retrieval scores

    def __post_init__(self):
        self.relevant_set = set(self.relevant)
        self.retrieved_relevant = [
            (i, doc_id) for i, doc_id in enumerate(self.retrieved)
            if doc_id in self.relevant_set
        ]


@dataclass
class EvaluationResult:
    """Complete evaluation results for a strategy."""
    strategy_name: str
    strategy_config: Dict[str, Any]
    metrics: Dict[str, float]
    per_query_metrics: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "EvaluationResult":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class Evaluator:
    """Compute retrieval evaluation metrics."""

    def __init__(
        self,
        qrels: Dict[str, List[str]],
        k_values: List[int] = None,
    ):
        """Initialize evaluator.

        Args:
            qrels: Dict mapping query_id -> list of relevant chunk IDs
            k_values: K values for metrics (default: [1, 3, 5, 10, 20])
        """
        self.qrels = qrels
        self.k_values = k_values or [1, 3, 5, 10, 20]

    @classmethod
    def from_qrels_file(cls, path: Path, **kwargs) -> "Evaluator":
        """Load qrels from JSON file.

        Expected format:
        {
            "queries": {
                "q1": {
                    "query_text": "...",
                    "relevant_chunks": ["chunk_id_1", "chunk_id_2", ...]
                },
                ...
            }
        }

        Or simpler format:
        {
            "q1": ["chunk_id_1", "chunk_id_2", ...],
            ...
        }
        """
        with open(path) as f:
            data = json.load(f)

        # Handle nested format
        if "queries" in data:
            qrels = {
                qid: q["relevant_chunks"]
                for qid, q in data["queries"].items()
            }
        else:
            # Assume simple format
            qrels = {
                qid: (v["relevant_chunks"] if isinstance(v, dict) else v)
                for qid, v in data.items()
            }

        return cls(qrels, **kwargs)

    def mrr_at_k(self, result: QueryResult, k: int) -> float:
        """Mean Reciprocal Rank at k.

        Args:
            result: Query result
            k: Cutoff

        Returns:
            1/rank of first relevant, or 0 if none in top k
        """
        for i, doc_id in enumerate(result.retrieved[:k]):
            if doc_id in result.relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def precision_at_k(self, result: QueryResult, k: int) -> float:
        """Precision at k.

        Args:
            result: Query result
            k: Cutoff

        Returns:
            Fraction of top k that are relevant
        """
        retrieved_k = result.retrieved[:k]
        relevant_in_k = sum(1 for doc_id in retrieved_k if doc_id in result.relevant_set)
        return relevant_in_k / k if k > 0 else 0.0

    def recall_at_k(self, result: QueryResult, k: int) -> float:
        """Recall at k.

        Args:
            result: Query result
            k: Cutoff

        Returns:
            Fraction of relevant docs found in top k
        """
        if not result.relevant:
            return 0.0

        retrieved_k = result.retrieved[:k]
        relevant_in_k = sum(1 for doc_id in retrieved_k if doc_id in result.relevant_set)
        return relevant_in_k / len(result.relevant)

    def ndcg_at_k(
        self,
        result: QueryResult,
        k: int,
        relevance_grades: Optional[Dict[str, int]] = None,
    ) -> float:
        """Normalized Discounted Cumulative Gain at k.

        Args:
            result: Query result
            k: Cutoff
            relevance_grades: Optional dict of doc_id -> relevance grade.
                             If None, uses binary relevance (1 if in qrels, else 0)

        Returns:
            NDCG score in [0, 1]
        """
        # Get relevance for retrieved docs
        retrieved_k = result.retrieved[:k]

        if relevance_grades is None:
            # Binary relevance
            gains = [1.0 if doc_id in result.relevant_set else 0.0 for doc_id in retrieved_k]
        else:
            gains = [float(relevance_grades.get(doc_id, 0)) for doc_id in retrieved_k]

        # DCG
        dcg = sum(
            gain / np.log2(i + 2)  # +2 because log2(1) = 0
            for i, gain in enumerate(gains)
        )

        # IDCG (ideal DCG with perfect ranking)
        if relevance_grades is None:
            ideal_gains = [1.0] * min(k, len(result.relevant))
        else:
            ideal_gains = sorted(
                [relevance_grades.get(doc_id, 0) for doc_id in result.relevant],
                reverse=True
            )[:k]

        idcg = sum(
            gain / np.log2(i + 2)
            for i, gain in enumerate(ideal_gains)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def average_precision(self, result: QueryResult, k: int = None) -> float:
        """Average Precision.

        Args:
            result: Query result
            k: Optional cutoff (None = use all retrieved)

        Returns:
            AP score
        """
        if not result.relevant:
            return 0.0

        retrieved = result.retrieved[:k] if k else result.retrieved

        precisions = []
        relevant_count = 0

        for i, doc_id in enumerate(retrieved):
            if doc_id in result.relevant_set:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))

        if not precisions:
            return 0.0

        return sum(precisions) / len(result.relevant)

    def evaluate_query(
        self,
        query_id: str,
        retrieved: List[str],
        scores: Optional[Dict[str, float]] = None,
        query_text: str = "",
    ) -> Dict[str, float]:
        """Evaluate a single query.

        Args:
            query_id: Query identifier
            retrieved: List of retrieved chunk IDs in ranked order
            scores: Optional retrieval scores
            query_text: Optional query text

        Returns:
            Dict of metric_name -> value
        """
        if query_id not in self.qrels:
            return {}

        result = QueryResult(
            query_id=query_id,
            query_text=query_text,
            retrieved=retrieved,
            relevant=self.qrels[query_id],
            scores=scores or {},
        )

        metrics = {}

        for k in self.k_values:
            metrics[f"mrr@{k}"] = self.mrr_at_k(result, k)
            metrics[f"p@{k}"] = self.precision_at_k(result, k)
            metrics[f"recall@{k}"] = self.recall_at_k(result, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(result, k)
            metrics[f"map@{k}"] = self.average_precision(result, k)

        return metrics

    def evaluate_batch(
        self,
        results: Dict[str, List[str]],
        query_texts: Optional[Dict[str, str]] = None,
        scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Evaluate multiple queries.

        Args:
            results: Dict of query_id -> list of retrieved chunk IDs
            query_texts: Optional dict of query_id -> query text
            scores: Optional dict of query_id -> {chunk_id: score}

        Returns:
            Tuple of (aggregated_metrics, per_query_metrics)
        """
        query_texts = query_texts or {}
        scores = scores or {}

        per_query = []
        all_metrics = {
            f"{metric}@{k}": []
            for metric in ["mrr", "p", "recall", "ndcg", "map"]
            for k in self.k_values
        }

        for query_id, retrieved in results.items():
            if query_id not in self.qrels:
                continue

            metrics = self.evaluate_query(
                query_id=query_id,
                retrieved=retrieved,
                scores=scores.get(query_id),
                query_text=query_texts.get(query_id, ""),
            )

            per_query.append({
                "query_id": query_id,
                "query_text": query_texts.get(query_id, ""),
                "num_retrieved": len(retrieved),
                "num_relevant": len(self.qrels[query_id]),
                "metrics": metrics,
            })

            for metric_name, value in metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(value)

        # Aggregate
        aggregated = {
            metric_name: np.mean(values) if values else 0.0
            for metric_name, values in all_metrics.items()
        }

        # Add summary stats
        aggregated["num_queries"] = len(per_query)
        aggregated["num_queries_with_relevant"] = sum(
            1 for pq in per_query if pq["num_relevant"] > 0
        )

        return aggregated, per_query

    def evaluate_strategy(
        self,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        results: Dict[str, List[str]],
        query_texts: Optional[Dict[str, str]] = None,
        scores: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate a complete strategy run.

        Args:
            strategy_name: Name of the strategy
            strategy_config: Strategy configuration
            results: Dict of query_id -> list of retrieved chunk IDs
            query_texts: Optional dict of query_id -> query text
            scores: Optional dict of query_id -> {chunk_id: score}
            metadata: Optional additional metadata

        Returns:
            EvaluationResult object
        """
        aggregated, per_query = self.evaluate_batch(
            results=results,
            query_texts=query_texts,
            scores=scores,
        )

        return EvaluationResult(
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            metrics=aggregated,
            per_query_metrics=per_query,
            metadata=metadata or {},
        )


class ResultsAggregator:
    """Aggregate and compare results across strategies."""

    def __init__(self):
        self.results: List[EvaluationResult] = []

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.results.append(result)

    def get_leaderboard(
        self,
        primary_metric: str = "mrr@5",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get sorted leaderboard by metric.

        Args:
            primary_metric: Metric to sort by
            ascending: Sort in ascending order

        Returns:
            List of strategy results sorted by metric
        """
        leaderboard = [
            {
                "rank": 0,
                "strategy": r.strategy_name,
                "config": r.strategy_config,
                primary_metric: r.metrics.get(primary_metric, 0),
                **{k: v for k, v in r.metrics.items() if k != primary_metric},
            }
            for r in self.results
        ]

        leaderboard.sort(
            key=lambda x: x[primary_metric],
            reverse=not ascending,
        )

        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def get_best_strategy(
        self,
        metric: str = "mrr@5",
    ) -> Optional[EvaluationResult]:
        """Get best strategy by metric.

        Args:
            metric: Metric to maximize

        Returns:
            Best EvaluationResult or None
        """
        if not self.results:
            return None

        return max(self.results, key=lambda r: r.metrics.get(metric, 0))

    def compare_strategies(
        self,
        metrics: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare all strategies across metrics.

        Args:
            metrics: Metrics to include (default: all)

        Returns:
            Dict of strategy_name -> {metric: value}
        """
        comparison = {}

        for result in self.results:
            if metrics:
                filtered = {k: v for k, v in result.metrics.items() if k in metrics}
            else:
                filtered = result.metrics

            comparison[result.strategy_name] = filtered

        return comparison

    def to_json(self, path: Path) -> None:
        """Save all results to JSON."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "leaderboard": self.get_leaderboard(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "ResultsAggregator":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)

        agg = cls()
        for r in data.get("results", []):
            agg.add_result(EvaluationResult(**r))

        return agg


def quick_evaluate(
    retrieved: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Quick evaluation helper.

    Args:
        retrieved: Dict of query_id -> retrieved chunk IDs
        qrels: Dict of query_id -> relevant chunk IDs
        k_values: K values (default: [1, 5, 10])

    Returns:
        Aggregated metrics
    """
    evaluator = Evaluator(qrels, k_values or [1, 5, 10])
    metrics, _ = evaluator.evaluate_batch(retrieved)
    return metrics
