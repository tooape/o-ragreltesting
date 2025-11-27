"""
O-RAG Ranking Strategies

37 strategies organized into categories:
- C1-C6: Single Signal (dense only, BM25 only, Matryoshka dims)
- C7-C12: Simple Fusion (RRF, CombMNZ, CombSUM, interpolation)
- C13-C18: Weighted Hybrid (dense-heavy, BM25-heavy, weighted RRF)
- C19-C24: Temporal/Metadata Boost (recency, tags, pageType)
- C25-C30: Two-Stage (BM25->dense, dense->BM25, three-stage)
- C31-C33: Query-Adaptive (query type, length, confidence)
- C34-C37: Learned Fusion (grid search, Bayesian, ensemble)
"""

from .base import (
    BaseStrategy,
    DenseRetrievalStrategy,
    FusionStrategy,
    TwoStageStrategy,
    QueryAdaptiveStrategy,
    StrategyConfig,
    RankingResult,
    STRATEGY_REGISTRY,
    register_strategy,
    get_strategy,
    list_strategies,
)

# Import all strategy modules to trigger registration
from . import single_signal
from . import simple_fusion
from . import weighted_hybrid
from . import temporal_metadata
from . import two_stage
from . import query_adaptive
from . import learned_fusion


# Strategy categories for easy access
CATEGORIES = {
    "single_signal": [
        "c1_dense_only",
        "c2_bm25_only",
        "c3_dense_256",
        "c4_dense_512",
        "c5_dense_768",
        "c6_bm25_plus",
    ],
    "simple_fusion": [
        "c7_rrf_basic",
        "c8_rrf_low_k",
        "c9_combmnz",
        "c10_combsum",
        "c11_interpolation",
        "c12_max_fusion",
    ],
    "weighted_hybrid": [
        "c13_dense_heavy",
        "c14_bm25_heavy",
        "c15_weighted_rrf",
        "c16_zscore_norm",
        "c17_rank_norm",
        "c18_multi_rrf",
    ],
    "temporal_metadata": [
        "c19_recency_boost",
        "c20_recency_decay",
        "c21_tag_boost",
        "c22_pagetype_boost",
        "c23_recency_tag_combo",
        "c24_full_metadata",
    ],
    "two_stage": [
        "c25_bm25_then_dense",
        "c26_dense_then_bm25",
        "c27_hybrid_then_dense",
        "c28_bm25_then_hybrid",
        "c29_three_stage",
        "c30_large_first_stage",
    ],
    "query_adaptive": [
        "c31_query_type_adaptive",
        "c32_length_adaptive",
        "c33_confidence_adaptive",
    ],
    "learned_fusion": [
        "c34_grid_search",
        "c35_bayesian",
        "c36_per_query",
        "c37_ensemble",
    ],
}

# All strategy IDs in order
ALL_STRATEGIES = [
    sid
    for category in [
        "single_signal", "simple_fusion", "weighted_hybrid",
        "temporal_metadata", "two_stage", "query_adaptive", "learned_fusion"
    ]
    for sid in CATEGORIES[category]
]


def get_strategies_by_category(category: str) -> list:
    """Get all strategy IDs in a category."""
    return CATEGORIES.get(category, [])


def get_all_strategy_ids() -> list:
    """Get all registered strategy IDs."""
    return ALL_STRATEGIES.copy()


def create_strategy(strategy_id: str, **kwargs) -> BaseStrategy:
    """Create a strategy instance by ID.

    Args:
        strategy_id: Strategy identifier (e.g., 'c1_dense_only')
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance
    """
    return get_strategy(strategy_id, **kwargs)


def print_strategy_summary():
    """Print summary of all strategies."""
    print(f"\nO-RAG Ranking Strategies ({len(ALL_STRATEGIES)} total)")
    print("=" * 60)

    for category, strategy_ids in CATEGORIES.items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(strategy_ids)} strategies)")
        print("-" * 40)
        for sid in strategy_ids:
            if sid in STRATEGY_REGISTRY:
                cls = STRATEGY_REGISTRY[sid]
                print(f"  {sid}: {cls.DESCRIPTION}")
            else:
                print(f"  {sid}: (not registered)")


__all__ = [
    # Base classes
    "BaseStrategy",
    "DenseRetrievalStrategy",
    "FusionStrategy",
    "TwoStageStrategy",
    "QueryAdaptiveStrategy",
    "StrategyConfig",
    "RankingResult",
    # Registry
    "STRATEGY_REGISTRY",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    # Helpers
    "CATEGORIES",
    "ALL_STRATEGIES",
    "get_strategies_by_category",
    "get_all_strategy_ids",
    "create_strategy",
    "print_strategy_summary",
]
