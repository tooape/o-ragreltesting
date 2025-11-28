"""
O-RAG Ranking Strategies - Aligned to Test Candidates File

Categories:
- P1-P3: Proven Baselines (November 2025 benchmark winners)
- C1-C5: Fusion Methods (CombMNZ, CombSUM variants)
- C6-C9: Score Normalization Variants
- C10-C14: Custom Weights (data-driven)
- C15-C19: Temporal Boosting Variants
- C20-C23: Metadata-Enhanced Reranking
- C24-C27: Two-Stage Approaches
- C28-C29: Query-Adaptive Strategies
- C30-C32: Advanced Reranking
- C33-C35: BM25 Parameter Tuning
- C36-C37: Learned Fusion
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
from . import proven_baselines
from . import fusion_methods
from . import normalization
from . import custom_weights
from . import temporal_boosting
from . import metadata_enhanced
from . import two_stage
from . import query_adaptive
from . import advanced_reranking
from . import bm25_tuning
from . import learned_fusion


# Strategy categories aligned to test candidates file
CATEGORIES = {
    "proven_baselines": [
        "p1_recency_boost",
        "p2_recency_decay",
        "p3_recency_tag_combo",
    ],
    "fusion_methods": [
        "c1_combmnz_basic",
        "c2_combmnz_graph",
        "c3_combmnz_all",
        "c4_combsum_basic",
        "c5_combsum_all",
    ],
    "normalization": [
        "c6_softmax_weighted",
        "c7_rank_weighted",
        "c8_percentile_weighted",
        "c9_logscale_weighted",
    ],
    "custom_weights": [
        "c10_bm25_heavy",
        "c11_semantic_heavy",
        "c12_equal_weight",
        "c13_multi_signal_optimized",
        "c14_multi_signal_bm25_dominant",
    ],
    "temporal_boosting": [
        "c15_exp_decay_30d",
        "c16_exp_decay_60d",
        "c17_linear_decay_90d",
        "c18_step_function",
        "c19_sigmoid_boost",
    ],
    "metadata_enhanced": [
        "c20_pagetype_aware",
        "c21_tag_overlap",
        "c22_link_density",
        "c23_combined_metadata",
    ],
    "two_stage": [
        "c24_bm25_then_combmnz",
        "c25_semantic_then_combmnz",
        "c26_union_then_metadata",
        "c27_confidence_routing",
    ],
    "query_adaptive": [
        "c28_query_type_router",
        "c29_acronym_aware",
    ],
    "advanced_reranking": [
        "c30_cached_gemma",
        "c31_mixedbread_reranker",
        "c32_multi_stage_progressive",
    ],
    "bm25_tuning": [
        "c33_bm25_short_docs",
        "c34_bm25_long_docs",
        "c35_bm25_plus_expansion",
    ],
    "learned_fusion": [
        "c36_logistic_regression",
        "c37_lambdamart",
    ],
}

# All strategy IDs in order
ALL_STRATEGIES = [
    sid
    for category in [
        "proven_baselines", "fusion_methods", "normalization", "custom_weights",
        "temporal_boosting", "metadata_enhanced", "two_stage", "query_adaptive",
        "advanced_reranking", "bm25_tuning", "learned_fusion"
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
        strategy_id: Strategy identifier (e.g., 'p1_recency_boost')
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
