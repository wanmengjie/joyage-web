"""
Research Module for Scientific Analysis
科研分析模块
"""

from .scientific_strategy_manager import (
    ScientificStrategyManager,
    ResearchStrategy,
    ResearchPhase,
    MethodologicalRigor,
    run_research_pipeline
)

__all__ = [
    'ScientificStrategyManager',
    'ResearchStrategy', 
    'ResearchPhase',
    'MethodologicalRigor',
    'run_research_pipeline'
] 