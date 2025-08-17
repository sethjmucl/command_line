"""
Evaluation harness for NL→DSL→SQL pipeline.

Components:
- evaluate.py: Main evaluation script with denotation, execution, abstention, latency metrics
"""

from .evaluate import run_evaluation

__all__ = ['run_evaluation']
