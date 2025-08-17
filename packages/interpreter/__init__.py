"""
Interpreter package for natural language to DSL translation.

Components:
- baseline.py: Rule-based NL→DSL mapper
"""

from .baseline import predict_dsl, batch_predict

__all__ = ['predict_dsl', 'batch_predict']
