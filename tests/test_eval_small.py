#!/usr/bin/env python3
"""
Tests for evaluation harness functionality.
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
import pandas as pd

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.eval_harness.evaluate import run_evaluation


class TestEvaluationHarness:
    """Test evaluation harness functionality."""
    
    def test_evaluation_harness_basic(self):
        """Test basic evaluation harness functionality."""
        # Check that required files exist
        db_path = "data/pms.db"
        test_jsonl = "data/ask_dataset.jsonl"
        
        if not Path(db_path).exists():
            pytest.skip(f"Database not found: {db_path}")
        
        if not Path(test_jsonl).exists():
            pytest.skip(f"Test dataset not found: {test_jsonl}")
        
        # Run evaluation to temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            temp_metrics = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            temp_predictions = f2.name
        
        try:
            # Run evaluation
            metrics = run_evaluation(
                db_path=db_path,
                test_jsonl_path=test_jsonl,
                model_name="baseline",
                output_metrics_path=temp_metrics,
                output_predictions_path=temp_predictions
            )
            
            # Verify metrics structure
            assert 'overall' in metrics, "Metrics should have 'overall' section"
            assert 'by_tier' in metrics, "Metrics should have 'by_tier' section"
            
            overall = metrics['overall']
            required_overall_keys = {
                'total_examples', 'attempted', 'abstained', 'abstention_rate',
                'execution_accuracy', 'denotation_accuracy', 
                'latency_p50_ms', 'latency_p90_ms', 'latency_mean_ms', 'latency_max_ms'
            }
            assert set(overall.keys()) == required_overall_keys, f"Missing overall metrics: {required_overall_keys - set(overall.keys())}"
            
            # Check that files were created
            assert Path(temp_metrics).exists(), "Metrics file should be created"
            assert Path(temp_predictions).exists(), "Predictions file should be created"
            
            # Validate metrics JSON
            with open(temp_metrics, 'r') as f:
                loaded_metrics = json.load(f)
            assert loaded_metrics == metrics, "Saved metrics should match returned metrics"
            
            # Validate predictions CSV
            predictions_df = pd.read_csv(temp_predictions)
            expected_columns = {
                'instruction', 'gold_dsl', 'pred_dsl', 'ok_denotation', 
                'latency_ms', 'tier', 'persona', 'abstained', 'ok_execution', 'error'
            }
            assert set(predictions_df.columns) == expected_columns, f"Predictions CSV missing columns: {expected_columns - set(predictions_df.columns)}"
            
            # Should have 15 rows for the seed dataset
            assert len(predictions_df) == 15, f"Expected 15 predictions, got {len(predictions_df)}"
            
            # Check tier distribution
            tier_counts = predictions_df['tier'].value_counts()
            assert 'T1' in tier_counts, "Should have T1 examples"
            assert 'T2' in tier_counts, "Should have T2 examples" 
            assert 'T3' in tier_counts, "Should have T3 examples"
            
            print(f"✅ Evaluation harness working correctly")
            print(f"   Overall denotation accuracy: {100*overall['denotation_accuracy']:.1f}%")
            print(f"   Latency p50: {overall['latency_p50_ms']:.1f}ms")
            
        finally:
            # Clean up temp files
            Path(temp_metrics).unlink(missing_ok=True)
            Path(temp_predictions).unlink(missing_ok=True)
    
    def test_actual_output_files_exist(self):
        """Test that the actual output files exist and are valid."""
        metrics_path = "out/metrics.json"
        predictions_path = "out/predictions.csv"
        
        if not Path(metrics_path).exists():
            pytest.skip(f"Metrics file not found: {metrics_path}")
        
        if not Path(predictions_path).exists():
            pytest.skip(f"Predictions file not found: {predictions_path}")
        
        # Validate metrics JSON
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        assert 'overall' in metrics, "Metrics should have overall section"
        assert 'by_tier' in metrics, "Metrics should have by_tier section"
        
        overall = metrics['overall']
        assert overall['total_examples'] == 15, "Should have 15 examples"
        assert 0 <= overall['denotation_accuracy'] <= 1, "Denotation accuracy should be 0-1"
        assert 0 <= overall['execution_accuracy'] <= 1, "Execution accuracy should be 0-1"
        assert overall['latency_p50_ms'] >= 0, "Latency should be non-negative"
        
        # Validate predictions CSV
        predictions_df = pd.read_csv(predictions_path)
        assert len(predictions_df) == 15, "Should have 15 predictions"
        assert all(predictions_df['ok_denotation'].isin([True, False])), "ok_denotation should be boolean"
        assert all(predictions_df['ok_execution'].isin([True, False])), "ok_execution should be boolean"
        assert all(predictions_df['latency_ms'] >= 0), "Latency should be non-negative"
        
        print(f"✅ Validated output files:")
        print(f"   {metrics_path}: {overall['total_examples']} examples")
        print(f"   {predictions_path}: {len(predictions_df)} predictions")
    
    def test_evaluation_deterministic(self):
        """Test that evaluation is deterministic."""
        db_path = "data/pms.db"
        test_jsonl = "data/ask_dataset.jsonl"
        
        if not Path(db_path).exists() or not Path(test_jsonl).exists():
            pytest.skip("Required files not found")
        
        # Run evaluation twice to temporary files
        results = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
                temp_metrics = f1.name
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
                temp_predictions = f2.name
            
            try:
                metrics = run_evaluation(
                    db_path=db_path,
                    test_jsonl_path=test_jsonl,
                    model_name="baseline",
                    output_metrics_path=temp_metrics,
                    output_predictions_path=temp_predictions
                )
                results.append(metrics)
                
            finally:
                Path(temp_metrics).unlink(missing_ok=True)
                Path(temp_predictions).unlink(missing_ok=True)
        
        # Compare results (allowing for small floating point differences in latency)
        metrics1, metrics2 = results
        
        # Overall metrics should be identical except for latency (which can have tiny variations)
        assert metrics1['overall']['total_examples'] == metrics2['overall']['total_examples']
        assert metrics1['overall']['attempted'] == metrics2['overall']['attempted']
        assert metrics1['overall']['denotation_accuracy'] == metrics2['overall']['denotation_accuracy']
        assert metrics1['overall']['execution_accuracy'] == metrics2['overall']['execution_accuracy']
        
        # Tier metrics should be identical
        assert metrics1['by_tier'] == metrics2['by_tier']
        
        print("✅ Evaluation is deterministic (modulo latency variations)")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
