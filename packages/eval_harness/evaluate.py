#!/usr/bin/env python3
"""
Evaluation harness for NL→DSL→SQL pipeline.

Computes denotation accuracy, execution accuracy, abstention precision/recall, 
and latency metrics with T1/T2/T3 breakdowns.

Usage:
    python -m packages.eval_harness.evaluate --db sqlite:///data/pms.db 
                                              --test data/ask_dataset.jsonl 
                                              --out out/metrics.json 
                                              --pred out/predictions.csv 
                                              --model baseline
"""
import argparse
import json
import sqlite3
import pandas as pd
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from packages.dsl.compiler import compile_dsl
from packages.interpreter.baseline import predict_dsl
from scripts.make_dataset import denotation_hash
from .enhanced_metrics import compute_enhanced_metrics


def load_model(model_name: str):
    """Load the specified model for prediction."""
    if model_name == "baseline":
        return predict_dsl
    elif model_name in ["llm", "hybrid"]:
        # Import LLM prediction function
        from packages.interpreter.llm import predict_dsl as predict_dsl_llm
        return lambda nl: predict_dsl_llm(nl, strategy=model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: baseline, llm, hybrid")


def safe_compile_and_execute(dsl: str, conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Safely compile DSL to SQL and execute, returning results and metadata.
    
    Returns dict with: success, sql, params, result_df, error, compile_time, execute_time
    """
    result = {
        'success': False,
        'sql': None,
        'params': None,
        'result_df': None,
        'error': None,
        'compile_time': 0.0,
        'execute_time': 0.0
    }
    
    try:
        # Compile DSL to SQL
        start_time = time.time()
        sql, params = compile_dsl(dsl, conn)
        compile_end = time.time()
        result['compile_time'] = compile_end - start_time
        result['sql'] = sql
        result['params'] = params
        
        # Execute SQL
        result_df = pd.read_sql_query(sql, conn, params=params)
        execute_end = time.time()
        result['execute_time'] = execute_end - compile_end
        result['result_df'] = result_df
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def evaluate_single_example(example: Dict[str, Any], model_fn, conn: sqlite3.Connection) -> Dict[str, Any]:
    """Evaluate a single example and return detailed results."""
    instruction = example['instruction']
    gold_dsl = example['dsl']
    gold_hash = example['denotation']['hash']
    persona = example['persona']
    tier = example['tier']
    
    # Predict DSL
    start_time = time.time()
    pred_dsl = model_fn(instruction)
    prediction_time = time.time() - start_time
    
    # Check for abstention
    abstained = (pred_dsl == "NO_TOOL")
    
    result = {
        'instruction': instruction,
        'gold_dsl': gold_dsl,
        'pred_dsl': pred_dsl,
        'persona': persona,
        'tier': tier,
        'abstained': abstained,
        'prediction_time_ms': prediction_time * 1000,
        'ok_denotation': False,
        'ok_execution': False,
        'pred_hash': None,
        'pred_n_rows': 0,
        'total_latency_ms': prediction_time * 1000,
        'error': None
    }
    
    if not abstained:
        # Try to compile and execute predicted DSL
        exec_result = safe_compile_and_execute(pred_dsl, conn)
        
        total_time = prediction_time + exec_result['compile_time'] + exec_result['execute_time']
        result['total_latency_ms'] = total_time * 1000
        result['ok_execution'] = exec_result['success']
        
        if exec_result['success']:
            # Compute denotation hash and compare
            pred_hash = denotation_hash(exec_result['result_df'])
            result['pred_hash'] = pred_hash
            result['pred_n_rows'] = len(exec_result['result_df'])
            result['ok_denotation'] = (pred_hash == gold_hash)
        else:
            result['error'] = exec_result['error']
    
    return result


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall and per-tier metrics from evaluation results."""
    
    # Overall metrics
    total = len(results)
    abstained = sum(1 for r in results if r['abstained'])
    attempted = total - abstained
    
    # Execution accuracy (among attempted)
    executed_ok = sum(1 for r in results if not r['abstained'] and r['ok_execution'])
    execution_accuracy = executed_ok / attempted if attempted > 0 else 0.0
    
    # Denotation accuracy (among attempted)
    denotation_ok = sum(1 for r in results if not r['abstained'] and r['ok_denotation'])
    denotation_accuracy = denotation_ok / attempted if attempted > 0 else 0.0
    
    # Latency stats (among attempted)
    latencies = [r['total_latency_ms'] for r in results if not r['abstained']]
    if latencies:
        latency_p50 = statistics.median(latencies)
        latency_p90 = statistics.quantiles(latencies, n=10)[8]  # 90th percentile
        latency_mean = statistics.mean(latencies)
        latency_max = max(latencies)
    else:
        latency_p50 = latency_p90 = latency_mean = latency_max = 0.0
    
    # Per-tier breakdown
    tiers = set(r['tier'] for r in results)
    tier_metrics = {}
    
    for tier in sorted(tiers):
        tier_results = [r for r in results if r['tier'] == tier]
        tier_total = len(tier_results)
        tier_abstained = sum(1 for r in tier_results if r['abstained'])
        tier_attempted = tier_total - tier_abstained
        
        tier_executed_ok = sum(1 for r in tier_results if not r['abstained'] and r['ok_execution'])
        tier_execution_acc = tier_executed_ok / tier_attempted if tier_attempted > 0 else 0.0
        
        tier_denotation_ok = sum(1 for r in tier_results if not r['abstained'] and r['ok_denotation'])
        tier_denotation_acc = tier_denotation_ok / tier_attempted if tier_attempted > 0 else 0.0
        
        tier_metrics[tier] = {
            'total': tier_total,
            'attempted': tier_attempted,
            'abstained': tier_abstained,
            'execution_accuracy': tier_execution_acc,
            'denotation_accuracy': tier_denotation_acc
        }
    
    # Abstention precision/recall (would need out-of-scope examples for full computation)
    # For now, just compute abstention rate
    abstention_rate = abstained / total if total > 0 else 0.0
    
    return {
        'overall': {
            'total_examples': total,
            'attempted': attempted,
            'abstained': abstained,
            'abstention_rate': abstention_rate,
            'execution_accuracy': execution_accuracy,
            'denotation_accuracy': denotation_accuracy,
            'latency_p50_ms': latency_p50,
            'latency_p90_ms': latency_p90,
            'latency_mean_ms': latency_mean,
            'latency_max_ms': latency_max
        },
        'by_tier': tier_metrics
    }

def run_evaluation(db_path: str, test_jsonl_path: str, model_name: str, 
                  output_metrics_path: str, output_predictions_path: str) -> Dict[str, Any]:
    """Run complete evaluation and save results."""
    
    print(f"🧪 Evaluation Harness")
    print(f"Database: {db_path}")
    print(f"Test data: {test_jsonl_path}")
    print(f"Model: {model_name}")
    print()
    
    # Load model
    model_fn = load_model(model_name)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Load test examples
    examples = []
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} test examples")
    
    # Run evaluation
    results = []
    for i, example in enumerate(examples):
        print(f"Evaluating {i+1}/{len(examples)}: {example['instruction'][:50]}...")
        result = evaluate_single_example(example, model_fn, conn)
        results.append(result)
    
    print(f"\nCompleted evaluation of {len(results)} examples")
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Compute enhanced metrics
    enhanced_metrics = compute_enhanced_metrics(results)
    metrics.update(enhanced_metrics)
    
    # Save metrics
    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Saved metrics: {output_metrics_path}")
    
    # Save predictions
    output_predictions_path = Path(output_predictions_path)
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to DataFrame for CSV output
    predictions_df = pd.DataFrame([
        {
            'instruction': r['instruction'],
            'gold_dsl': r['gold_dsl'],
            'pred_dsl': r['pred_dsl'],
            'ok_denotation': r['ok_denotation'],
            'latency_ms': r['total_latency_ms'],
            'tier': r['tier'],
            'persona': r['persona'],
            'abstained': r['abstained'],
            'ok_execution': r['ok_execution'],
            'error': r['error'] or ''
        }
        for r in results
    ])
    
    predictions_df.to_csv(output_predictions_path, index=False, encoding='utf-8')
    print(f"📝 Saved predictions: {output_predictions_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    
    overall = metrics['overall']
    print(f"Total examples: {overall['total_examples']}")
    print(f"Attempted: {overall['attempted']} ({100*overall['attempted']/overall['total_examples']:.1f}%)")
    print(f"Abstained: {overall['abstained']} ({100*overall['abstention_rate']:.1f}%)")
    print()
    print(f"Execution accuracy: {100*overall['execution_accuracy']:.1f}%")
    print(f"Denotation accuracy: {100*overall['denotation_accuracy']:.1f}%")
    print()
    print(f"Latency p50: {overall['latency_p50_ms']:.1f}ms")
    print(f"Latency p90: {overall['latency_p90_ms']:.1f}ms")
    print(f"Latency max: {overall['latency_max_ms']:.1f}ms")
    
    print("\n📊 BY TIER:")
    for tier in sorted(metrics['by_tier'].keys()):
        tier_data = metrics['by_tier'][tier]
        print(f"  {tier}: {100*tier_data['denotation_accuracy']:.1f}% denotation accuracy ({tier_data['attempted']}/{tier_data['total']} attempted)")
    
    print("\n✅ Evaluation complete!")
    
    conn.close()
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate NL→DSL→SQL pipeline")
    parser.add_argument("--db", required=True,
                       help="Database path (e.g., sqlite:///data/pms.db)")
    parser.add_argument("--test", required=True,
                       help="Test JSONL file path (e.g., data/ask_dataset.jsonl)")
    parser.add_argument("--out", default="out/metrics.json",
                       help="Output metrics JSON path")
    parser.add_argument("--pred", default="out/predictions.csv",
                       help="Output predictions CSV path")
    parser.add_argument("--model", default="baseline",
                       help="Model to evaluate (default: baseline)")
    
    args = parser.parse_args()
    
    # Extract SQLite path from URI if needed
    if args.db.startswith("sqlite:///"):
        db_path = args.db[10:]
    else:
        db_path = args.db
    
    # Validate inputs
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    if not Path(args.test).exists():
        print(f"❌ Test file not found: {args.test}")
        sys.exit(1)
    
    # Run evaluation
    run_evaluation(db_path, args.test, args.model, args.out, args.pred)


if __name__ == "__main__":
    main()
