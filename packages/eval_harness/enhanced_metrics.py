#!/usr/bin/env python3
"""
Enhanced evaluation metrics for NL→DSL→SQL pipeline.

Implements intent accuracy, slot-F1, semantic accuracy, robustness, and coverage metrics.
"""

import re
import sqlite3
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


def classify_intent(dsl: str) -> str:
    """Classify DSL into intent categories."""
    if dsl == "NO_TOOL" or dsl.startswith("SUGGEST:"):
        return "abstention"
    elif dsl.startswith("FIND kpi:"):
        return "kpi"
    elif re.match(r"FIND (patients|invoices|payments|practitioners) LIMIT", dsl):
        return "entity_list"
    elif dsl.startswith("FIND appointments WHERE"):
        # Count distinct condition types for complexity
        conditions = len(re.findall(r'(practitioner=|day=|time|slot=|status=|last_7d|last_week|this_week|next_week|role=)', dsl))
        if conditions >= 3:
            return "complex_combo"
        else:
            return "filtered_appointments"
    elif dsl.startswith("FIND appointments LIMIT"):
        return "entity_list"
    else:
        return "unknown"


def parse_dsl_slots(dsl: str) -> Dict[str, str]:
    """Parse DSL into structured slots for slot-F1 computation."""
    slots = {}
    
    if dsl == "NO_TOOL" or dsl.startswith("SUGGEST:"):
        return slots
    
    # Extract domain and KPI
    if "FIND kpi:" in dsl:
        slots["domain"] = "kpi"
        kpi_match = re.search(r"kpi:(\w+)", dsl)
        if kpi_match:
            slots["kpi"] = kpi_match.group(1)
    elif "FIND " in dsl:
        entity_match = re.search(r"FIND (\w+)", dsl)
        if entity_match:
            slots["domain"] = entity_match.group(1)
    
    # Extract filters
    filters = ["practitioner", "day", "slot", "status", "role"]
    for filter_type in filters:
        pattern = f"{filter_type}='([^']+)'"
        match = re.search(pattern, dsl)
        if match:
            slots[filter_type] = match.group(1)
    
    # Time filter
    time_match = re.search(r"time >= '([^']+)'", dsl)
    if time_match:
        slots["time_ge"] = time_match.group(1)
    
    # Relative dates
    for reldate in ["last_7d", "last_week", "this_week", "next_week"]:
        if reldate in dsl:
            slots["reldate"] = reldate
            break
    
    return slots


def compute_slot_metrics(gold_slots_list: List[Dict[str, str]], pred_slots_list: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute precision, recall, F1 for each slot type across dataset."""
    all_slot_types = set()
    for slots in gold_slots_list + pred_slots_list:
        all_slot_types.update(slots.keys())
    
    slot_metrics = {}
    
    for slot_type in all_slot_types:
        tp = fp = fn = 0
        
        for gold_slots, pred_slots in zip(gold_slots_list, pred_slots_list):
            gold_val = gold_slots.get(slot_type)
            pred_val = pred_slots.get(slot_type)
            
            if gold_val is not None and pred_val is not None and gold_val == pred_val:
                tp += 1
            elif pred_val is not None and (gold_val is None or gold_val != pred_val):
                fp += 1
            elif gold_val is not None and (pred_val is None or gold_val != pred_val):
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        slot_metrics[f"{slot_type}_precision"] = precision
        slot_metrics[f"{slot_type}_recall"] = recall
        slot_metrics[f"{slot_type}_f1"] = f1
    
    # Macro averages
    precisions = [v for k, v in slot_metrics.items() if k.endswith('_precision')]
    recalls = [v for k, v in slot_metrics.items() if k.endswith('_recall')]
    f1s = [v for k, v in slot_metrics.items() if k.endswith('_f1')]
    
    slot_metrics['macro_precision'] = sum(precisions) / len(precisions) if precisions else 0.0
    slot_metrics['macro_recall'] = sum(recalls) / len(recalls) if recalls else 0.0
    slot_metrics['macro_f1'] = sum(f1s) / len(f1s) if f1s else 0.0
    
    return slot_metrics


def compute_intent_accuracy(gold_intents: List[str], pred_intents: List[str]) -> Dict[str, float]:
    """Compute intent classification accuracy and per-class metrics."""
    total = len(gold_intents)
    correct = sum(1 for g, p in zip(gold_intents, pred_intents) if g == p)
    accuracy = correct / total if total > 0 else 0.0
    
    intent_types = set(gold_intents + pred_intents)
    per_class_metrics = {}
    
    for intent in intent_types:
        tp = sum(1 for g, p in zip(gold_intents, pred_intents) if g == intent and p == intent)
        fp = sum(1 for g, p in zip(gold_intents, pred_intents) if g != intent and p == intent)
        fn = sum(1 for g, p in zip(gold_intents, pred_intents) if g == intent and p != intent)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[f"{intent}_precision"] = precision
        per_class_metrics[f"{intent}_recall"] = recall
        per_class_metrics[f"{intent}_f1"] = f1
    
    return {'intent_accuracy': accuracy, **per_class_metrics}


def analyze_coverage(gold_dsl_list: List[str]) -> Dict[str, Any]:
    """Analyze dataset coverage and feature utilization."""
    intent_counts = defaultdict(int)
    slot_counts = defaultdict(int)
    feature_usage = defaultdict(int)
    
    for dsl in gold_dsl_list:
        intent = classify_intent(dsl)
        intent_counts[intent] += 1
        
        slots = parse_dsl_slots(dsl)
        for slot_type in slots:
            slot_counts[slot_type] += 1
        
        # Feature usage analysis
        for reldate in ["last_7d", "last_week", "this_week", "next_week"]:
            if reldate in dsl:
                feature_usage[f"reldate_{reldate}"] += 1
    
    total = len(gold_dsl_list)
    return {
        'total_examples': total,
        'intent_distribution': {k: v/total for k, v in intent_counts.items()},
        'slot_usage': {k: v/total for k, v in slot_counts.items()},
        'feature_usage': {k: v/total for k, v in feature_usage.items()}
    }


def analyze_edge_cases(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze edge case handling."""
    total_attempted = sum(1 for r in results if not r['abstained'])
    
    empty_results = sum(1 for r in results if not r['abstained'] and r.get('pred_n_rows', 0) == 0)
    limit_capped = sum(1 for r in results if not r['abstained'] and r.get('pred_n_rows', 0) >= 50)
    
    return {
        'empty_result_rate': empty_results / total_attempted if total_attempted > 0 else 0.0,
        'limit_capped_rate': limit_capped / total_attempted if total_attempted > 0 else 0.0,
        'abstention_rate': sum(1 for r in results if r['abstained']) / len(results) if results else 0.0
    }


def compute_enhanced_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all enhanced evaluation metrics."""
    gold_dsls = [r['gold_dsl'] for r in results]
    pred_dsls = [r['pred_dsl'] for r in results]
    
    # 1. Intent accuracy
    gold_intents = [classify_intent(dsl) for dsl in gold_dsls]
    pred_intents = [classify_intent(dsl) for dsl in pred_dsls]
    intent_metrics = compute_intent_accuracy(gold_intents, pred_intents)
    
    # 2. Slot-F1
    gold_slots_list = [parse_dsl_slots(dsl) for dsl in gold_dsls]
    pred_slots_list = [parse_dsl_slots(dsl) for dsl in pred_dsls]
    slot_metrics = compute_slot_metrics(gold_slots_list, pred_slots_list)
    
    # 3. Coverage analysis
    coverage_metrics = analyze_coverage(gold_dsls)
    
    # 4. Edge case analysis
    edge_case_metrics = analyze_edge_cases(results)
    
    # 5. Add new columns to results
    for i, result in enumerate(results):
        result['gold_intent'] = gold_intents[i]
        result['pred_intent'] = pred_intents[i]
    
    return {
        'intent_metrics': intent_metrics,
        'slot_metrics': slot_metrics,
        'coverage_metrics': coverage_metrics,
        'edge_case_metrics': edge_case_metrics
    }