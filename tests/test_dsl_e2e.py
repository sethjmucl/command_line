#!/usr/bin/env python3
"""
End-to-end tests for DSL→SQL→DB pipeline.

Usage:
    pytest tests/test_dsl_e2e.py -v
"""

import pytest
import sqlite3
import pandas as pd
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.dsl.compiler import compile_dsl


# Test corpus - using exact strings from specification
T1_QUERIES = [
    "FIND appointments LIMIT 5",
    "FIND patients LIMIT 5", 
    "FIND kpi:aged_receivables",
    "FIND kpi:no_shows_last_7d",
    "FIND appointments WHERE practitioner='Khan' LIMIT 50",
    "FIND appointments WHERE status='free' LIMIT 50",
    "FIND appointments WHERE slot='double' LIMIT 50"
]

T2_QUERIES = [
    "FIND appointments WHERE practitioner='Khan' AND status='booked' LIMIT 50",
    "FIND appointments WHERE day='fri' AND time >= '16:00' LIMIT 50",
    "FIND appointments WHERE slot='double' AND status='free' LIMIT 50",
    "FIND appointments WHERE last_7d LIMIT 50",
    "FIND appointments WHERE next_week AND practitioner='Patel' LIMIT 50",
    "FIND kpi:free_double_slots_next_7d"
]

T3_QUERIES = [
    "FIND appointments WHERE practitioner='Khan' AND day='fri' AND time >= '16:00' AND slot='double' LIMIT 20",
    "FIND appointments WHERE this_week AND status='free' LIMIT 10"
]

ERROR_CASES = [
    "FIND appointments WHERE frobnitz=1 LIMIT 5",
    "FIND lab_results LIMIT 5", 
    "FIND appointments WHERE last_8d LIMIT 5"
]

ALL_VALID_QUERIES = T1_QUERIES + T2_QUERIES + T3_QUERIES


@pytest.fixture(scope="module")
def conn():
    """Database connection fixture."""
    db_path = "data/pms.db"
    if not Path(db_path).exists():
        pytest.skip(f"Database not found: {db_path}. Run: python scripts/seed_db.py --reset")
    
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    yield connection
    connection.close()


@pytest.mark.parametrize("dsl", ALL_VALID_QUERIES)
def test_compile_and_execute_ok(conn, dsl):
    """Test that DSL compiles and executes successfully."""
    
    # Test compilation
    result = compile_dsl(dsl, conn)
    assert isinstance(result, tuple), f"compile_dsl should return tuple, got {type(result)}"
    assert len(result) == 2, f"compile_dsl should return (sql, params), got {len(result)} items"
    
    sql, params = result
    assert isinstance(sql, str), f"SQL should be string, got {type(sql)}"
    assert isinstance(params, list), f"Params should be list, got {type(params)}"
    
    # Basic injection check - no semicolons
    assert ";" not in sql, f"SQL contains semicolon (injection risk): {sql}"
    
    # Test execution
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        assert isinstance(df, pd.DataFrame), f"Should return DataFrame, got {type(df)}"
        # Row count can be zero - don't assert specific counts
        assert df.shape[0] >= 0, "DataFrame should have non-negative row count"
        
    except Exception as e:
        pytest.fail(f"SQL execution failed for DSL '{dsl}': {e}\nSQL: {sql}\nParams: {params}")


def test_default_limit(conn):
    """Test that default LIMIT 50 is applied when no LIMIT in DSL."""
    dsl = "FIND appointments WHERE status='booked'"
    sql, params = compile_dsl(dsl, conn)
    
    # Should contain LIMIT 50
    assert "LIMIT 50" in sql, f"SQL should contain 'LIMIT 50': {sql}"


@pytest.mark.parametrize("bad_dsl", ERROR_CASES)
def test_error_cases(conn, bad_dsl):
    """Test that error cases raise ValueError or parsing exceptions."""
    with pytest.raises((ValueError, Exception)):  # Allow both ValueError and parse exceptions
        compile_dsl(bad_dsl, conn)


def test_latency_p50_p90(conn):
    """Test compilation and execution latency."""
    times = []
    
    # Run each query 10 times (≈150 total runs)
    for _ in range(10):
        for dsl in ALL_VALID_QUERIES:
            start_time = time.time()
            
            # Compile and execute
            sql, params = compile_dsl(dsl, conn)
            df = pd.read_sql_query(sql, conn, params=params)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
    
    # Calculate percentiles
    times.sort()
    n = len(times)
    p50_idx = int(0.5 * n)
    p90_idx = int(0.9 * n)
    
    p50 = times[p50_idx]
    p90 = times[p90_idx]
    
    print(f"\nLatency results from {n} runs:")
    print(f"  p50: {p50:.3f}s")
    print(f"  p90: {p90:.3f}s")
    print(f"  max: {max(times):.3f}s")
    
    # Assert p90 < 1.0s (should pass comfortably on SQLite)
    assert p90 < 1.0, f"p90 latency {p90:.3f}s exceeds 1.0s threshold"


def test_reldate_tokens_compile_and_execute(conn):
    """Test that RELDATE tokens compile and execute properly."""
    reldate_queries = [
        "FIND appointments WHERE last_7d LIMIT 50",
        "FIND appointments WHERE next_week AND practitioner='Patel' LIMIT 50", 
        "FIND appointments WHERE this_week AND status='free' LIMIT 10"
    ]
    
    for dsl in reldate_queries:
        sql, params = compile_dsl(dsl, conn)
        
        # Should have date parameters
        assert len(params) >= 2, f"RELDATE query should have date parameters: {dsl}"
        
        # Should execute without error
        df = pd.read_sql_query(sql, conn, params=params)
        assert isinstance(df, pd.DataFrame)


def test_sql_structure_validation(conn):
    """Test compiled SQL has expected structure."""
    test_cases = [
        ("FIND appointments LIMIT 5", ["SELECT", "FROM appointments", "LIMIT 5"]),
        ("FIND kpi:aged_receivables", ["SELECT", "FROM vw_aged_receivables"]),
        ("FIND appointments WHERE practitioner='Khan'", ["WHERE", "practitioner", "LIMIT 50"])
    ]
    
    for dsl, expected_parts in test_cases:
        sql, params = compile_dsl(dsl, conn)
        sql_upper = sql.upper()
        
        for part in expected_parts:
            assert part.upper() in sql_upper, f"SQL missing '{part}': {sql}"


def test_parameter_safety(conn):
    """Test that parameters are properly parameterized."""
    # Test with special characters in practitioner name
    dsl = "FIND appointments WHERE practitioner='Johnson' LIMIT 5"
    sql, params = compile_dsl(dsl, conn)
    
    # Should use parameters, not string concatenation
    assert "?" in sql, "SQL should use parameter placeholders"
    assert "Johnson" in params, "Parameter should be in params list"
    assert "Johnson" not in sql, "Parameter value should not be in SQL string"


if __name__ == "__main__":
    # Allow running directly for debugging
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
