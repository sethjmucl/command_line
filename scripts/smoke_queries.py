#!/usr/bin/env python3
"""
Smoke test runner for DSL queries.

Runs the complete DSL test corpus and prints results for manual inspection.

Usage:
    python scripts/smoke_queries.py --db sqlite:///data/pms.db
"""

import argparse
import sqlite3
import pandas as pd
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.dsl.compiler import compile_dsl


# Same test corpus as E2E tests
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


def format_params(params):
    """Format parameter list for display."""
    if not params:
        return "[]"
    
    formatted = []
    for p in params:
        if isinstance(p, str):
            formatted.append(f"'{p}'")
        else:
            formatted.append(str(p))
    return "[" + ", ".join(formatted) + "]"


def truncate_sql(sql, max_length=80):
    """Truncate SQL for display."""
    if len(sql) <= max_length:
        return sql
    return sql[:max_length-3] + "..."


def format_dataframe_preview(df, max_rows=5):
    """Format DataFrame for concise display."""
    if df.empty:
        return "  (No rows returned)"
    
    # Show first few rows
    preview_df = df.head(max_rows)
    
    # Format with limited width
    lines = []
    lines.append(f"  {len(df)} rows returned. First {min(max_rows, len(df))} rows:")
    
    # Column headers
    cols = list(preview_df.columns)
    if len(cols) > 6:  # Limit columns shown
        cols = cols[:6] + ["..."]
        preview_df = preview_df.iloc[:, :6]
    
    # Simple table format
    for i, row in preview_df.iterrows():
        row_str = "  " + " | ".join(str(val)[:15] for val in row.values)
        lines.append(row_str)
    
    return "\n".join(lines)


def run_query_safely(dsl, conn):
    """Run a single query safely and return results."""
    try:
        start_time = time.time()
        
        # Compile
        sql, params = compile_dsl(dsl, conn)
        compile_time = time.time()
        
        # Execute
        df = pd.read_sql_query(sql, conn, params=params)
        end_time = time.time()
        
        return {
            'success': True,
            'sql': sql,
            'params': params,
            'dataframe': df,
            'compile_time': compile_time - start_time,
            'execute_time': end_time - compile_time,
            'total_time': end_time - start_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def run_smoke_tests(db_path):
    """Run all smoke tests and print results."""
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    print(f"🏥 DSL Smoke Test Runner")
    print(f"Database: {db_path}")
    print("=" * 80)
    
    all_success = True
    total_queries = 0
    total_time = 0
    
    # Test valid queries by tier
    for tier_name, queries in [("T1", T1_QUERIES), ("T2", T2_QUERIES), ("T3", T3_QUERIES)]:
        print(f"\n📊 {tier_name} Queries ({len(queries)} total)")
        print("-" * 40)
        
        for i, dsl in enumerate(queries, 1):
            result = run_query_safely(dsl, conn)
            total_queries += 1
            
            if result['success']:
                total_time += result['total_time']
                
                print(f"\n{i}. ✅ {dsl}")
                print(f"   SQL: {truncate_sql(result['sql'])}")
                print(f"   Params: {format_params(result['params'])}")
                print(f"   Time: {result['total_time']:.3f}s (compile: {result['compile_time']:.3f}s)")
                print(format_dataframe_preview(result['dataframe']))
                
            else:
                all_success = False
                print(f"\n{i}. ❌ {dsl}")
                print(f"   Error: {result['error_type']}: {result['error']}")
    
    # Test error cases
    print(f"\n🚫 Error Cases ({len(ERROR_CASES)} total)")
    print("-" * 40)
    
    for i, dsl in enumerate(ERROR_CASES, 1):
        result = run_query_safely(dsl, conn)
        
        if result['success']:
            all_success = False
            print(f"\n{i}. ❌ {dsl}")
            print(f"   UNEXPECTED SUCCESS - should have failed!")
        else:
            print(f"\n{i}. ✅ {dsl}")
            print(f"   Expected error: {result['error_type']}: {result['error']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 Summary")
    print(f"  Total valid queries: {total_queries}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time: {total_time/total_queries:.3f}s")
    print(f"  Error cases handled: {len(ERROR_CASES)}")
    
    if all_success:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")
    
    conn.close()
    return all_success


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run DSL smoke tests")
    parser.add_argument("--db", default="sqlite:///data/pms.db",
                       help="Database path (default: sqlite:///data/pms.db)")
    
    args = parser.parse_args()
    
    # Extract SQLite path from URI
    if args.db.startswith("sqlite:///"):
        db_path = args.db[10:]  # Remove sqlite:/// prefix
    else:
        db_path = args.db
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print(f"Run: python scripts/seed_db.py --db {args.db} --reset")
        sys.exit(1)
    
    success = run_smoke_tests(db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
