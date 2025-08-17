#!/usr/bin/env python3
"""
Generate JSONL dataset from seed queries.

Loads seeds.csv, compiles DSL→SQL, executes against DB, computes denotation hashes.

Usage:
    python scripts/make_dataset.py --db sqlite:///data/pms.db --out data/ask_dataset.jsonl
"""

import argparse
import sqlite3
import pandas as pd
import json
import hashlib
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.dsl.compiler import compile_dsl


def canonicalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize DataFrame for deterministic hashing.
    
    - Sort by all columns ascending
    - Format timestamps to ISO (YYYY-MM-DD HH:MM:SS)
    - Round floats to 2 decimal places
    - Ensure consistent dtype casts
    """
    df2 = df.copy()
    
    # Handle timestamps
    for c in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[c]):
            df2[c] = pd.to_datetime(df2[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
        elif df2[c].dtype.kind == "f":  # float types
            df2[c] = df2[c].round(2)
        elif df2[c].dtype == "object":
            # Handle None/NULL values consistently
            df2[c] = df2[c].astype(str).replace('nan', 'None')
    
    # Sort by all columns ascending for deterministic ordering
    return df2.sort_values(list(df2.columns)).reset_index(drop=True)


def denotation_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of DataFrame contents."""
    csv_string = canonicalize_df(df).to_csv(index=False)
    return hashlib.sha256(csv_string.encode('utf-8')).hexdigest()


def make_dataset(db_path: str, seeds_path: str, output_path: str):
    """Generate dataset JSONL from seeds CSV."""
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Load seeds
    seeds_df = pd.read_csv(seeds_path)
    print(f"Loaded {len(seeds_df)} seed queries from {seeds_path}")
    
    results = []
    
    for idx, row in seeds_df.iterrows():
        instruction = row['instruction']
        dsl = row['dsl']
        persona = row['persona']
        tier = row['tier']
        
        print(f"Processing {idx+1}/{len(seeds_df)}: {instruction[:50]}...")
        
        try:
            # Compile DSL to SQL
            sql, params = compile_dsl(dsl, conn)
            
            # Execute query and get results
            result_df = pd.read_sql_query(sql, conn, params=params)
            
            # Compute denotation hash
            denotation_hash_str = denotation_hash(result_df)
            
            # Create result entry
            result_entry = {
                'instruction': instruction,
                'dsl': dsl,
                'sql': sql,
                'denotation': {
                    'hash': denotation_hash_str,
                    'n_rows': len(result_df)
                },
                'persona': persona,
                'tier': tier
            }
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"Error processing '{instruction}': {e}")
            # Create entry with error info
            result_entry = {
                'instruction': instruction,
                'dsl': dsl,
                'sql': None,
                'denotation': {
                    'hash': None,
                    'n_rows': 0,
                    'error': str(e)
                },
                'persona': persona,
                'tier': tier
            }
            results.append(result_entry)
    
    # Write JSONL output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Generated dataset: {output_path}")
    print(f"   Total entries: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if r['denotation']['hash'] is not None)}")
    print(f"   Errors: {sum(1 for r in results if r['denotation']['hash'] is None)}")
    
    # Print sample results
    print("\n📊 Sample Results:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result['instruction']}")
        print(f"   DSL: {result['dsl']}")
        if result['denotation']['hash']:
            print(f"   Hash: {result['denotation']['hash'][:16]}...")
            print(f"   Rows: {result['denotation']['n_rows']}")
        else:
            print(f"   Error: {result['denotation'].get('error', 'Unknown')}")
        print()
    
    conn.close()
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate dataset from seed queries")
    parser.add_argument("--db", default="sqlite:///data/pms.db",
                       help="Database path (default: sqlite:///data/pms.db)")
    parser.add_argument("--seeds", default="data/seeds.csv",
                       help="Seeds CSV file (default: data/seeds.csv)")
    parser.add_argument("--out", default="data/ask_dataset.jsonl",
                       help="Output JSONL file (default: data/ask_dataset.jsonl)")
    
    args = parser.parse_args()
    
    # Extract SQLite path from URI if needed
    if args.db.startswith("sqlite:///"):
        db_path = args.db[10:]  # Remove sqlite:/// prefix
    else:
        db_path = args.db
    
    # Validate inputs
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print(f"Run: python scripts/seed_db.py --db {args.db} --reset")
        sys.exit(1)
    
    if not Path(args.seeds).exists():
        print(f"❌ Seeds file not found: {args.seeds}")
        sys.exit(1)
    
    print(f"📁 Database: {db_path}")
    print(f"📁 Seeds: {args.seeds}")
    print(f"📁 Output: {args.out}")
    print()
    
    make_dataset(db_path, args.seeds, args.out)


if __name__ == "__main__":
    main()
