#!/usr/bin/env python3
"""
Expand an existing JSONL dataset with programmatic NL variants and practitioner swaps.

- Loads an existing JSONL dataset with fields: instruction, dsl, persona, tier
- Generates paraphrastic variants (typos/abbreviations/phrasing) that preserve DSL
- If practitioner filter is present, creates copies with other practitioner last names and updates DSL
- Compiles DSL → SQL, executes, and writes fresh denotation hash

Usage:
  python scripts/expand_dataset.py \
    --db sqlite:///data/pms.db \
    --in data/ask_dataset.jsonl \
    --out data/ask_dataset_expanded.jsonl \
    --variants 3
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))
from packages.dsl.compiler import compile_dsl
from scripts.make_dataset import denotation_hash


def load_practitioner_last_names(conn: sqlite3.Connection) -> List[str]:
    try:
        df = pd.read_sql_query("SELECT DISTINCT last_name FROM practitioners ORDER BY last_name", conn)
        return [str(x) for x in df['last_name'].dropna().tolist()]
    except Exception:
        return []


def generate_text_variants(nl: str) -> List[str]:
    """Generate paraphrastic NL variants that do not change semantics/DSL."""
    variants = set()
    base = nl

    # Abbreviations
    abbrev_map = [
        (r"\bappointments\b", "appts"),
        (r"\bpatients\b", "pts"),
        (r"\bno[- ]?shows\b", "DNA"),
        (r"\bdoctor(s)?\b", r"Dr\1"),
        (r"\bfree\b", "available"),
    ]
    for pat, rep in abbrev_map:
        if re.search(pat, base, flags=re.IGNORECASE):
            variants.add(re.sub(pat, rep, base, flags=re.IGNORECASE))

    # Typos (lightweight)
    typo_map = [
        (r"\bappointments\b", "appoinments"),
        (r"\bdoctors\b", "docters"),
    ]
    for pat, rep in typo_map:
        if re.search(pat, base, flags=re.IGNORECASE):
            variants.add(re.sub(pat, rep, base, flags=re.IGNORECASE))

    # Phrasing tweaks
    phr_map = [
        (r"\bshow\b", "list"),
        (r"\bnext week\b", "in the next week"),
        (r"\blast 7 days\b", "the last 7 days"),
        (r"\bafter 4pm\b", "after 16:00"),
    ]
    for pat, rep in phr_map:
        if re.search(pat, base, flags=re.IGNORECASE):
            variants.add(re.sub(pat, rep, base, flags=re.IGNORECASE))

    # Minor punctuation/noise
    if not base.endswith("."):
        variants.add(base + ".")

    # Limit the number of variants
    return list(variants)


def swap_practitioner(nl: str, dsl: str, available_last_names: List[str]) -> List[Tuple[str, str]]:
    """If DSL has practitioner='X', create copies for other surnames.

    Returns list of (new_nl, new_dsl)
    """
    m = re.search(r"practitioner='([^']+)'", dsl)
    if not m:
        return []
    old_last = m.group(1)
    candidates = [n for n in available_last_names if n.lower() != old_last.lower()]
    random.shuffle(candidates)
    candidates = candidates[:2]  # at most 2 swaps

    out = []
    for new_last in candidates:
        # Update DSL
        new_dsl = dsl.replace(f"practitioner='{old_last}'", f"practitioner='{new_last}'")
        # Update NL heuristically (replace surname occurrences)
        new_nl = re.sub(rf"\b{re.escape(old_last)}\b", new_last, nl, flags=re.IGNORECASE)
        # Prefer Dr formatting if present
        new_nl = re.sub(r"\bDr\.?\s+\w+\b", f"Dr {new_last}", new_nl)
        out.append((new_nl, new_dsl))
    return out


def expand_dataset(db_path: str, in_jsonl: str, out_jsonl: str, variants_per_item: int) -> None:
    # Connect DB
    if in_jsonl.endswith('.jsonl') and db_path.startswith('sqlite:///'):
        db_actual = db_path[10:]
    else:
        db_actual = db_path if not db_path.startswith('sqlite:///') else db_path[10:]

    conn = sqlite3.connect(db_actual)
    conn.execute('PRAGMA foreign_keys = ON')

    last_names = load_practitioner_last_names(conn)

    # Load base dataset
    base: List[Dict[str, Any]] = []
    with open(in_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            base.append(json.loads(line))

    out: List[Dict[str, Any]] = []

    for item in base:
        # Always include original
        out.append(item)

        instr = item['instruction']
        dsl = item['dsl']
        persona = item.get('persona', 'unknown')
        tier = item.get('tier', 'T1')

        # 1) Practitioner swaps (if applicable)
        for new_nl, new_dsl in swap_practitioner(instr, dsl, last_names):
            try:
                sql, params = compile_dsl(new_dsl, conn)
                df = pd.read_sql_query(sql, conn, params=params)
                out.append({
                    'instruction': new_nl,
                    'dsl': new_dsl,
                    'sql': sql,
                    'denotation': {
                        'hash': denotation_hash(df),
                        'n_rows': len(df)
                    },
                    'persona': persona,
                    'tier': tier
                })
            except Exception:
                continue

        # 2) Paraphrastic NL variants that keep DSL the same
        nl_vars = generate_text_variants(instr)
        random.shuffle(nl_vars)
        nl_vars = nl_vars[:max(0, variants_per_item)]
        if nl_vars:
            try:
                sql_base, params_base = compile_dsl(dsl, conn)
                df_base = pd.read_sql_query(sql_base, conn, params=params_base)
                hash_base = denotation_hash(df_base)
            except Exception:
                sql_base, hash_base, df_base = None, None, pd.DataFrame()
            for nl_var in nl_vars:
                out.append({
                    'instruction': nl_var,
                    'dsl': dsl,
                    'sql': sql_base,
                    'denotation': {
                        'hash': hash_base,
                        'n_rows': len(df_base)
                    },
                    'persona': persona,
                    'tier': tier
                })

    # Deduplicate (instruction + dsl)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in out:
        key = (r['instruction'], r['dsl'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # Save
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"✅ Expanded dataset saved: {out_path}")
    print(f"   Base: {len(base)}  →  Expanded: {len(deduped)}")


def main():
    ap = argparse.ArgumentParser(description='Expand dataset with programmatic variants')
    ap.add_argument('--db', required=True, help='Database path, e.g. sqlite:///data/pms.db')
    ap.add_argument('--in', dest='in_jsonl', required=True, help='Input JSONL (gold dataset)')
    ap.add_argument('--out', dest='out_jsonl', required=True, help='Output JSONL (expanded)')
    ap.add_argument('--variants', type=int, default=3, help='Paraphrase variants per item (default 3)')
    args = ap.parse_args()

    # Extract SQLite path if needed is handled in expand_dataset
    expand_dataset(args.db, args.in_jsonl, args.out_jsonl, args.variants)


if __name__ == '__main__':
    main()


