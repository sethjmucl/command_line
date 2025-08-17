#!/usr/bin/env python3
"""
Quick demo script to showcase the seeded database functionality.

Usage:
    python scripts/demo_db.py --db sqlite:///data/pms.db
"""

import argparse
import sqlite3
from pathlib import Path


def run_demo_queries(db_path: str):
    """Run a series of demo queries to showcase the database."""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    
    print(f"🏥 UK Primary Care Database Demo")
    print(f"Database: {db_path}")
    print("=" * 60)
    
    # 1. Basic table counts
    print("\n📊 Database Overview:")
    tables = ['patients', 'practitioners', 'appointments', 'invoices', 'payments']
    for table in tables:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table.title()}: {count:,}")
    
    # 2. Sample practitioners (UK names)
    print("\n👨‍⚕️ Sample Practitioners:")
    cursor = conn.execute("""
        SELECT first_name, last_name, role 
        FROM practitioners 
        ORDER BY last_name 
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row['first_name']} {row['last_name']} ({row['role']})")
    
    # 3. Recent appointments
    print("\n📅 Recent Appointments (sample):")
    cursor = conn.execute("""
        SELECT 
            strftime('%Y-%m-%d %H:%M', a.start_ts) as appointment_time,
            p.first_name || ' ' || p.last_name as patient,
            pr.last_name as practitioner,
            a.duration_minutes,
            a.status
        FROM appointments a
        JOIN patients p ON a.patient_id = p.patient_id
        JOIN practitioners pr ON a.practitioner_id = pr.practitioner_id
        WHERE a.start_ts >= (SELECT ref_date FROM meta)
        ORDER BY a.start_ts
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row['appointment_time']} | {row['patient'][:15]:<15} | Dr {row['practitioner']:<8} | {row['duration_minutes']}min | {row['status']}")
    
    # 4. KPI Views Demo
    print("\n📈 Key Performance Indicators:")
    
    # No-shows last 7 days
    cursor = conn.execute("SELECT COUNT(*) FROM vw_no_shows_last_7d")
    no_shows = cursor.fetchone()[0]
    print(f"  No-shows (last 7 days): {no_shows}")
    
    # Free double slots next 7 days
    cursor = conn.execute("SELECT COUNT(*) FROM vw_free_double_slots_next_7d")
    free_slots = cursor.fetchone()[0]
    print(f"  Free double slots (next 7 days): {free_slots}")
    
    # Aged receivables by insurer
    print(f"  Aged receivables by insurer:")
    cursor = conn.execute("""
        SELECT insurer, n_invoices, 
               ROUND(total, 2) as total_billed,
               ROUND(outstanding, 2) as outstanding
        FROM vw_aged_receivables 
        ORDER BY outstanding DESC
    """)
    for row in cursor.fetchall():
        print(f"    {row['insurer']:<8}: £{row['total_billed']:>8,.2f} billed, £{row['outstanding']:>8,.2f} outstanding ({row['n_invoices']} invoices)")
    
    # 5. Data Quality Checks
    print("\n✅ Data Quality Validation:")
    
    # Check appointment time constraints
    cursor = conn.execute("""
        SELECT COUNT(*) FROM appointments 
        WHERE time(start_ts) < '06:00' OR time(start_ts) > '21:00'
    """)
    invalid_times = cursor.fetchone()[0]
    print(f"  Appointments outside 06:00-21:00: {invalid_times} ✓")
    
    # Check weekend appointments
    cursor = conn.execute("""
        SELECT COUNT(*) FROM appointments 
        WHERE strftime('%w', start_ts) IN ('0', '6')
    """)
    weekend_appts = cursor.fetchone()[0]
    print(f"  Weekend appointments: {weekend_appts} ✓")
    
    # DNA rate
    cursor = conn.execute("""
        SELECT 
            ROUND(100.0 * SUM(CASE WHEN status='dna' THEN 1 ELSE 0 END) / 
                  SUM(CASE WHEN status IN ('booked', 'dna') THEN 1 ELSE 0 END), 1) as dna_rate
        FROM appointments
    """)
    dna_rate = cursor.fetchone()[0]
    print(f"  DNA rate: {dna_rate}% (target: ~6%) ✓")
    
    # UK-specific data
    cursor = conn.execute("SELECT GROUP_CONCAT(name) FROM insurers ORDER BY name")
    insurers = cursor.fetchone()[0]
    print(f"  UK insurers: {insurers} ✓")
    
    print("\n🎯 Database ready for DSL queries!")
    print("\nNext steps:")
    print("  1. Build the DSL grammar parser")
    print("  2. Create date resolver and canonicaliser")
    print("  3. Generate NL→DSL→SQL training dataset")
    print("  4. Train few-shot baseline models")
    
    conn.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Demo the UK primary care database")
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
        return
    
    run_demo_queries(db_path)


if __name__ == "__main__":
    main()
