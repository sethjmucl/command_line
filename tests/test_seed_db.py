#!/usr/bin/env python3
"""
Test suite for database seeding functionality.

Usage:
    python -m pytest tests/test_seed_db.py -v
    or 
    python tests/test_seed_db.py
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.seed_db import DatabaseSeeder


class TestDatabaseSeeder(unittest.TestCase):
    """Test the database seeding functionality."""
    
    def setUp(self):
        """Set up test database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_path = self.temp_db.name
        self.seeder = DatabaseSeeder(self.db_path, seed=42, anchor_date="2025-08-01")
        
        # Create and seed the database
        self.seeder.seed_all(reset=True)
        
        # Connect for testing
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def tearDown(self):
        """Clean up after each test."""
        self.conn.close()
        Path(self.db_path).unlink(missing_ok=True)
    
    def test_schema_exists(self):
        """Test that all tables and views are created."""
        # Check tables exist
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'appointments', 'documents', 'insurers', 'invoices', 
            'meta', 'patients', 'payments', 'practitioners', 'referrals'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found")
        
        # Check views exist
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='view' 
            ORDER BY name
        """)
        views = [row[0] for row in cursor.fetchall()]
        
        expected_views = [
            'vw_aged_receivables',
            'vw_free_double_slots_next_7d', 
            'vw_no_shows_last_7d'
        ]
        
        for view in expected_views:
            self.assertIn(view, views, f"View {view} not found")
    
    def test_meta_table(self):
        """Test meta table has correct reference date."""
        cursor = self.conn.execute("SELECT ref_date FROM meta")
        ref_date = cursor.fetchone()[0]
        self.assertEqual(ref_date, "2025-08-01T00:00:00")
    
    def test_row_counts_reasonable(self):
        """Test that seeded data has reasonable row counts."""
        # Define expected ranges
        expected_ranges = {
            'patients': (3000, 5000),
            'practitioners': (8, 10),
            'appointments': (10000, 25000),  # Should be substantial (180 days * 10 practitioners)
            'insurers': (3, 3),  # Exactly 3
            'invoices': (1, 15000),  # Should have some invoices
            'payments': (1, 12000),   # Most invoices should be paid
            'referrals': (50, 150),
            'documents': (200, 500)
        }
        
        for table, (min_count, max_count) in expected_ranges.items():
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            self.assertGreaterEqual(count, min_count, 
                                  f"{table} has {count} rows, expected >= {min_count}")
            self.assertLessEqual(count, max_count,
                               f"{table} has {count} rows, expected <= {max_count}")
    
    def test_appointment_time_constraints(self):
        """Test no appointments outside 06:00-21:00."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM appointments 
            WHERE time(start_ts) < '06:00' OR time(start_ts) > '21:00'
        """)
        invalid_count = cursor.fetchone()[0]
        self.assertEqual(invalid_count, 0, "Found appointments outside allowed hours")
    
    def test_appointment_weekdays_only(self):
        """Test appointments are only on weekdays."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM appointments 
            WHERE strftime('%w', start_ts) IN ('0', '6')
        """)
        weekend_count = cursor.fetchone()[0]
        self.assertEqual(weekend_count, 0, "Found appointments on weekends")
    
    def test_appointment_status_distribution(self):
        """Test appointment status distribution is reasonable."""
        cursor = self.conn.execute("""
            SELECT status, COUNT(*) 
            FROM appointments 
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Should have all expected statuses
        expected_statuses = {'free', 'booked', 'cancelled', 'dna'}
        for status in expected_statuses:
            self.assertIn(status, status_counts, f"Status {status} not found")
        
        # DNA rate should be reasonable (3-10%)
        total_completed = status_counts.get('booked', 0) + status_counts.get('dna', 0)
        if total_completed > 0:
            dna_rate = status_counts.get('dna', 0) / total_completed
            self.assertGreaterEqual(dna_rate, 0.03, "DNA rate too low")
            self.assertLessEqual(dna_rate, 0.10, "DNA rate too high")
    
    def test_foreign_key_constraints(self):
        """Test foreign key relationships are valid."""
        # Test patients.insurer -> insurers.name
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM patients p
            LEFT JOIN insurers i ON p.insurer = i.name
            WHERE i.name IS NULL
        """)
        orphaned_patients = cursor.fetchone()[0]
        self.assertEqual(orphaned_patients, 0, "Found patients with invalid insurer references")
        
        # Test appointments.patient_id -> patients.patient_id
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM appointments a
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.patient_id IS NOT NULL AND p.patient_id IS NULL
        """)
        orphaned_appointments = cursor.fetchone()[0]
        self.assertEqual(orphaned_appointments, 0, "Found appointments with invalid patient references")
        
        # Test payments.invoice_id -> invoices.invoice_id
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM payments p
            LEFT JOIN invoices i ON p.invoice_id = i.invoice_id
            WHERE i.invoice_id IS NULL
        """)
        orphaned_payments = cursor.fetchone()[0]
        self.assertEqual(orphaned_payments, 0, "Found payments with invalid invoice references")
    
    def test_aged_receivables_view(self):
        """Test aged receivables view has expected structure and data quality."""
        cursor = self.conn.execute("SELECT * FROM vw_aged_receivables LIMIT 1")
        columns = [description[0] for description in cursor.description]
        
        expected_columns = {'insurer', 'n_invoices', 'total', 'outstanding'}
        self.assertEqual(set(columns), expected_columns, 
                        "Aged receivables view has wrong columns")
        
        # Test no negative outstanding amounts
        cursor = self.conn.execute("SELECT COUNT(*) FROM vw_aged_receivables WHERE outstanding < 0")
        negative_outstanding = cursor.fetchone()[0]
        self.assertEqual(negative_outstanding, 0, "Found negative outstanding amounts")
        
        # Test all insurers are represented
        cursor = self.conn.execute("SELECT COUNT(DISTINCT insurer) FROM vw_aged_receivables")
        insurer_count = cursor.fetchone()[0]
        self.assertGreaterEqual(insurer_count, 1, "No insurers in aged receivables view")
    
    def test_no_shows_view(self):
        """Test no-shows view returns reasonable data."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM vw_no_shows_last_7d")
        no_show_count = cursor.fetchone()[0]
        
        # Should be non-negative
        self.assertGreaterEqual(no_show_count, 0, "No-shows count cannot be negative")
        
        # All entries should be DNA status
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM vw_no_shows_last_7d 
            WHERE status != 'dna'
        """)
        non_dna_count = cursor.fetchone()[0]
        self.assertEqual(non_dna_count, 0, "Found non-DNA appointments in no-shows view")
    
    def test_free_double_slots_view(self):
        """Test free double slots view returns reasonable data."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM vw_free_double_slots_next_7d")
        free_slot_count = cursor.fetchone()[0]
        
        # Should be non-negative
        self.assertGreaterEqual(free_slot_count, 0, "Free slots count cannot be negative")
        
        # All entries should be free status and >= 30 minutes
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM vw_free_double_slots_next_7d 
            WHERE status != 'free' OR duration_minutes < 30
        """)
        invalid_slots = cursor.fetchone()[0]
        self.assertEqual(invalid_slots, 0, "Found invalid entries in free double slots view")
    
    def test_invoice_payment_logic(self):
        """Test invoice and payment generation follows business rules."""
        # Test no payments exceed invoice amounts
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM payments p
            JOIN invoices i ON p.invoice_id = i.invoice_id
            WHERE p.amount > i.amount + 0.01
        """)
        overpayments = cursor.fetchone()[0]
        self.assertEqual(overpayments, 0, "Found payments exceeding invoice amounts")
        
        # Test invoices have reasonable amounts
        cursor = self.conn.execute("SELECT MIN(amount), MAX(amount) FROM invoices")
        min_amount, max_amount = cursor.fetchone()
        self.assertGreaterEqual(min_amount, 5, "Invoice amount too low")
        self.assertLessEqual(max_amount, 200, "Invoice amount too high")
        
        # Test payment methods are valid
        cursor = self.conn.execute("SELECT DISTINCT method FROM payments")
        methods = [row[0] for row in cursor.fetchall()]
        valid_methods = {'Card', 'BACS', 'Cheque', 'Cash'}
        for method in methods:
            self.assertIn(method, valid_methods, f"Invalid payment method: {method}")
    
    def test_uk_specific_data(self):
        """Test UK-specific data elements are present."""
        # Test practitioner surnames include expected UK names
        cursor = self.conn.execute("SELECT DISTINCT last_name FROM practitioners")
        surnames = [row[0] for row in cursor.fetchall()]
        
        expected_surnames = {'Khan', 'Ng', 'Patel', 'Smith', 'Jones'}
        found_expected = any(surname in surnames for surname in expected_surnames)
        self.assertTrue(found_expected, "Expected UK surnames not found in practitioners")
        
        # Test insurers are UK-specific
        cursor = self.conn.execute("SELECT name FROM insurers ORDER BY name")
        insurers = [row[0] for row in cursor.fetchall()]
        expected_insurers = ['AXA', 'Aviva', 'Bupa']
        self.assertEqual(insurers, expected_insurers, "Incorrect UK insurers")
    
    def test_deterministic_seeding(self):
        """Test that seeding with same parameters produces identical results."""
        # Create second database with same parameters
        temp_db2 = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db2.close()
        
        try:
            seeder2 = DatabaseSeeder(temp_db2.name, seed=42, anchor_date="2025-08-01")
            seeder2.seed_all(reset=True)
            
            conn2 = sqlite3.connect(temp_db2.name)
            
            # Compare row counts
            tables = ['patients', 'practitioners', 'appointments', 'invoices', 'payments']
            for table in tables:
                cursor1 = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                count1 = cursor1.fetchone()[0]
                
                cursor2 = conn2.execute(f"SELECT COUNT(*) FROM {table}")
                count2 = cursor2.fetchone()[0]
                
                self.assertEqual(count1, count2, f"Row count mismatch in {table}")
            
            conn2.close()
            
        finally:
            Path(temp_db2.name).unlink(missing_ok=True)
    
    def test_data_directory_creation(self):
        """Test that data directory is created if it doesn't exist."""
        # This is tested implicitly by the temp file creation,
        # but let's verify the seeder handles directory creation
        non_existent_path = "/tmp/test_nonexistent_dir/test.db"
        
        # Clean up any existing test directory
        import shutil
        test_dir = Path("/tmp/test_nonexistent_dir")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        try:
            seeder = DatabaseSeeder(non_existent_path, seed=42, anchor_date="2025-08-01")
            conn = seeder.connect()
            conn.close()
            
            # Verify directory was created
            self.assertTrue(test_dir.exists(), "Directory not created")
            self.assertTrue(Path(non_existent_path).exists(), "Database file not created")
            
        finally:
            # Clean up
            if test_dir.exists():
                shutil.rmtree(test_dir)


def run_tests():
    """Run all tests."""
    # Check for required dependencies
    try:
        import faker
    except ImportError:
        print("Error: faker not installed. Run: pip install faker")
        return False
    
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
