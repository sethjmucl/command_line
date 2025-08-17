#!/usr/bin/env python3
"""
Seed a SQLite database with realistic UK primary care data.

Usage:
    python scripts/seed_db.py --db sqlite:///data/pms.db --seed 42 --anchor 2025-08-01 --reset
"""

import argparse
import sqlite3
import random
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import List, Tuple, Optional
import os

# Try to import faker, with helpful error if not available
try:
    from faker import Faker
    from faker.providers import phone_number, internet
except ImportError:
    print("Error: faker not installed. Run: pip install faker")
    sys.exit(1)


class DatabaseSeeder:
    """Seed a primary care management system database with realistic UK data."""
    
    def __init__(self, db_path: str, seed: int, anchor_date: str):
        self.db_path = db_path
        self.seed = seed
        self.anchor_date = datetime.fromisoformat(anchor_date)
        
        # Set up faker with UK locale and fixed seed
        self.fake = Faker('en_GB')
        Faker.seed(seed)
        random.seed(seed)
        
        # Constants for data generation
        self.INSURERS = ['Bupa', 'AXA', 'Aviva']
        self.PRACTITIONER_SURNAMES = ['Khan', 'Ng', 'Patel', 'Smith', 'Jones', 'Williams', 'Brown', 'Taylor']
        self.ROLES = ['GP', 'GP', 'GP', 'Nurse Practitioner', 'Practice Nurse']  # GP weighted higher
        
        # Appointment scheduling constraints
        self.WEEKDAYS = [0, 1, 2, 3, 4]  # Monday=0 to Friday=4
        self.CORE_START = time(9, 0)      # 09:00
        self.CORE_END = time(17, 0)       # 17:00  
        self.LUNCH_START = time(12, 30)   # 12:30
        self.LUNCH_END = time(13, 30)     # 13:30
        self.EVENING_END = time(19, 0)    # 19:00
        self.SLOT_DURATIONS = [10, 15, 30]  # minutes
        
        # Business rules
        self.DNA_RATE = 0.06          # 6% DNA rate
        self.CANCELLED_RATE = 0.04    # 4% cancellation rate
        self.INVOICE_PROBABILITY = {
            'booked': 0.80,
            'dna': 0.10,
            'cancelled': 0.00
        }
        
    def connect(self) -> sqlite3.Connection:
        """Create database connection with foreign key support."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def create_schema(self, conn: sqlite3.Connection) -> None:
        """Create all tables, views, and constraints."""
        
        # Drop existing tables if reset
        drop_statements = [
            "DROP VIEW IF EXISTS vw_free_double_slots_next_7d",
            "DROP VIEW IF EXISTS vw_aged_receivables", 
            "DROP VIEW IF EXISTS vw_no_shows_last_7d",
            "DROP TABLE IF EXISTS payments",
            "DROP TABLE IF EXISTS invoices",
            "DROP TABLE IF EXISTS documents",
            "DROP TABLE IF EXISTS referrals",
            "DROP TABLE IF EXISTS appointments",
            "DROP TABLE IF EXISTS practitioners",
            "DROP TABLE IF EXISTS patients",
            "DROP TABLE IF EXISTS insurers",
            "DROP TABLE IF EXISTS meta"
        ]
        
        for stmt in drop_statements:
            conn.execute(stmt)
        
        # Create tables
        schema_statements = [
            # Meta table for deterministic date references
            """
            CREATE TABLE meta (
                ref_date TEXT PRIMARY KEY
            )
            """,
            
            # Insurers table
            """
            CREATE TABLE insurers (
                name TEXT PRIMARY KEY
            )
            """,
            
            # Patients table  
            """
            CREATE TABLE patients (
                patient_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                dob DATE NOT NULL,
                phone TEXT,
                email TEXT,
                insurer TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (insurer) REFERENCES insurers(name)
            )
            """,
            
            # Practitioners table
            """
            CREATE TABLE practitioners (
                practitioner_id INTEGER PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                role TEXT NOT NULL
            )
            """,
            
            # Appointments table
            """
            CREATE TABLE appointments (
                appt_id INTEGER PRIMARY KEY,
                patient_id INTEGER NULL,
                practitioner_id INTEGER NOT NULL,
                start_ts TIMESTAMP NOT NULL,
                duration_minutes INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('free','booked','cancelled','dna')),
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (practitioner_id) REFERENCES practitioners(practitioner_id)
            )
            """,
            
            # Invoices table
            """
            CREATE TABLE invoices (
                invoice_id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                insurer TEXT NOT NULL,
                appointment_id INTEGER NULL,
                amount NUMERIC NOT NULL,
                issued_at TIMESTAMP NOT NULL,
                due_at TIMESTAMP NOT NULL,
                status TEXT NOT NULL DEFAULT 'outstanding',
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (insurer) REFERENCES insurers(name),
                FOREIGN KEY (appointment_id) REFERENCES appointments(appt_id)
            )
            """,
            
            # Payments table
            """
            CREATE TABLE payments (
                payment_id INTEGER PRIMARY KEY,
                invoice_id INTEGER NOT NULL,
                amount NUMERIC NOT NULL,
                paid_at TIMESTAMP NOT NULL,
                method TEXT NOT NULL,
                FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
            )
            """,
            
            # Referrals table
            """
            CREATE TABLE referrals (
                referral_id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                to_specialty TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """,
            
            # Documents table
            """
            CREATE TABLE documents (
                doc_id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
            """
        ]
        
        for stmt in schema_statements:
            conn.execute(stmt)
            
        # Insert reference date into meta table
        conn.execute("INSERT INTO meta (ref_date) VALUES (?)", (self.anchor_date.isoformat(),))
        
        # Create KPI views
        view_statements = [
            """
            CREATE VIEW vw_no_shows_last_7d AS
            SELECT * FROM appointments
            WHERE status='dna'
              AND start_ts >= datetime((SELECT ref_date FROM meta), '-7 day')
              AND start_ts <  datetime((SELECT ref_date FROM meta), '+1 day')
            """,
            
            """
            CREATE VIEW vw_aged_receivables AS
            SELECT 
                i.insurer,
                COUNT(*) AS n_invoices,
                SUM(i.amount) AS total,
                SUM(i.amount - IFNULL(p.amount, 0)) AS outstanding
            FROM invoices i
            LEFT JOIN (
                SELECT invoice_id, SUM(amount) AS amount 
                FROM payments 
                GROUP BY invoice_id
            ) p USING(invoice_id)
            GROUP BY i.insurer
            """,
            
            """
            CREATE VIEW vw_free_double_slots_next_7d AS
            SELECT * FROM appointments
            WHERE status='free'
              AND duration_minutes >= 30
              AND start_ts >= datetime((SELECT ref_date FROM meta))
              AND start_ts <  datetime((SELECT ref_date FROM meta), '+7 day')
            ORDER BY start_ts
            """
        ]
        
        for stmt in view_statements:
            conn.execute(stmt)
        
        conn.commit()
        print(f"✓ Schema created with anchor date {self.anchor_date.date()}")
    
    def seed_insurers(self, conn: sqlite3.Connection) -> None:
        """Seed the insurers table."""
        for insurer in self.INSURERS:
            conn.execute("INSERT INTO insurers (name) VALUES (?)", (insurer,))
        print(f"✓ Seeded {len(self.INSURERS)} insurers")
    
    def seed_practitioners(self, conn: sqlite3.Connection) -> int:
        """Seed practitioners table and return count."""
        practitioners = []
        target_count = random.randint(8, 10)  # 8-10 practitioners
        for i in range(target_count):
            first_name = self.fake.first_name()
            last_name = random.choice(self.PRACTITIONER_SURNAMES)
            role = random.choice(self.ROLES)
            practitioners.append((first_name, last_name, role))
        
        conn.executemany(
            "INSERT INTO practitioners (first_name, last_name, role) VALUES (?, ?, ?)",
            practitioners
        )
        
        count = len(practitioners)
        print(f"✓ Seeded {count} practitioners")
        return count
    
    def seed_patients(self, conn: sqlite3.Connection) -> int:
        """Seed patients table and return count."""
        patients = []
        target_count = random.randint(3000, 5000)
        
        for _ in range(target_count):
            # Generate realistic UK patient data
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            dob = self.fake.date_of_birth(minimum_age=0, maximum_age=100)
            phone = self.fake.phone_number()
            email = self.fake.email()
            insurer = random.choice(self.INSURERS)
            created_at = self.fake.date_time_between(
                start_date=self.anchor_date - timedelta(days=365*2),
                end_date=self.anchor_date
            )
            
            patients.append((first_name, last_name, dob, phone, email, insurer, created_at))
        
        conn.executemany(
            """INSERT INTO patients 
               (first_name, last_name, dob, phone, email, insurer, created_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            patients
        )
        
        count = len(patients)
        print(f"✓ Seeded {count} patients")
        return count
    
    def generate_appointment_slots(self, practitioner_count: int) -> List[Tuple]:
        """Generate realistic appointment slots over ±90 days."""
        appointments = []
        
        # Generate for ±90 days around anchor
        start_date = self.anchor_date - timedelta(days=90)
        end_date = self.anchor_date + timedelta(days=90)
        
        current_date = start_date
        while current_date <= end_date:
            # Only weekdays
            if current_date.weekday() in self.WEEKDAYS:
                appointments.extend(self._generate_day_schedule(current_date, practitioner_count))
            current_date += timedelta(days=1)
        
        return appointments
    
    def _generate_day_schedule(self, date: datetime, practitioner_count: int) -> List[Tuple]:
        """Generate appointment slots for a single day."""
        day_appointments = []
        
        for practitioner_id in range(1, practitioner_count + 1):
            # Core hours (09:00-17:00 minus lunch)
            current_time = datetime.combine(date.date(), self.CORE_START)
            end_time = datetime.combine(date.date(), self.CORE_END)
            lunch_start = datetime.combine(date.date(), self.LUNCH_START)
            lunch_end = datetime.combine(date.date(), self.LUNCH_END)
            
            while current_time < end_time:
                # Skip lunch break
                if lunch_start <= current_time < lunch_end:
                    current_time = lunch_end
                    continue
                
                duration = random.choice(self.SLOT_DURATIONS)
                
                # Ensure slot doesn't run past end time or into lunch
                slot_end = current_time + timedelta(minutes=duration)
                if slot_end > end_time or (current_time < lunch_start < slot_end):
                    break
                
                day_appointments.append((
                    None,  # patient_id (filled later)
                    practitioner_id,
                    current_time,
                    duration,
                    'free'  # status (updated later)
                ))
                
                current_time = slot_end
            
            # Evening slots (some days only)
            if random.random() < 0.4:  # 2 days per week average
                evening_start = datetime.combine(date.date(), self.CORE_END)
                evening_end = datetime.combine(date.date(), self.EVENING_END)
                
                current_time = evening_start
                while current_time < evening_end:
                    duration = random.choice(self.SLOT_DURATIONS)
                    slot_end = current_time + timedelta(minutes=duration)
                    if slot_end > evening_end:
                        break
                    
                    day_appointments.append((
                        None,
                        practitioner_id,
                        current_time,
                        duration,
                        'free'
                    ))
                    
                    current_time = slot_end
        
        return day_appointments
    
    def seed_appointments(self, conn: sqlite3.Connection, patient_count: int, practitioner_count: int) -> int:
        """Seed appointments with realistic booking patterns."""
        
        # Generate all possible slots
        appointment_slots = self.generate_appointment_slots(practitioner_count)
        
        # Determine how many to book (targeting 12-15k total with realistic booking rate)
        total_slots = len(appointment_slots)
        booking_rate = min(0.85, 15000 / total_slots)  # Cap at 85% utilization
        n_booked = int(total_slots * booking_rate)
        
        # Randomly select which slots to book
        slots_to_book = random.sample(range(total_slots), n_booked)
        
        # Assign patients to booked slots and set status
        for i, slot in enumerate(appointment_slots):
            if i in slots_to_book:
                # Book this slot
                patient_id = random.randint(1, patient_count)
                
                # Determine status based on whether it's past anchor date
                if slot[2] < self.anchor_date:  # Past appointments
                    rand = random.random()
                    if rand < self.DNA_RATE:
                        status = 'dna'
                    elif rand < self.DNA_RATE + self.CANCELLED_RATE:
                        status = 'cancelled'
                    else:
                        status = 'booked'
                else:  # Future appointments
                    status = 'booked'
                
                appointment_slots[i] = (patient_id, slot[1], slot[2], slot[3], status)
            # else: keep as free slot (patient_id=None, status='free')
        
        # Insert all appointments
        conn.executemany(
            """INSERT INTO appointments 
               (patient_id, practitioner_id, start_ts, duration_minutes, status) 
               VALUES (?, ?, ?, ?, ?)""",
            appointment_slots
        )
        
        count = len(appointment_slots)
        print(f"✓ Seeded {count} appointments ({n_booked} booked, {count-n_booked} free)")
        return count
    
    def seed_invoices_and_payments(self, conn: sqlite3.Connection) -> Tuple[int, int]:
        """Generate invoices for appointments with realistic payment patterns."""
        
        # Get all booked and DNA appointments
        cursor = conn.execute("""
            SELECT appt_id, patient_id, status, start_ts
            FROM appointments 
            WHERE status IN ('booked', 'dna', 'cancelled')
        """)
        appointments = cursor.fetchall()
        
        invoices = []
        payments = []
        invoice_id = 1
        payment_id = 1
        
        for appt_id, patient_id, status, start_ts in appointments:
            # Determine if this appointment generates an invoice
            if random.random() < self.INVOICE_PROBABILITY[status]:
                
                # Get patient's insurer
                cursor = conn.execute("SELECT insurer FROM patients WHERE patient_id = ?", (patient_id,))
                insurer = cursor.fetchone()[0]
                
                # Generate invoice amount
                if status == 'dna':
                    amount = random.uniform(10, 25)  # DNA fee
                else:
                    amount = random.triangular(40, 180, 90)  # Regular appointment
                
                # Calculate issue date (lag from appointment)
                appt_date = datetime.fromisoformat(start_ts)
                issue_lag_days = random.choices([0, 1, 3, 7], weights=[0.5, 0.2, 0.2, 0.1])[0]
                issued_at = appt_date + timedelta(days=issue_lag_days)
                due_at = issued_at + timedelta(days=30)
                
                invoices.append((
                    invoice_id, patient_id, insurer, appt_id, round(amount, 2),
                    issued_at, due_at, 'outstanding'
                ))
                
                # Generate payment (if any)
                payment_lag_days = random.choices([0, 7, 30], weights=[0.5, 0.3, 0.2])[0]
                
                # Self-pay tends toward faster payment
                if insurer in ['Self-pay', 'Private']:  # Adjust if needed
                    payment_lag_days = random.choices([0, 7, 30], weights=[0.7, 0.2, 0.1])[0]
                
                # 80% of invoices get paid
                if random.random() < 0.8:
                    paid_at = issued_at + timedelta(days=payment_lag_days)
                    method = random.choice(['Card', 'BACS', 'Cheque', 'Cash'])
                    
                    # Usually full payment, occasionally partial
                    payment_amount = amount if random.random() < 0.95 else amount * random.uniform(0.5, 0.9)
                    
                    payments.append((
                        payment_id, invoice_id, round(payment_amount, 2), paid_at, method
                    ))
                    payment_id += 1
                
                invoice_id += 1
        
        # Insert invoices and payments
        conn.executemany(
            """INSERT INTO invoices 
               (invoice_id, patient_id, insurer, appointment_id, amount, issued_at, due_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            invoices
        )
        
        conn.executemany(
            """INSERT INTO payments
               (payment_id, invoice_id, amount, paid_at, method)
               VALUES (?, ?, ?, ?, ?)""",
            payments
        )
        
        print(f"✓ Seeded {len(invoices)} invoices and {len(payments)} payments")
        return len(invoices), len(payments)
    
    def seed_referrals_and_documents(self, conn: sqlite3.Connection, patient_count: int) -> Tuple[int, int]:
        """Generate some referrals and documents for completeness."""
        
        # Generate referrals (small number)
        specialties = ['Cardiology', 'Dermatology', 'Orthopaedics', 'Neurology', 'Psychiatry']
        referrals = []
        
        for i in range(random.randint(50, 150)):
            patient_id = random.randint(1, patient_count)
            specialty = random.choice(specialties)
            created_at = self.fake.date_time_between(
                start_date=self.anchor_date - timedelta(days=90),
                end_date=self.anchor_date
            )
            status = random.choice(['pending', 'sent', 'accepted', 'declined'])
            
            referrals.append((patient_id, specialty, created_at, status))
        
        # Generate documents
        doc_types = ['Letter', 'Lab Result', 'X-Ray', 'Prescription', 'Referral Letter']
        documents = []
        
        for i in range(random.randint(200, 500)):
            patient_id = random.randint(1, patient_count)
            kind = random.choice(doc_types)
            created_at = self.fake.date_time_between(
                start_date=self.anchor_date - timedelta(days=180),
                end_date=self.anchor_date
            )
            
            documents.append((patient_id, kind, created_at))
        
        # Insert data
        conn.executemany(
            "INSERT INTO referrals (patient_id, to_specialty, created_at, status) VALUES (?, ?, ?, ?)",
            referrals
        )
        
        conn.executemany(
            "INSERT INTO documents (patient_id, kind, created_at) VALUES (?, ?, ?)",
            documents
        )
        
        print(f"✓ Seeded {len(referrals)} referrals and {len(documents)} documents")
        return len(referrals), len(documents)
    
    def run_sanity_checks(self, conn: sqlite3.Connection) -> None:
        """Run basic sanity checks on generated data."""
        print("\n--- Sanity Checks ---")
        
        # Check appointment time constraints
        cursor = conn.execute("""
            SELECT COUNT(*) FROM appointments 
            WHERE time(start_ts) < '06:00' OR time(start_ts) > '21:00'
        """)
        invalid_times = cursor.fetchone()[0]
        print(f"✓ Appointments outside 06:00-21:00: {invalid_times} (should be 0)")
        
        # Check DNA rate
        cursor = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='dna'")
        dna_count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='booked'")
        booked_count = cursor.fetchone()[0]
        dna_rate = dna_count / (dna_count + booked_count) if (dna_count + booked_count) > 0 else 0
        print(f"✓ DNA rate: {dna_rate:.1%} (target: ~6%)")
        
        # Check payment amounts don't exceed invoices
        cursor = conn.execute("""
            SELECT COUNT(*) FROM payments p
            JOIN invoices i ON p.invoice_id = i.invoice_id
            WHERE p.amount > i.amount
        """)
        overpayments = cursor.fetchone()[0]
        print(f"✓ Overpayments: {overpayments} (should be 0)")
        
        # Check view functionality
        cursor = conn.execute("SELECT COUNT(*) FROM vw_no_shows_last_7d")
        no_shows = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM vw_free_double_slots_next_7d")
        free_slots = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM vw_aged_receivables")
        receivables = cursor.fetchone()[0]
        
        print(f"✓ No-shows last 7d: {no_shows}")
        print(f"✓ Free double slots next 7d: {free_slots}")
        print(f"✓ Aged receivables entries: {receivables}")
    
    def seed_all(self, reset: bool = False) -> None:
        """Main seeding orchestrator."""
        conn = self.connect()
        
        try:
            if reset:
                self.create_schema(conn)
            
            self.seed_insurers(conn)
            practitioner_count = self.seed_practitioners(conn)
            patient_count = self.seed_patients(conn)
            self.seed_appointments(conn, patient_count, practitioner_count)
            self.seed_invoices_and_payments(conn)
            self.seed_referrals_and_documents(conn, patient_count)
            
            conn.commit()
            self.run_sanity_checks(conn)
            
            print(f"\n✅ Database seeded successfully: {self.db_path}")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Error seeding database: {e}")
            raise
        finally:
            conn.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed UK primary care database")
    parser.add_argument("--db", default="sqlite:///data/pms.db", 
                       help="Database path (default: sqlite:///data/pms.db)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--anchor", default="2025-08-01",
                       help="Anchor date for time-based data (default: 2025-08-01)")
    parser.add_argument("--reset", action="store_true",
                       help="Drop and recreate all tables")
    
    args = parser.parse_args()
    
    # Extract SQLite path from URI
    if args.db.startswith("sqlite:///"):
        db_path = args.db[10:]  # Remove sqlite:/// prefix
    else:
        db_path = args.db
    
    # Validate anchor date
    try:
        datetime.fromisoformat(args.anchor)
    except ValueError:
        print(f"Error: Invalid anchor date '{args.anchor}'. Use ISO format (YYYY-MM-DD)")
        sys.exit(1)
    
    print(f"Seeding database: {db_path}")
    print(f"Random seed: {args.seed}")
    print(f"Anchor date: {args.anchor}")
    print(f"Reset schema: {args.reset}")
    print()
    
    seeder = DatabaseSeeder(db_path, args.seed, args.anchor)
    seeder.seed_all(reset=args.reset)


if __name__ == "__main__":
    main()
