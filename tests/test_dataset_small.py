#!/usr/bin/env python3
"""
Tests for dataset generation functionality.
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
import pandas as pd
import sqlite3

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.make_dataset import make_dataset, canonicalize_df, denotation_hash


class TestDatasetGeneration:
    """Test dataset generation from seeds."""
    
    def test_canonicalize_df(self):
        """Test DataFrame canonicalization for deterministic hashing."""
        # Create test DataFrame with various data types
        df1 = pd.DataFrame({
            'id': [3, 1, 2],
            'name': ['Alice', 'Bob', 'Charlie'], 
            'value': [123.456789, 456.789, 789.123],
            'timestamp': pd.to_datetime(['2025-01-01 10:30:00', '2025-01-02 15:45:00', '2025-01-01 08:15:00'])
        })
        
        # Same data in different order
        df2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Bob', 'Charlie', 'Alice'],
            'value': [456.789, 789.123, 123.456789],
            'timestamp': pd.to_datetime(['2025-01-02 15:45:00', '2025-01-01 08:15:00', '2025-01-01 10:30:00'])
        })
        
        # Canonicalize both
        canon1 = canonicalize_df(df1)
        canon2 = canonicalize_df(df2)
        
        # Should be identical after canonicalization
        pd.testing.assert_frame_equal(canon1, canon2)
        
        # Check formatting - should be sorted by all columns
        # After sorting, first row should be id=1 (Bob's row)
        assert canon1['id'].iloc[0] == 1
        assert canon1['value'].iloc[0] == 456.79  # Rounded to 2 dp
        assert canon1['timestamp'].iloc[0] == '2025-01-02 15:45:00'  # ISO format
        
        # Check rounding worked
        assert all(canon1['value'].apply(lambda x: len(str(x).split('.')[-1]) <= 2))  # Max 2 decimal places
    
    def test_denotation_hash_deterministic(self):
        """Test that denotation hashing is deterministic."""
        # Same data, different order
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        df2 = pd.DataFrame({'a': [3, 1, 2], 'b': ['z', 'x', 'y']})
        
        hash1 = denotation_hash(df1)
        hash2 = denotation_hash(df2)
        
        assert hash1 == hash2, "Hashes should be identical for same data in different order"
        assert len(hash1) == 64, "Should be SHA256 hex digest"
    
    def test_denotation_hash_float_rounding(self):
        """Test that float rounding is stable for hashing."""
        # Slightly different float precision
        df1 = pd.DataFrame({'value': [123.456]})
        df2 = pd.DataFrame({'value': [123.4560]})  # Same value, different precision
        
        hash1 = denotation_hash(df1)
        hash2 = denotation_hash(df2)
        
        assert hash1 == hash2, "Hashes should be identical after rounding"
    
    def test_make_dataset_basic(self):
        """Test basic dataset generation functionality."""
        # Check that required files exist
        db_path = "data/pms.db"
        seeds_path = "data/seeds.csv"
        
        if not Path(db_path).exists():
            pytest.skip(f"Database not found: {db_path}")
        
        if not Path(seeds_path).exists():
            pytest.skip(f"Seeds file not found: {seeds_path}")
        
        # Generate dataset to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_output = f.name
        
        try:
            results = make_dataset(db_path, seeds_path, temp_output)
            
            # Verify file was created
            assert Path(temp_output).exists(), "Output file should be created"
            
            # Check results structure
            assert len(results) == 15, "Should have 15 entries from seeds"
            
            # Load and validate JSONL
            entries = []
            with open(temp_output, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    entries.append(entry)
            
            assert len(entries) == 15, "JSONL should have 15 lines"
            
            # Validate structure of each entry
            required_keys = {'instruction', 'dsl', 'sql', 'denotation', 'persona', 'tier'}
            for i, entry in enumerate(entries):
                assert set(entry.keys()) == required_keys, f"Entry {i} missing required keys"
                assert 'hash' in entry['denotation'], f"Entry {i} missing denotation hash"
                assert 'n_rows' in entry['denotation'], f"Entry {i} missing row count"
                assert entry['denotation']['hash'] is not None, f"Entry {i} has null hash"
                assert isinstance(entry['denotation']['n_rows'], int), f"Entry {i} row count not integer"
            
            # Verify some known patterns
            kpi_entries = [e for e in entries if 'kpi:' in e['dsl']]
            assert len(kpi_entries) >= 3, "Should have at least 3 KPI entries"
            
            appointment_entries = [e for e in entries if e['dsl'].startswith('FIND appointments')]
            assert len(appointment_entries) >= 8, "Should have several appointment queries"
            
            print(f"✅ Generated {len(entries)} valid entries")
            
        finally:
            # Clean up temp file
            Path(temp_output).unlink(missing_ok=True)
    
    def test_dataset_file_exists(self):
        """Test that the actual dataset file exists and is valid."""
        dataset_path = "data/ask_dataset.jsonl"
        
        if not Path(dataset_path).exists():
            pytest.skip(f"Dataset file not found: {dataset_path}")
        
        # Load and validate the actual dataset
        entries = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON on line {line_num}: {e}")
        
        assert len(entries) == 15, f"Expected 15 entries, got {len(entries)}"
        
        # Validate all entries have required structure
        for i, entry in enumerate(entries):
            assert 'instruction' in entry, f"Entry {i} missing instruction"
            assert 'dsl' in entry, f"Entry {i} missing dsl"
            assert 'sql' in entry, f"Entry {i} missing sql"
            assert 'denotation' in entry, f"Entry {i} missing denotation"
            assert 'persona' in entry, f"Entry {i} missing persona"
            assert 'tier' in entry, f"Entry {i} missing tier"
            
            denotation = entry['denotation']
            assert 'hash' in denotation, f"Entry {i} missing denotation.hash"
            assert 'n_rows' in denotation, f"Entry {i} missing denotation.n_rows"
        
        print(f"✅ Validated {len(entries)} entries in {dataset_path}")
    
    def test_deterministic_dataset_generation(self):
        """Test that dataset generation is deterministic."""
        db_path = "data/pms.db"
        seeds_path = "data/seeds.csv"
        
        if not Path(db_path).exists() or not Path(seeds_path).exists():
            pytest.skip("Required files not found")
        
        # Generate dataset twice to temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f1:
            temp_output1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f2:
            temp_output2 = f2.name
        
        try:
            results1 = make_dataset(db_path, seeds_path, temp_output1)
            results2 = make_dataset(db_path, seeds_path, temp_output2)
            
            # Compare results
            assert len(results1) == len(results2), "Should generate same number of entries"
            
            for i, (r1, r2) in enumerate(zip(results1, results2)):
                assert r1['instruction'] == r2['instruction'], f"Entry {i} instruction differs"
                assert r1['dsl'] == r2['dsl'], f"Entry {i} DSL differs"
                assert r1['sql'] == r2['sql'], f"Entry {i} SQL differs"
                assert r1['denotation']['hash'] == r2['denotation']['hash'], f"Entry {i} hash differs"
                assert r1['denotation']['n_rows'] == r2['denotation']['n_rows'], f"Entry {i} row count differs"
            
            print("✅ Dataset generation is deterministic")
            
        finally:
            Path(temp_output1).unlink(missing_ok=True)
            Path(temp_output2).unlink(missing_ok=True)


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
