#!/usr/bin/env python3
"""
Tests for baseline NL→DSL interpreter.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.interpreter.baseline import predict_dsl, batch_predict


class TestBaselineInterpreter:
    """Test the rule-based baseline interpreter."""
    
    def test_kpi_patterns(self):
        """Test KPI pattern recognition."""
        test_cases = [
            ("aged receivables by insurer", "FIND kpi:aged_receivables"),
            ("show aged receivables", "FIND kpi:aged_receivables"),
            ("dna last 7 days", "FIND kpi:no_shows_last_7d"),
            ("no shows last 7 days", "FIND kpi:no_shows_last_7d"),
            ("show no-shows in the last 7 days", "FIND kpi:no_shows_last_7d"),
            ("free double slots next week", "FIND kpi:free_double_slots_next_7d"),
            ("free double slots next week for Dr Khan", "FIND kpi:free_double_slots_next_7d")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_complex_appointment_queries(self):
        """Test complex appointment queries with canonical ordering."""
        test_cases = [
            (
                "dr. khan after 4pm friday free double slots next week",
                "FIND kpi:free_double_slots_next_7d"  # Should match KPI pattern first
            ),
            (
                "booked appts Friday after 16 with Patel", 
                "FIND appointments WHERE practitioner='Patel' AND day='fri' AND time >= '16:00' AND status='booked' LIMIT 50"
            ),
            (
                "free double slots next week for dr khan",
                "FIND kpi:free_double_slots_next_7d"  # Should match KPI pattern first
            ),
            (
                "Double slots after 16:00 Friday for Dr Khan",
                "FIND appointments WHERE practitioner='Khan' AND day='fri' AND time >= '16:00' AND slot='double' LIMIT 50"
            ),
            (
                "this week free appointments",
                "FIND appointments WHERE this_week AND status='free' LIMIT 50"
            ),
            (
                "next week with Dr Ng",
                "FIND appointments WHERE next_week AND practitioner='Ng' LIMIT 50"
            )
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_practitioner_matching(self):
        """Test practitioner name matching with various formats."""
        test_cases = [
            ("appointments for Dr Khan", "FIND appointments WHERE practitioner='Khan' LIMIT 50"),
            ("appointments for dr. khan", "FIND appointments WHERE practitioner='Khan' LIMIT 50"),
            ("appointments for KHAN", "FIND appointments WHERE practitioner='Khan' LIMIT 50"),
            ("appointments for Dr. Patel", "FIND appointments WHERE practitioner='Patel' LIMIT 50"),
            ("appointments for ng", "FIND appointments WHERE practitioner='Ng' LIMIT 50"),
            ("appointments for Dr Ng", "FIND appointments WHERE practitioner='Ng' LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_time_parsing(self):
        """Test various time format parsing."""
        test_cases = [
            ("appointments after 4pm", "FIND appointments WHERE time >= '16:00' LIMIT 50"),
            ("appointments after 4 PM", "FIND appointments WHERE time >= '16:00' LIMIT 50"),
            ("appointments after 16:00", "FIND appointments WHERE time >= '16:00' LIMIT 50"),
            ("appointments after 16", "FIND appointments WHERE time >= '16:00' LIMIT 50"),
            ("appointments after 4", "FIND appointments WHERE time >= '16:00' LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_day_extraction(self):
        """Test day of week extraction."""
        test_cases = [
            ("Friday appointments", "FIND appointments WHERE day='fri' LIMIT 50"),
            ("friday appointments", "FIND appointments WHERE day='fri' LIMIT 50"),
            ("FRI appointments", "FIND appointments WHERE day='fri' LIMIT 50"),
            ("Monday slots", "FIND appointments WHERE day='mon' LIMIT 50"),
            ("Tuesday bookings", "FIND appointments WHERE day='tue' LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_status_and_slot_extraction(self):
        """Test status and slot extraction."""
        test_cases = [
            ("free appointments", "FIND appointments WHERE status='free' LIMIT 50"),
            ("booked appointments", "FIND appointments WHERE status='booked' LIMIT 50"),
            ("double slots", "FIND appointments WHERE slot='double' LIMIT 50"),
            ("free double slots", "FIND appointments WHERE slot='double' AND status='free' LIMIT 50"),
            ("double free slots", "FIND appointments WHERE slot='double' AND status='free' LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_relative_dates(self):
        """Test relative date extraction.""" 
        test_cases = [
            ("appointments last 7 days", "FIND appointments WHERE last_7d LIMIT 50"),
            ("appointments last week", "FIND appointments WHERE last_week LIMIT 50"),
            ("appointments this week", "FIND appointments WHERE this_week LIMIT 50"),
            ("appointments next week", "FIND appointments WHERE next_week LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_entity_queries(self):
        """Test basic entity queries."""
        test_cases = [
            ("patients list", "FIND patients LIMIT 50"),
            ("show patients", "FIND patients LIMIT 50"),
            ("invoices listing", "FIND invoices LIMIT 50"),
            ("payments received", "FIND payments LIMIT 50"),
            ("show appointments", "FIND appointments LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"'{nl}' → '{result}', expected '{expected}'"
    
    def test_out_of_scope(self):
        """Test out-of-scope queries return NO_TOOL."""
        out_of_scope = [
            "show ECG results",
            "blood pressure readings",
            "prescribe medication",
            "schedule surgery",
            "random nonsense text",
            "",
            "   "  # whitespace only
        ]
        
        for nl in out_of_scope:
            result = predict_dsl(nl)
            assert result == "NO_TOOL", f"'{nl}' should return 'NO_TOOL', got '{result}'"
    
    def test_canonical_filter_ordering(self):
        """Test that filters are always emitted in canonical order."""
        # Test query with all filter types in mixed order
        nl = "booked double khan friday after 4pm next week"
        result = predict_dsl(nl)
        expected = "FIND appointments WHERE next_week AND practitioner='Khan' AND day='fri' AND time >= '16:00' AND slot='double' AND status='booked' LIMIT 50"
        assert result == expected, f"Filter ordering incorrect: '{result}'"
    
    def test_batch_predict(self):
        """Test batch prediction functionality."""
        nl_list = [
            "aged receivables by insurer",
            "free appointments",
            "show ECG results"
        ]
        expected = [
            "FIND kpi:aged_receivables",
            "FIND appointments WHERE status='free' LIMIT 50",
            "NO_TOOL"
        ]
        
        results = batch_predict(nl_list)
        assert results == expected, f"Batch prediction failed: {results}"
    
    def test_seed_dataset_examples(self):
        """Test exact examples from the seeds dataset."""
        test_cases = [
            ("Show aged receivables by insurer", "FIND kpi:aged_receivables"),
            ("Show no-shows in the last 7 days", "FIND kpi:no_shows_last_7d"),
            ("Free double slots next week for Dr Khan", "FIND kpi:free_double_slots_next_7d"),
            ("Show free appointments", "FIND appointments WHERE status='free' LIMIT 50"),
            ("Appointments for Dr Khan", "FIND appointments WHERE practitioner='Khan' LIMIT 50"),
            ("Booked appointments Friday after 16:00 with Dr Patel", "FIND appointments WHERE practitioner='Patel' AND day='fri' AND time >= '16:00' AND status='booked' LIMIT 50"),
            ("Double slots after 16:00 Friday for Dr Khan", "FIND appointments WHERE practitioner='Khan' AND day='fri' AND time >= '16:00' AND slot='double' LIMIT 50"),
            ("This week free appointments", "FIND appointments WHERE this_week AND status='free' LIMIT 50"),
            ("Next week with Dr Ng", "FIND appointments WHERE next_week AND practitioner='Ng' LIMIT 50"),
            ("Patients list", "FIND patients LIMIT 50"),
            ("Payments and invoices listing", "FIND invoices LIMIT 50"),
            ("Payments received listing", "FIND payments LIMIT 50"),
            ("Free slots next week", "FIND appointments WHERE next_week AND status='free' LIMIT 50"),
            ("Free double slots", "FIND appointments WHERE slot='double' AND status='free' LIMIT 50"),
            ("Booked double slots Friday after 16:00 with Khan", "FIND appointments WHERE practitioner='Khan' AND day='fri' AND time >= '16:00' AND slot='double' AND status='booked' LIMIT 50")
        ]
        
        for nl, expected in test_cases:
            result = predict_dsl(nl)
            assert result == expected, f"Seed example failed: '{nl}' → '{result}', expected '{expected}'"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
