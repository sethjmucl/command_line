"""
Rule-based baseline for NL→DSL translation.

Implements deterministic pattern matching with canonical filter ordering.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Set


def norm(s: str) -> str:
    """Normalize string: lowercase, strip accents/punct, collapse whitespace."""
    # Remove accents
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    
    # Remove punctuation and extra spaces
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s.lower().strip())
    
    return s


def parse_time_ish(s: str) -> Optional[str]:
    """Parse time expressions and return canonical time filter."""
    s_norm = norm(s)
    
    # Patterns for "after 4", "after 4pm", "16:00", "16", etc.
    patterns = [
        r'\bafter\s+4\b',
        r'\bafter\s+4\s*pm\b',
        r'\bafter\s+16\b',
        r'\b16:00\b',
        r'\b16\b(?!\d)',  # 16 not followed by more digits
        r'\b4\s*pm\b',
        r'\b4\s*PM\b'
    ]
    
    for pattern in patterns:
        if re.search(pattern, s_norm):
            return "time >= '16:00'"
    
    return None


def match_practitioner(s: str) -> Optional[str]:
    """Match practitioner names, handling 'dr' prefix."""
    s_norm = norm(s)
    
    # Remove 'dr' prefix
    s_clean = re.sub(r'\bdr\b', '', s_norm).strip()
    
    practitioners = {
        'khan': 'Khan',
        'patel': 'Patel', 
        'ng': 'Ng'
    }
    
    for lastname, canonical in practitioners.items():
        # Use word boundaries to avoid false matches
        if re.search(r'\b' + lastname + r'\b', s_clean):
            return f"practitioner='{canonical}'"
    
    return None


def extract_day(s: str) -> Optional[str]:
    """Extract day of week."""
    s_norm = norm(s)
    
    day_map = {
        'monday': 'mon', 'mon': 'mon',
        'tuesday': 'tue', 'tue': 'tue', 
        'wednesday': 'wed', 'wed': 'wed',
        'thursday': 'thu', 'thu': 'thu',
        'friday': 'fri', 'fri': 'fri',
        'saturday': 'sat', 'sat': 'sat',
        'sunday': 'sun', 'sun': 'sun'
    }
    
    for day_name, day_code in day_map.items():
        if day_name in s_norm:
            return f"day='{day_code}'"
    
    return None


def extract_reldate(s: str) -> Optional[str]:
    """Extract relative date tokens."""
    s_norm = norm(s)
    
    # Map phrases to RELDATE tokens - more flexible matching
    # Order matters: check more specific patterns first
    if any(phrase in s_norm for phrase in ['last 7 days', 'past 7 days']):
        return 'last_7d'
    elif any(phrase in s_norm for phrase in ['last week', 'past week']):
        return 'last_week'
    elif any(phrase in s_norm for phrase in ['this week', 'current week']):
        return 'this_week'
    elif any(phrase in s_norm for phrase in ['next week', 'upcoming week', 'following week']):
        return 'next_week'
    
    return None


def extract_slot(s: str) -> Optional[str]:
    """Extract slot type (double)."""
    s_norm = norm(s)
    
    if 'double' in s_norm:
        return "slot='double'"
    
    return None


def extract_status(s: str) -> Optional[str]:
    """Extract status (free/booked)."""
    s_norm = norm(s)
    
    if any(x in s_norm for x in ['free', 'available', 'open']):
        return "status='free'"
    elif any(x in s_norm for x in ['booked', 'scheduled', 'occupied', 'taken']):
        return "status='booked'"
    
    return None


def check_kpi_patterns(s: str) -> Optional[str]:
    """Check for KPI pattern matches."""
    s_norm = norm(s)
    
    # KPI patterns - more flexible matching
    # Revenue-like intents → aged receivables (closest supported KPI)
    if any(x in s_norm for x in ['aged receivables', 'outstanding invoices', 'money owed', 'debt', 'accounts receivable', 'outstanding payments']):
        return 'FIND kpi:aged_receivables'
    # Revenue by practitioner intent
    if any(x in s_norm for x in ['revenue by practitioner', 'revenue per doctor', 'which practitioner has', 'practitioner revenue', 'doctor revenue', 'least revenue', 'most revenue']):
        # Default ordering must be handled downstream; baseline returns KPI view
        return 'FIND kpi:revenue_by_practitioner'
    
    if (any(x in s_norm for x in ['no shows', 'no-shows', 'dna', 'did not attend']) and 
        any(x in s_norm for x in ['last 7 days', 'past week', 'recent'])):
        return 'FIND kpi:no_shows_last_7d'
    
    # Catch standalone "no shows" or "dna" queries
    if any(x in s_norm for x in ['no shows', 'no-shows', 'dna', 'did not attend']) and not any(x in s_norm for x in ['appointments', 'slots']):
        return 'FIND kpi:no_shows_last_7d'
    
    if (any(x in s_norm for x in ['free double', 'double slots']) and 
        any(x in s_norm for x in ['next week', 'upcoming'])):
        return 'FIND kpi:free_double_slots_next_7d'
    
    return None


def get_helpful_suggestion(nl: str) -> str:
    """Provide helpful suggestions for unclear or out-of-scope queries."""
    s_norm = norm(nl)
    
    # Check for partial matches that we can suggest clarifications for
    if any(x in s_norm for x in ['revenue', 'money', 'income', 'profit', 'financial']):
        return "SUGGEST: Try 'aged receivables by insurer' to see outstanding payments"
    
    if any(x in s_norm for x in ['appointment', 'booking', 'schedule']) and not any(x in s_norm for x in ['free', 'booked', 'khan', 'patel', 'ng']):
        return "SUGGEST: Try 'free appointments' or 'appointments for Dr Khan'"
    
    if any(x in s_norm for x in ['doctor', 'practitioner', 'staff']) and 'list' not in s_norm:
        return "SUGGEST: Try 'show practitioners' or 'appointments for Dr Khan'"
    
    if any(x in s_norm for x in ['patient']) and 'list' not in s_norm:
        return "SUGGEST: Try 'patients list' or 'appointments for [patient name]'"
    
    if any(x in s_norm for x in ['time', 'when', 'schedule', 'calendar']):
        return "SUGGEST: Try 'free appointments Friday after 4pm' or 'next week appointments'"
    
    if any(x in s_norm for x in ['report', 'summary', 'stats', 'analytics']):
        return "SUGGEST: Try 'aged receivables' or 'no-shows last 7 days'"
    
    # For clinical/medical terms
    if any(x in s_norm for x in ['ecg', 'blood pressure', 'prescription', 'diagnosis', 'treatment', 'medication', 'lab', 'test']):
        return "SUGGEST: I can help with practice management, not clinical data. Try 'appointments' or 'patient list'"
    
    return "NO_TOOL"


def predict_dsl(nl: str) -> str:
    """
    Convert natural language to DSL using rule-based pattern matching.
    
    Returns canonical DSL string, helpful suggestion (SUGGEST:...), or "NO_TOOL" if no patterns match.
    """
    if not nl or not nl.strip():
        return "NO_TOOL"
    
    s_norm = norm(nl)
    
    # Check for KPI patterns first
    kpi_result = check_kpi_patterns(nl)
    if kpi_result:
        return kpi_result
    
    # Check for basic entity patterns early (before building complex queries)
    if any(x in s_norm for x in ['patients', 'patient list']) and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND patients LIMIT 50"
    elif any(x in s_norm for x in ['practitioners', 'practitioner list', 'doctor list', 'staff', 'clinicians']) and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND practitioners LIMIT 50"
    elif any(x in s_norm for x in ['doctors', 'gps', 'general practitioners']) and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND practitioners WHERE role='GP' LIMIT 50"
    elif any(x in s_norm for x in ['outstanding invoices', 'unpaid invoices', 'open invoices']) and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND invoices WHERE status='outstanding' LIMIT 50"
    elif any(x in s_norm for x in ['invoices', 'bills', 'billing']) and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND invoices LIMIT 50"
    elif ('payments' in s_norm and not any(x in s_norm for x in ['appointments', 'slots', 'double'])):
        return "FIND payments LIMIT 50"
    elif any(x in s_norm for x in ['revenue', 'receivables', 'outstanding', 'money owed', 'debt', 'finance', 'outstanding invoices']) and 'aged' not in s_norm:
        return "FIND kpi:aged_receivables"
    
    # Otherwise, build appointments query
    filters = []
    
    # Extract components in canonical order: RELDATE → practitioner → day → time → slot → status
    
    # 1. RELDATE
    reldate = extract_reldate(nl)
    if reldate:
        filters.append(reldate)
    
    # 2. Practitioner
    practitioner = match_practitioner(nl)
    if practitioner:
        filters.append(practitioner)
    
    # 3. Day
    day = extract_day(nl)
    if day:
        filters.append(day)
    
    # 4. Time
    time_filter = parse_time_ish(nl)
    if time_filter:
        filters.append(time_filter)
    
    # 5. Slot and Status (handle "free double" specially)
    if 'free' in s_norm and 'double' in s_norm:
        # Both slot and status
        filters.append("slot='double'")
        filters.append("status='free'")
    else:
        # Individual checks
        slot = extract_slot(nl)
        if slot:
            filters.append(slot)
        
        status = extract_status(nl)
        if status:
            filters.append(status)

    # 6. Doctor-specific queries should exclude nurses when user intent is Dr/GP, but only if no explicit name filter
    if (any(x in s_norm for x in [' dr ', 'doctor', 'doctors', 'gp', 'gps']) 
        and (not any(f.startswith("practitioner='") for f in filters))
        and not any("role='" in f for f in filters)):
        filters.append("role='GP'")
    
    # If no filters found, check for remaining entity patterns
    if not filters:
        if 'appointments' in s_norm:
            return "FIND appointments LIMIT 50"
        else:
            # Try to provide helpful suggestions instead of just "NO_TOOL"
            return get_helpful_suggestion(nl)
    
    # Build DSL query
    dsl_parts = ["FIND appointments"]
    
    if filters:
        dsl_parts.append("WHERE " + " AND ".join(filters))
    
    dsl_parts.append("LIMIT 50")
    
    return " ".join(dsl_parts)


def batch_predict(nl_list: List[str]) -> List[str]:
    """Batch prediction for multiple NL inputs."""
    return [predict_dsl(nl) for nl in nl_list]


# Test the implementation
if __name__ == "__main__":
    # Quick test cases
    test_cases = [
        "show aged receivables by insurer",
        "free double slots next week for dr khan", 
        "dna last 7 days",
        "booked appts Friday after 4pm with Patel",
        "show ECG results"  # out of scope
    ]
    
    for test in test_cases:
        result = predict_dsl(test)
        print(f"'{test}' → '{result}'")
