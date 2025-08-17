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
    
    # Map phrases to RELDATE tokens
    if re.search(r'\blast\s+7\s+days?\b', s_norm):
        return 'last_7d'
    elif re.search(r'\blast\s+week\b', s_norm):
        return 'last_week'
    elif re.search(r'\bthis\s+week\b', s_norm):
        return 'this_week'
    elif re.search(r'\bnext\s+week\b', s_norm):
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
    
    if 'free' in s_norm:
        return "status='free'"
    elif 'booked' in s_norm:
        return "status='booked'"
    
    return None


def check_kpi_patterns(s: str) -> Optional[str]:
    """Check for KPI pattern matches."""
    s_norm = norm(s)
    
    # KPI patterns
    if 'aged' in s_norm and 'receivables' in s_norm:
        return 'FIND kpi:aged_receivables'
    
    if (('no' in s_norm and 'shows' in s_norm) or 'dna' in s_norm) and ('last' in s_norm and '7' in s_norm):
        return 'FIND kpi:no_shows_last_7d'
    
    if ('free' in s_norm and 'double' in s_norm and 'next' in s_norm and 'week' in s_norm):
        return 'FIND kpi:free_double_slots_next_7d'
    
    return None


def predict_dsl(nl: str) -> str:
    """
    Convert natural language to DSL using rule-based pattern matching.
    
    Returns canonical DSL string or "NO_TOOL" if no patterns match.
    """
    if not nl or not nl.strip():
        return "NO_TOOL"
    
    s_norm = norm(nl)
    
    # Check for KPI patterns first
    kpi_result = check_kpi_patterns(nl)
    if kpi_result:
        return kpi_result
    
    # Check for basic entity patterns early (before building complex queries)
    if 'patients' in s_norm and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND patients LIMIT 50"
    elif 'invoices' in s_norm and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND invoices LIMIT 50"
    elif 'payments' in s_norm and not any(x in s_norm for x in ['appointments', 'slots', 'double']):
        return "FIND payments LIMIT 50"
    
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
    
    # If no filters found, check for remaining entity patterns
    if not filters:
        if 'appointments' in s_norm:
            return "FIND appointments LIMIT 50"
        else:
            return "NO_TOOL"
    
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
