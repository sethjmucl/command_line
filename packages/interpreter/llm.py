#!/usr/bin/env python3
"""
LLM-based NL→DSL prediction using OpenAI API.

Implements few-shot prompting with grammar examples and handles API errors gracefully.
"""

import os
import openai
from typing import Optional
from pathlib import Path

# Try to load environment variables
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or .env file."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Fallback: try common alternative names
        api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_TOKEN')
    return api_key

def create_few_shot_prompt() -> str:
    """Create few-shot prompt with DSL grammar and examples."""
    return """You are an expert at converting natural language queries about a UK primary care practice into a Domain-Specific Language (DSL).

DSL GRAMMAR:
```
query: "FIND" target ("WHERE" filters)? ("ORDER BY" field ("ASC"|"DESC")?)? ("LIMIT" INT)?
target: ENTITY | KPI
ENTITY: "appointments" | "patients" | "invoices" | "payments" | "practitioners"  
KPI: "kpi:" ("aged_receivables" | "no_shows_last_7d" | "free_double_slots_next_7d" | "revenue_by_practitioner")
filters: filter ("AND" filter)*
filter: field OP value | RELDATE
field: practitioner | day | slot | status | time | role
OP: "=" | "!=" | ">=" | "<=" | "IN" | "LIKE"
RELDATE: "last_7d" | "last_week" | "this_week" | "next_week"
```

EXAMPLES:
NL: "Show aged receivables by insurer"
DSL: FIND kpi:aged_receivables

NL: "Appointments for Dr Khan"  
DSL: FIND appointments WHERE practitioner='Khan' LIMIT 50

NL: "Free slots next week"
DSL: FIND appointments WHERE next_week AND status='free' LIMIT 50

NL: "Show no-shows in the last 7 days"
DSL: FIND kpi:no_shows_last_7d

NL: "Booked appointments Friday after 16:00 with Dr Pat"
DSL: FIND appointments WHERE day='fri' AND time>='16:00' AND status='booked' AND practitioner='Pat' LIMIT 50

NL: "Patients list"
DSL: FIND patients LIMIT 50

NL: "Double slots after 16:00 Friday for Dr Khan"
DSL: FIND appointments WHERE time>='16:00' AND day='fri' AND slot='double' AND practitioner='Khan' LIMIT 50

RULES:
- If you cannot determine a valid DSL query, return "NO_TOOL"
- Practitioner names should use last name only (e.g., "Dr Smith" → practitioner='Smith')
- Time format: 24-hour (e.g., "4pm" → '16:00')
- Days: use 3-letter abbreviations (mon, tue, wed, thu, fri, sat, sun)
- Status values: 'free', 'booked', 'cancelled', 'dna'
- Slot values: 'single', 'double'
- Always include LIMIT 50 for entity queries unless specified otherwise
- For out-of-scope queries, return "NO_TOOL"

Convert this natural language query to DSL:"""

def predict_dsl_openai(nl: str, model: str = "gpt-4o-mini", max_retries: int = 2) -> str:
    """
    Predict DSL using OpenAI API with few-shot prompting.
    
    Args:
        nl: Natural language query
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        max_retries: Number of retry attempts on API errors
        
    Returns:
        DSL string or "NO_TOOL" if prediction fails
    """
    api_key = get_api_key()
    if not api_key:
        print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return "NO_TOOL"
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Create prompt
    system_prompt = create_few_shot_prompt()
    user_prompt = f"""
NL: "{nl}"
DSL:"""
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.0,  # Deterministic for consistency
                timeout=10.0
            )
            
            # Extract and clean response
            dsl = response.choices[0].message.content.strip()
            
            # Basic validation - should start with FIND or be NO_TOOL
            if dsl.startswith("FIND ") or dsl == "NO_TOOL":
                return dsl
            else:
                # If response doesn't match expected format, try to extract
                lines = dsl.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith("FIND ") or line == "NO_TOOL":
                        return line
                
                # If we can't find a valid DSL, return NO_TOOL
                return "NO_TOOL"
                
        except openai.RateLimitError:
            print(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries + 1})")
            if attempt < max_retries:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "NO_TOOL"
                
        except openai.APIError as e:
            print(f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                import time
                time.sleep(1)
            else:
                return "NO_TOOL"
                
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                import time
                time.sleep(1)
            else:
                return "NO_TOOL"
    
    return "NO_TOOL"

def predict_dsl_hybrid(nl: str, model: str = "gpt-4o-mini") -> str:
    """
    Hybrid prediction: try rule-based first, fall back to LLM for complex cases.
    
    Args:
        nl: Natural language query
        model: OpenAI model to use for LLM fallback
        
    Returns:
        DSL string from best available method
    """
    from .baseline import predict_dsl as predict_dsl_baseline
    
    # Try rule-based first
    baseline_result = predict_dsl_baseline(nl)
    
    # If rule-based gives a definitive answer, use it
    if baseline_result not in ["NO_TOOL"] and not baseline_result.startswith("SUGGEST:"):
        return baseline_result
    
    # For complex cases or suggestions, try LLM
    llm_result = predict_dsl_openai(nl, model)
    
    # If LLM gives a better result, use it
    if llm_result != "NO_TOOL":
        return llm_result
    
    # Fall back to baseline (even if it's NO_TOOL or suggestion)
    return baseline_result

# Main prediction function for easy import
def predict_dsl(nl: str, strategy: str = "hybrid", model: str = "gpt-4o-mini") -> str:
    """
    Main prediction function that supports different strategies.
    
    Args:
        nl: Natural language query
        strategy: "baseline", "llm", or "hybrid" 
        model: OpenAI model for LLM strategies
        
    Returns:
        Predicted DSL string
    """
    if strategy == "baseline":
        from .baseline import predict_dsl as predict_dsl_baseline
        return predict_dsl_baseline(nl)
    elif strategy == "llm":
        return predict_dsl_openai(nl, model)
    elif strategy == "hybrid":
        return predict_dsl_hybrid(nl, model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'baseline', 'llm', or 'hybrid'")
