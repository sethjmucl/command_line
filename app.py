#!/usr/bin/env python3
"""
Streamlit web interface for NL→DSL→SQL command palette.

Run with: streamlit run app.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import time
import sys
from pathlib import Path

# Add packages to path
sys.path.append(str(Path(__file__).parent))

from packages.interpreter.baseline import predict_dsl
from packages.dsl.compiler import compile_dsl


# Page config
st.set_page_config(
    page_title="Health Data Command Palette",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection (cached)
@st.cache_resource
def get_database_connection():
    """Get cached database connection."""
    conn = sqlite3.connect('data/pms.db', check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# Header
st.title("🏥 Health Data Command Palette")
st.markdown("**Natural Language Interface for Primary Care Management**")
st.markdown("*Ask questions in plain English about appointments, patients, practitioners, and KPIs*")

# Sidebar with examples and info
with st.sidebar:
    st.header("📚 Example Queries")
    
    example_queries = [
        "Show aged receivables by insurer",
        "Free appointments for Dr Khan",
        "No-shows in the last 7 days", 
        "Booked appointments Friday after 4pm",
        "Free double slots next week",
        "Patients list",
        "Payments received listing",
        "Double slots after 16:00 Friday for Dr Khan",
        "This week free appointments"
    ]
    
    for example in example_queries:
        if st.button(f"💡 {example}", key=f"example_{example}", use_container_width=True):
            st.session_state.query_input = example
    
    st.markdown("---")
    st.header("ℹ️ System Info")
    st.markdown("""
    **Pipeline**: NL → DSL → SQL → Results
    
    **Supported Queries**:
    - 👥 Patients, practitioners
    - 📅 Appointments (free/booked)
    - 💰 Invoices, payments
    - 📊 KPI reports
    - 🕐 Time filters (today, this week, etc.)
    - 👨‍⚕️ Practitioner filters (Khan, Patel, Ng)
    
    **Out of Scope**: Clinical data, prescriptions, lab results
    """)

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_input(
        "💬 Ask a question about the practice:",
        value=st.session_state.get('query_input', ''),
        placeholder="e.g., Show me free appointments for Dr Khan next week",
        key="main_query"
    )

with col2:
    # Process button
    process_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)

# Clear the session state after using it
if 'query_input' in st.session_state:
    del st.session_state.query_input

# Process query
if query and (process_clicked or query != st.session_state.get('last_query', '')):
    st.session_state.last_query = query
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: NL → DSL
        with st.spinner("🧠 Understanding your question..."):
            dsl = predict_dsl(query)
            nl_to_dsl_time = time.time() - start_time
        
        # Check for abstention or suggestions
        if dsl == "NO_TOOL":
            st.error("🚫 **Sorry, I can't help with that query.**")
            st.info("""
            This question seems to be outside my scope. I can help with:
            - Appointment scheduling and availability
            - Patient and practitioner information  
            - Financial reports (invoices, payments)
            - Practice KPIs and metrics
            
            Try rephrasing or ask about appointments, patients, or practice management.
            """)
        elif dsl.startswith("SUGGEST:"):
            suggestion = dsl[8:]  # Remove "SUGGEST:" prefix
            st.warning("🤔 **I think I understand, but need more details.**")
            st.info(f"💡 **Suggestion**: {suggestion}")
            
            # Extract the suggested query if it's in quotes
            import re
            quoted_suggestions = re.findall(r"'([^']*)'|'([^']*)'", suggestion)
            if quoted_suggestions:
                st.markdown("**Quick actions:**")
                for match in quoted_suggestions:
                    suggested_query = match[0] or match[1]  # Handle both quote types
                    if suggested_query:
                        if st.button(f"🎯 Try: {suggested_query}", key=f"suggest_{suggested_query}"):
                            st.session_state.query_input = suggested_query
                            st.rerun()
        else:
            # Step 2: DSL → SQL
            with st.spinner("⚙️ Generating database query..."):
                conn = get_database_connection()
                sql, params = compile_dsl(dsl, conn)
                compile_time = time.time() - start_time - nl_to_dsl_time
            
            # Step 3: Execute SQL
            with st.spinner("🗄️ Fetching results..."):
                results_df = pd.read_sql_query(sql, conn, params=params)
                total_time = time.time() - start_time
            
            # Display results
            st.success(f"✅ **Found {len(results_df)} results in {total_time*1000:.1f}ms**")
            
            # Pipeline transparency (expandable)
            with st.expander("🔍 See how I understood your question", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🗣️ Natural Language**")
                    st.code(query, language=None)
                    
                    st.markdown("**🤖 Domain-Specific Language**")
                    st.code(dsl, language=None)
                
                with col2:
                    st.markdown("**💾 SQL Query**")
                    sql_display = sql
                    if params:
                        sql_display += f"\n-- Parameters: {params}"
                    st.code(sql_display, language="sql")
                    
                    st.markdown("**⏱️ Performance**")
                    st.metric("Total Latency", f"{total_time*1000:.1f}ms")
                    st.metric("NL→DSL", f"{nl_to_dsl_time*1000:.1f}ms")
                    st.metric("DSL→SQL", f"{compile_time*1000:.1f}ms")
            
            # Results table
            st.markdown("### 📊 Results")
            if len(results_df) == 0:
                st.info("No results found for your query.")
            else:
                # Format the dataframe nicely
                display_df = results_df.copy()
                
                # Format timestamps if any
                for col in display_df.columns:
                    if 'timestamp' in col.lower() or 'date' in col.lower() or col.endswith('_at'):
                        if display_df[col].dtype == 'object':
                            display_df[col] = pd.to_datetime(display_df[col], errors='ignore')
                    
                    # Format currency columns
                    if 'amount' in col.lower() or 'total' in col.lower():
                        if pd.api.types.is_numeric_dtype(display_df[col]):
                            display_df[col] = display_df[col].apply(lambda x: f"£{x:.2f}" if pd.notna(x) else "")
                
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    height=min(400, len(display_df) * 35 + 50)  # Dynamic height
                )
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"query_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ **Error processing your query**")
        st.exception(e)
        
        # Show debugging info
        with st.expander("🔧 Debug Information"):
            st.markdown("**Query Input**")
            st.code(query)
            if 'dsl' in locals():
                st.markdown("**Generated DSL**") 
                st.code(dsl)
            if 'sql' in locals():
                st.markdown("**Generated SQL**")
                st.code(sql)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
💡 <strong>Tip:</strong> Try asking about appointments, practitioners, patients, or practice KPIs<br/>
🔬 <strong>Research Project:</strong> Natural Language Interface for Health Data | Built with Streamlit
</div>
""", unsafe_allow_html=True)
