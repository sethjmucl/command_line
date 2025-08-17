AI-Powered Health Data Command Palette

Natural language → DSL → SQL over a curated, reproducible SQLite database for UK primary care. Hybrid interpreter (rule-based + OpenAI LLM) with a Streamlit UI and a comprehensive evaluation harness.

Overview

- Hybrid NL→DSL interpreter: fast rule-based baseline plus optional OpenAI LLM and a hybrid strategy
- Minimal, safe DSL compiled to parameterized SQL
- Deterministically seeded SQLite with realistic UK-flavoured data and KPI views
- Streamlit app for interactive querying
- Evaluation harness with enhanced metrics (intent accuracy, slot-F1, robustness, coverage, edge cases, latency)

Quick start

1) Install

```bash
pip install -r requirements.txt
```

2) Seed the database (deterministic)

```bash
python scripts/seed_db.py --db sqlite:///data/pms.db --seed 42 --reset
```

3) (Optional) Configure OpenAI for LLM/hybrid strategies

Create a .env file in the project root:

```bash
echo OPENAI_API_KEY=sk-... > .env
```

4) Run the Streamlit UI

```bash
streamlit run app.py
```

In the sidebar: pick strategy (baseline, llm, hybrid) and, for LLM, a model (gpt-4o-mini/gpt-4o/gpt-3.5-turbo).

What you can ask

- "Show appointments for Dr Khan"
- "Free slots next week"
- "No-shows in the last 7 days"
- "Outstanding invoices"
- "Revenue by practitioner" (includes per-practitioner n_invoices, gross/collected/outstanding)

Project structure

```
command_line/
  app.py                         # Streamlit UI (strategy selector: baseline/llm/hybrid)
  data/                          # DB and datasets (gitignored outputs)
  packages/
    dsl/
      grammar.lark               # EBNF grammar
      compiler.py                # DSL → SQL compiler
      date_resolver.py           # Relative date helpers (tokens)
    interpreter/
      baseline.py                # Rule-based NL→DSL
      llm.py                     # OpenAI NL→DSL + hybrid strategy
    eval_harness/
      evaluate.py                # Evaluation CLI and metrics aggregation
      enhanced_metrics.py        # Intent, slot-F1, coverage, edge cases
  scripts/
    seed_db.py                   # Deterministic schema + seeding + KPI views
    make_dataset.py              # Gold dataset generation and denotation hashing
  tests/                         # Pytest suites (DSL, interpreter, seeding, eval)
  README.md
  requirements.txt
```

Database (SQLite)

Core tables

- patients(patient_id, first_name, last_name, dob, phone, email, insurer, created_at)
- practitioners(practitioner_id, first_name, last_name, role)
- appointments(appt_id, patient_id NULL, practitioner_id, start_ts, duration_minutes, status IN('free','booked','cancelled','dna'))
- insurers(name)
- invoices(invoice_id, patient_id, insurer, appointment_id NULL, amount, issued_at, due_at, status)
- payments(payment_id, invoice_id, amount, paid_at, method)
- referrals, documents

Deterministic KPI views

- vw_no_shows_last_7d: recent DNAs
- vw_aged_receivables: insurer-level totals and outstanding
- vw_free_double_slots_next_7d: staffing availability
- vw_revenue_by_practitioner: per-practitioner n_invoices, gross, collected, outstanding

DSL summary

- Targets: ENTITY (appointments, patients, invoices, payments, practitioners) or KPI (aged_receivables, no_shows_last_7d, free_double_slots_next_7d, revenue_by_practitioner)
- WHERE: whitelisted fields (practitioner, day, time, slot, status, role) and RELDATE tokens (last_7d, last_week, this_week, next_week)
- ORDER BY: optional field + direction; LIMIT defaults to 50

Examples

```text
FIND appointments WHERE practitioner='Khan' LIMIT 50
FIND appointments WHERE next_week AND status='free' LIMIT 50
FIND kpi:revenue_by_practitioner
FIND kpi:no_shows_last_7d
```

Interpreter strategies

- baseline: robust rule-based patterns, suggestions for out-of-scope
- llm: OpenAI few-shot prompting to produce DSL (valid DSL or NO_TOOL)
- hybrid: baseline first; if abstains or suggests, fall back to LLM

Evaluation

Datasets

- data/ask_dataset.jsonl: gold questions with DSL, SQL, denotation hashes
- data/robustness_processed.jsonl: variations (typos, abbrevs, UK phrasing) with hashes

Run

```bash
# Baseline
python -m packages.eval_harness.evaluate \
  --db sqlite:///data/pms.db \
  --test data/ask_dataset.jsonl \
  --model baseline \
  --out out/baseline_metrics.json \
  --pred out/baseline_predictions.csv

# LLM (requires OPENAI_API_KEY)
python -m packages.eval_harness.evaluate \
  --db sqlite:///data/pms.db \
  --test data/ask_dataset.jsonl \
  --model llm \
  --out out/llm_metrics.json \
  --pred out/llm_predictions.csv

# Hybrid
python -m packages.eval_harness.evaluate \
  --db sqlite:///data/pms.db \
  --test data/ask_dataset.jsonl \
  --model hybrid \
  --out out/hybrid_metrics.json \
  --pred out/hybrid_predictions.csv

# Robustness set (hybrid)
python -m packages.eval_harness.evaluate \
  --db sqlite:///data/pms.db \
  --test data/robustness_processed.jsonl \
  --model hybrid \
  --out out/robustness_metrics.json \
  --pred out/robustness_predictions.csv
```

Enhanced metrics (automatically computed)

- Intent accuracy (kpi, entity_list, filtered_appointments, complex_combo)
- Slot-F1 (practitioner, day, time_ge, slot, status, role, domain/kpi)
- Coverage & feature utilization
- Edge cases (empty results, LIMIT-capped, abstention rate)
- Latency (p50/p90/mean/max)

Development & testing

```bash
pytest
```

Troubleshooting

- OpenAI 401 invalid_api_key in Streamlit: restart Streamlit after updating .env; the app now reloads .env on start and prints the last 4 chars of the key it picked up.
- Windows PowerShell multi-line python -c quirks: prefer scripts over long -c commands.
- Pandas FutureWarning for to_datetime(errors='ignore'): replaced with try/except conversion in the UI.

License

MIT

# AI-Powered Health Data Command Palette

> **Hybrid NL→DSL→SQL System for Primary Care Management**  
> *Combining rule-based precision with LLM intelligence*

[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)](#current-status)
[![Evaluation](https://img.shields.io/badge/Baseline%20Accuracy-100%25-brightgreen)](#evaluation-results)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20Integrated-blue)](#llm-integration)
[![Metrics](https://img.shields.io/badge/Enhanced%20Metrics-Implemented-orange)](#enhanced-evaluation)

## 🎯 Project Overview

This project implements a **natural language command line** for primary care operations that translates plain English questions into structured queries. The system combines **rule-based precision** with **LLM intelligence** to provide robust, accurate, and cost-effective natural language data access.

## 1) Goal

Build a **natural-language command palette** for primary-care operations that can **answer questions** ("Ask" Component) by translating NL → **Query DSL** → **parameterised SQL** over a small, curated SQLite/Postgres schema seeded with realistic fake data. 
For later: enable a tiny **Do-mode** (write) for one or two endpoints.

### Priorities

- **Read-first** capabilities and **evaluation metrics** (denotation, execution, relevance, latency)
- **Reproducibility** (seeded DB, deterministic views, tagged release)
- **Simplicity** over production concerns (no auth/roles/audit until after paper)

## Scope

### Key Components

- **Ask tiers**: T1 lookups; T2 joins/filters; T3 aggregates/time windows
- **Database**: `patients`, `practitioners`, `appointments`, `insurers`, `invoices`, `payments`, `referrals`, `documents`, `messages` + KPI **views**
- **Translator**: NL → **Query DSL** → SQL. Primary model: 7-8B instruct (LoRA) for NL→DSL. Baseline: few-shot prompt
- **Evaluation**: denotation accuracy (gold result match), execution accuracy, relevance P/R (abstain), latency p50/p90; *(optional)* DSL/SQL exact-match

### Non-Goals

For the paper scope, we exclude: auth, SSO, audit trails, RBAC, idempotency/rollback.

## Repository Structure

```
apps/
  cli/                 # text UI; runs queries; prints tables & summaries
  web/                 # (optional) minimal viewer for results
packages/
  dsl/                 # grammar, parser, compiler to SQL
  interpreter/         # NL->DSL model wrappers (SFT + prompting)
  eval_harness/        # metrics: denotation, execution, relevance, latency
  datastore/           # SQLite/Postgres schema, seeders, KPI views
infra/
  docker/              # compose file for Postgres + app (optional)
scripts/
  make_dataset.py      # generate NL/DSL/SQL/denotation JSONL
  seed_db.py           # create & seed the DB deterministically
README.md              # this file (copied)
```

## Data Store

### Database Schema

Using SQLite by default with the following minimal schema:

```sql
-- patients
CREATE TABLE patients (
  patient_id INTEGER PRIMARY KEY,
  first_name TEXT, last_name TEXT, dob DATE,
  phone TEXT, email TEXT, insurer_id INTEGER, created_at TIMESTAMP
);

-- practitioners
CREATE TABLE practitioners (
  practitioner_id INTEGER PRIMARY KEY,
  first_name TEXT, last_name TEXT, role TEXT
);

-- appointments
CREATE TABLE appointments (
  appt_id INTEGER PRIMARY KEY,
  patient_id INTEGER, practitioner_id INTEGER,
  start_ts TIMESTAMP, duration_minutes INTEGER,
  status TEXT CHECK(status IN ('free','booked','cancelled','dna'))
);

-- insurers, invoices, payments, referrals, documents, messages (similar minimal columns)
```

### KPI Views

Deterministic views for common business queries:

```sql
CREATE VIEW vw_no_shows_last_7d AS
SELECT a.* FROM appointments a
WHERE a.status='dna' AND a.start_ts >= date('now','-7 day');

CREATE VIEW vw_aged_receivables AS
SELECT i.insurer_id, COUNT(*) AS n_invoices,
       SUM(i.amount) AS total, SUM(i.amount - IFNULL(p.amount,0)) AS outstanding
FROM invoices i
LEFT JOIN (
  SELECT invoice_id, SUM(amount) AS amount FROM payments GROUP BY invoice_id
) p USING(invoice_id)
GROUP BY i.insurer_id;
```

### Database Seeding

`seed_db.py` uses `faker` (UK) to create approximately:
- 5k patients
- 10 practitioners  
- 15k appointments
- Insurers (Bupa/AXA/Aviva)
- Associated invoices/payments

Uses a fixed RNG seed for reproducibility.

## Query DSL

Tiny and safe domain-specific language for queries.

### Grammar (EBNF)

```
QUERY      := FIND CLAUSE [WHERE FILTERS] [GROUP BY FIELDS] [ORDER BY FIELD [ASC|DESC]] [LIMIT INT]
FIND       := 'FIND' (ENTITY | KPI)
ENTITY     := 'appointments' | 'patients' | 'invoices' | 'payments' | 'referrals'
KPI        := 'kpi' ':' ('aged_receivables' | 'no_shows_last_7d' | 'free_double_slots_next_7d')
FILTERS    := FILTER { 'AND' FILTER }
FILTER     := FIELD OP VALUE | DATEWINDOW
FIELD      := IDENT
OP         := '=' | '!=' | '>=' | '<=' | 'IN' | 'LIKE'
VALUE      := STRING | NUMBER | DATE | TIME | '[' VALUE {',' VALUE } ']'
FIELDS     := FIELD { ',' FIELD }
```

*(Keep it minimal; we can extend later.)*

### Examples

```text
FIND appointments WHERE practitioner='Khan' AND day IN 'Fri' AND time >= '16:00' AND slot='double' ORDER BY time ASC LIMIT 20
FIND kpi:aged_receivables
FIND patients WHERE last_name LIKE 'Ali' LIMIT 5
```

### Compiler

- Map DSL fields to **whitelisted** columns
- Always apply a default `LIMIT 50` unless specified
- Parameterize values to avoid injection

## Dataset (Ask-mode)

### Seeds

~25 canonical questions **per persona** × tiers. Store CSV with: `utterance, dsl, sql, denotation_hash, persona, tier, seed_id`.

Sample seeds:

- Reception: "Show Mrs Ali's next appointment." → DSL to filtered `appointments`
- Manager: "Unpaid invoices >30 days by insurer for July." → DSL to KPI view + date window
- Billing: "Payments received yesterday by method." → GROUP BY on payments

### Expansion

`scripts/make_dataset.py` paraphrases each seed 5-8× (LLM), preserves meaning, injects UK tokens ("surgery", "locum"). It compiles DSL→SQL, runs against the seeded DB, and records **denotation hashes** (e.g., SHA256 of CSV rows) for evaluation.

### Splits

Stratified 80/10/10 by persona & tier. Add **abstention set** (out-of-scope Qs) and **robustness set** (OOD names, date math, typos).

### JSONL Format

```json
{"instruction": "Free double slots next week after 4pm for Dr Khan",
 "dsl": "FIND appointments WHERE practitioner='Khan' AND day IN 'Fri' AND time >= '16:00' AND slot='double' ORDER BY time ASC LIMIT 20",
 "sql": "SELECT ...",
 "denotation": {"hash":"b2c...","n_rows":3},
 "persona":"reception","tier":2}
```

---

## Models

- **Primary**: NL→DSL with **Mistral-7B-Instruct** or **Llama-3.1-8B-Instruct** (LoRA). Train 1-2 epochs; curriculum from simple lookups → aggregates
- **Baselines**: (a) few-shot prompt (no fine-tune); (b) NL→SQL direct (whitelisted) to benchmark the DSL approach

### Decoding

Grammar-constrained for DSL (prevents malformed queries). For metrics, use beams; for demos, you can sample.

## Evaluation

- **Denotation accuracy**: result/aggregate matches gold (hash or numeric equality)
- **Execution accuracy**: SQL runs without error & within guards
- **Relevance P/R**: correctly abstain on out-of-scope questions
- **Latency**: p50/p90 wall-clock from NL→rows
- *(Optional)* **Exact-match**: AST-aware equality of DSL/SQL

### Reporting

Tables by persona & tier; short error taxonomy (wrong join, wrong filter, date-math error, abstention mistake).

### Done Criteria (Read-phase)

≥0.90/0.85/0.80 denotation (T1/T2/T3), ≥0.95 precision on abstention, p50 ≤ 2s; reproducible `metrics.json`.

## Quick Start

```bash
# 1) Create & seed DB
python scripts/seed_db.py --db sqlite:///data/pms.db --seed 42

# 2) Generate dataset (seeds + paraphrases)
python scripts/make_dataset.py --db sqlite:///data/pms.db --out data/ask_dataset.jsonl

# 3) Train NL->DSL (LoRA)
python -m packages.interpreter.train_sft \
  --train data/ask_dataset.jsonl --model mistral-7b-instruct \
  --epochs 2 --output runs/nl2dsl_mistral_lora

# 4) Evaluate
python -m packages.eval_harness.evaluate \
  --db sqlite:///data/pms.db --test data/ask_dataset_test.jsonl \
  --model_dir runs/nl2dsl_mistral_lora --out out/metrics.json

# 5) CLI demo
python apps/cli/main.py --db sqlite:///data/pms.db
```

## Optional Do-mode (Appendix-level)

If time permits, enable **1-2 write endpoints** (e.g., `bookAppointment`, `sendSMS`) with **JSON schemas** and an **outbox** table. Report Exact-Match JSON, Execution accuracy, Slot-F1, Hallucination rate. Otherwise, discuss writes qualitatively in the paper.

## Milestones (Read-first)

- **R0-R1**: schema/views + DSL compiler
- **R2-R3**: guardrails + dataset  
- **R4**: baselines
- **R5**: SFT + eval
- **R6**: figures & release

## Acceptance Checklist

- [ ] Seeds, dataset splits, and seeded DB committed (no PHI)
- [ ] NL→DSL model reproducibly hits done criteria
- [ ] CLI demo runs and returns tables & 1-line summaries
- [ ] `metrics.json` reproducible from README steps
- [ ] Tag `v0.1` and archive to Zenodo
