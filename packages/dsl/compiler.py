from lark import Lark, Transformer, v_args
import re
from .date_resolver import resolve_rel
GRAMMAR = open(__file__.replace("compiler.py","grammar.lark"), "r").read()
PARSER = Lark(GRAMMAR, start="query")

ENTITY_MAP = {
    "appointments": "SELECT a.* FROM appointments a JOIN practitioners p USING(practitioner_id)",
    "patients":     "SELECT * FROM patients",
    "invoices":     "SELECT * FROM invoices",
    "payments":     "SELECT * FROM payments",
    "practitioners": "SELECT * FROM practitioners",
}
KPI_MAP = {
    "kpi:aged_receivables":        "SELECT * FROM vw_aged_receivables",
    "kpi:no_shows_last_7d":        "SELECT * FROM vw_no_shows_last_7d",
    "kpi:free_double_slots_next_7d": "SELECT * FROM vw_free_double_slots_next_7d",
    "kpi:revenue_by_practitioner":  "SELECT * FROM vw_revenue_by_practitioner",
}
WHITELIST_FIELDS = {"practitioner","day","time","slot","status","role"}

class AST(Transformer):
    def target(self, children):
        # Extract the actual target from the list
        return str(children[0]) if children else ""
    def ENTITY(self, token): return str(token)
    def KPI(self, token): return str(token)
    def field(self, children): return str(children[0])
    def s(self, children): return children[0][1:-1]  # strip quotes (works for both single and double)
    def n(self, children): return float(children[0])
    def list(self, children): return list(children)
    def OP(self, token): return str(token)
    def cmp(self, children): return ("cmp", children[0], children[1], children[2])
    def rel(self, children): return ("rel", str(children[0]))
    def filters(self, children): return list(children)
    def fields(self, children): return list(children)
    def order(self, children):
        # children[0] is field name, optional children[1] is direction token
        field = str(children[0]) if children else "1"
        direction = str(children[1]) if len(children) > 1 else "ASC"
        return ("ORDER", field, direction)
    def query(self, children): return children

def compile_dsl(dsl: str, conn):
    tree = PARSER.parse(dsl)
    parts = AST().transform(tree)
    
    # Handle both simple queries (string) and complex queries (list)
    if isinstance(parts, str):
        # Simple query: FIND target (no WHERE clause)
        target = parts
        where, group_by, order_by, limit = [], None, None, 50
    else:
        # Complex query: FIND target WHERE ...
        target = parts[0]
        where, group_by, order_by, limit = [], None, None, 50
        
        # Parse optional clauses by type
        for p in parts[1:]:
            if isinstance(p, list) and p and isinstance(p[0], tuple):
                where = p
            elif isinstance(p, list):
                # Currently unused (fields/grouping not implemented)
                group_by = p
            elif isinstance(p, tuple) and len(p) >= 2 and p[0] == "ORDER":
                order_by = p  # ("ORDER", field, direction)
            elif isinstance(p, int) or isinstance(p, float):
                limit = int(p)

    # Base SQL
    if target in ENTITY_MAP:
        base = ENTITY_MAP[target]
        kpi = False
    elif target in KPI_MAP:
        base = KPI_MAP[target]
        kpi = True
    else:
        raise ValueError(f"Unknown target: '{target}'")

    clauses, params = [], []
    if not kpi:
        for kind, *rest in where:
            if kind == "rel":
                start, end = resolve_rel(rest[0], conn)
                clauses.append("a.start_ts >= ? AND a.start_ts < ?")
                params += [start, end]
            else:
                f, op, val = rest
                if f not in WHITELIST_FIELDS:
                    raise ValueError(f"Field {f} not allowed")
                if f == "practitioner":
                    clauses.append("p.last_name = ?"); params.append(val)
                elif f == "day":
                    # Accept 'Mon'..'Sun' or numeric strings 0..6 (Mon=0)
                    dow = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6}.get(str(val).lower(), None)
                    if dow is None:
                        dow = int(val)
                    clauses.append("strftime('%w', a.start_ts) = ?"); params.append(str((dow+1)%7))  # sqlite Sun=0
                elif f == "time":
                    clauses.append("time(a.start_ts) " + op + " ?")
                    params.append(val if len(val)==8 else (val+":00" if len(val)==5 else val))
                elif f == "slot":
                    if str(val).lower() == "double":
                        clauses.append("a.duration_minutes >= 30")
                    else:
                        clauses.append("a.duration_minutes < 30")
                elif f == "status":
                    clauses.append("a.status " + op + " ?"); params.append(val)
                elif f == "role":
                    # Allow role filter on practitioners when joining appointments or querying practitioners
                    if target == "practitioners":
                        clauses.append("role " + op + " ?"); params.append(val)
                    else:
                        clauses.append("p.role " + op + " ?"); params.append(val)

    sql = base
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    # Apply ordering
    if order_by:
        _, field, direction = order_by
        # Basic safety: ensure field is alphanumeric/underscore
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", field):
            raise ValueError("Invalid ORDER BY field")
        if direction.upper() not in {"ASC","DESC"}:
            raise ValueError("Invalid ORDER BY direction")
        sql += f" ORDER BY {field} {direction.upper()}"
    else:
        sql += " ORDER BY 1"
    sql += f" LIMIT {limit}"
    return sql, params
