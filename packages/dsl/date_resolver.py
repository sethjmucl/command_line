from datetime import datetime, timedelta

def get_anchor(conn) -> datetime:
    (s,) = conn.execute("SELECT ref_date FROM meta").fetchone()
    return datetime.fromisoformat(s)

def week_bounds(d: datetime):
    start = d - timedelta(days=(d.weekday()))        # Monday
    end = start + timedelta(days=7)
    return start, end

def resolve_rel(token: str, conn):
    """Returns (start_iso, end_iso) half-open [start, end)."""
    anchor = get_anchor(conn)
    token = token.lower()
    if token == "last_7d":
        return (anchor - timedelta(days=7)).isoformat(" "), (anchor + timedelta(days=1)).isoformat(" ")
    if token == "last_30d":
        return (anchor - timedelta(days=30)).isoformat(" "), (anchor + timedelta(days=1)).isoformat(" ")
    if token in {"this_week","last_week","next_week"}:
        shift = {"last_week": -7, "this_week": 0, "next_week": 7}[token]
        start, end = week_bounds(anchor + timedelta(days=shift))
        return start.isoformat(" "), end.isoformat(" ")
    raise ValueError(f"Unknown RELDATE {token}")
