from __future__ import annotations


def choose_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column name (case-insensitive) from candidates."""
    lowered = {column.lower(): column for column in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None
