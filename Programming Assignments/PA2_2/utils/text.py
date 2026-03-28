"""Text formatting helpers."""


def truncate_str(s: str, max_chars: int = 200) -> str:
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def format_sample_table(rows: list[dict], columns: list[str]) -> str:
    """
    Build a simple markdown table from a list of row dicts.
    """
    header = "| " + " | ".join(columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines  = [header, sep]
    for row in rows:
        cells = [truncate_str(str(row.get(c, "")), 150) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
