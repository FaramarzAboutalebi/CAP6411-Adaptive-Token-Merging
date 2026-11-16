"""Report-ready Markdown/LaTeX tables from benchmark rows and latency-matched pairs.

Speedup is the ratio of *measured* median latencies against the full-resolution
baseline measured under the same conditions; token counts are labeled
theoretical and are never turned into a speedup.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _fmt(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _escape_latex(text: str) -> str:
    for char, replacement in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                              ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(char, replacement)
    return text


def _render(header: Sequence[str], body: Sequence[Sequence[str]], fmt: str, caption: str | None) -> str:
    if fmt == "markdown":
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines) + "\n"
    if fmt == "latex":
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
            r"\toprule",
            " & ".join(_escape_latex(h) for h in header) + r" \\",
            r"\midrule",
        ]
        lines += [" & ".join(_escape_latex(c) for c in row) + r" \\" for row in body]
        lines += [r"\bottomrule", r"\end{tabular}"]
        if caption:
            lines.append(r"\caption{" + _escape_latex(caption) + "}")
        lines.append(r"\end{table}")
        return "\n".join(lines) + "\n"
    raise ValueError(f"fmt must be 'markdown' or 'latex', received {fmt!r}")


def results_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 1,
    fmt: str = "markdown",
    caption: str | None = None,
) -> str:
    """One line per variant at ``batch_size``: tokens, latency, speedup, memory, Dice.

    Rows are sorted by measured median latency (slowest first).  ``Speedup`` is
    baseline median / variant median for rows sharing the baseline's device,
    precision, and input kind; it is blank when no such baseline row exists.
    """

    selected = sorted(
        (r for r in rows if r["batch_size"] == batch_size),
        key=lambda r: -r["latency_median_ms"],
    )
    if not selected:
        raise ValueError(f"no rows at batch size {batch_size}")
    baselines = {
        (r.get("device"), r.get("precision"), r.get("input")): r["latency_median_ms"]
        for r in selected
        if r["family"] == "baseline"
    }
    has_dice = any(isinstance(r.get("macro_dice"), (int, float)) for r in selected)
    header = ["Variant", "Family", "Tokens (theor.)", "Median ms", "p90 ms", "Speedup", "Peak MB"]
    if has_dice:
        header += ["WT", "TC", "ET", "Macro Dice"]
    body = []
    for row in selected:
        base = baselines.get((row.get("device"), row.get("precision"), row.get("input")))
        speedup = base / row["latency_median_ms"] if base else None
        line = [
            str(row["variant"]),
            str(row["family"]).replace("_", " "),
            _fmt(row.get("tokens_theoretical"), 0),
            _fmt(row["latency_median_ms"], 2),
            _fmt(row.get("latency_p90_ms"), 2),
            f"{speedup:.2f}×" if speedup else "–",
            _fmt(row.get("peak_memory_mb"), 0),
        ]
        if has_dice:
            line += [_fmt(row.get(f"dice_{region}"), 3) for region in ("WT", "TC", "ET")]
            line.append(_fmt(row.get("macro_dice"), 3))
        body.append(line)
    return _render(header, body, fmt, caption)


def matched_table(
    pairs: Sequence[Mapping[str, Any]],
    *,
    fmt: str = "markdown",
    caption: str | None = None,
) -> str:
    """Latency-matched comparison from ``match_latency``: merger vs closest reference."""

    if not pairs:
        raise ValueError("no latency-matched pairs")
    has_dice = any("macro_dice_delta" in p for p in pairs)
    header = ["Merging variant", "Median ms", "Matched reference", "Ref. median ms", "Latency gap %"]
    if has_dice:
        header += ["Macro Dice", "Ref. Macro Dice", "ΔDice"]
    body = []
    for pair in pairs:
        line = [
            str(pair["variant"]),
            _fmt(pair["latency_median_ms"], 2),
            str(pair["matched_variant"]),
            _fmt(pair["matched_latency_median_ms"], 2),
            f"{pair['latency_gap_pct']:+.1f}",
        ]
        if has_dice:
            delta = pair.get("macro_dice_delta")
            line += [
                _fmt(pair.get("macro_dice"), 3),
                _fmt(pair.get("matched_macro_dice"), 3),
                f"{delta:+.3f}" if delta is not None else "–",
            ]
        body.append(line)
    return _render(header, body, fmt, caption)
