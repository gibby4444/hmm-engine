"""Shared HTML report scaffolding: navy background, teal accents."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
NAVY, PANEL, TEAL, TEXT, MUTED = "#0b1530", "#131f42", "#2dd4bf", "#e6edf7", "#93a3c4"
STATE_COLORS = ["#2dd4bf", "#60a5fa", "#f59e0b", "#f43f5e", "#a78bfa", "#94a3b8"]

CSS = f"""
body{{background:{NAVY};color:{TEXT};font:15px/1.5 -apple-system,Segoe UI,Inter,sans-serif;margin:0}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{color:{TEAL};font-size:18px;margin:34px 0 10px;
border-bottom:1px solid #22325f;padding-bottom:6px}} h3{{font-size:15px;margin:18px 0 6px}}
.sub{{color:{MUTED};font-size:13px}} .card{{background:{PANEL};border:1px solid #22325f;
border-radius:10px;padding:14px 16px;margin:12px 0}} .verdict{{border-left:4px solid {TEAL}}}
.warn{{border-left:4px solid #f59e0b}} .bad{{border-left:4px solid #f43f5e}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 8px;border-bottom:1px solid #22325f;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:{TEAL};font-weight:600}}
tr.best td{{background:rgba(45,212,191,.12)}} img{{max-width:100%;border-radius:8px}}
.tablewrap{{overflow-x:auto}} code{{color:{TEAL}}} ul{{margin:6px 0 6px 18px;padding:0}}
"""


def fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def styled_ax(figsize=(11, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=NAVY)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#22325f")
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax


def table(df: pd.DataFrame, fmt: dict | None = None, best_row=None) -> str:
    fmt = fmt or {}
    rows = []
    head = "".join(f"<th>{c}</th>" for c in [df.index.name or ""] + list(df.columns))
    for idx, r in df.iterrows():
        cells = [f"<td>{idx}</td>"]
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                s = (f or "{:.3f}").format(v)
            elif isinstance(v, (float, np.floating)):
                s = "—"
            else:
                s = str(v)
            cells.append(f"<td>{s}</td>")
        cls = ' class="best"' if best_row is not None and idx == best_row else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="tablewrap"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def card(html: str, kind: str = "") -> str:
    return f'<div class="card {kind}">{html}</div>'


def write(name: str, title: str, body: str, subtitle: str = "") -> Path:
    REPORTS.mkdir(exist_ok=True)
    fp = REPORTS / name
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fp.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{CSS}</style></head><body><main><h1>{title}</h1>
<div class="sub">{subtitle} · generated {stamp}</div>{body}</main></body></html>""")
    return fp


PERF_FMT = {"cagr": "{:.1%}", "vol": "{:.1%}", "sharpe": "{:.2f}", "sortino": "{:.2f}",
            "max_dd": "{:.1%}", "calmar": "{:.2f}", "exposure_avg": "{:.0%}",
            "cost_drag_ann": "{:.2%}", "trades": "{:.0f}", "switches": "{:.0f}"}
