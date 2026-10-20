"""Inline engine + data into web/regime_desk.html (single self-contained page).

    python tools/export_dashboard.py && python tools/build_dashboard.py
"""
from pathlib import Path

W = Path(__file__).resolve().parents[1] / "web"
html = (W / "template.html").read_text()
data = (W / "data.json").read_text().replace("</", "<\\/")
html = html.replace("__DATA__", data).replace("__ENGINE__", (W / "engine.js").read_text())
(W / "regime_desk.html").write_text(html)
print(W / "regime_desk.html", round(len(html) / 1e6, 2), "MB")
