#!/usr/bin/env python3
"""Build a static HTML page from extensions.json."""
from __future__ import annotations

import json
from pathlib import Path

TAG_COLORS = {
    "tool": "#3b82f6",
    "safety": "#ef4444",
    "git": "#f97316",
    "ui": "#8b5cf6",
    "provider": "#06b6d4",
    "workflow": "#10b981",
    "game": "#ec4899",
    "observability": "#6366f1",
}

STATUS_BADGES = {
    "published": ("Published", "#22c55e"),
    "planned": ("Planned", "#eab308"),
    "builtin": ("Built-in", "#3b82f6"),
    "deprecated": ("Deprecated", "#6b7280"),
}


def build() -> None:
    root = Path(__file__).resolve().parent.parent
    registry = json.loads((root / "extensions.json").read_text(encoding="utf-8"))

    cards_html = ""
    for ext in registry["extensions"]:
        tags = " ".join(
            f'<span class="tag" style="background:{TAG_COLORS.get(t, "#6b7280")}">{t}</span>'
            for t in ext["tags"]
        )
        status = ext.get("status", "published")
        badge_label, badge_color = STATUS_BADGES.get(status, ("Unknown", "#6b7280"))
        status_badge = f'<span class="status" style="background:{badge_color}">{badge_label}</span>'

        repo_link = ""
        if ext.get("repository"):
            repo_link = f' \u00b7 <a href="{ext["repository"]}">repo</a>'

        cards_html += f"""
    <div class="card">
      <div class="card-header">
        <h3>{ext["name"]}</h3>
        {status_badge}
      </div>
      <p class="desc">{ext["description"]}</p>
      <div class="meta">
        <code>pip install {ext["pypi"]}</code>
        <span class="author">by {ext["author"]}{repo_link}</span>
      </div>
      <div class="tags">{tags}</div>
    </div>"""

    ext_count = len(registry["extensions"])

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aar Extensions</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: var(--bg); color: var(--text); padding: 2rem; max-width: 900px; margin: auto; }
    h1 { font-size: 2rem; margin-bottom: 0.25rem; }
    .subtitle { color: var(--muted); margin-bottom: 2rem; }
    .card { background: var(--card); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
             border: 1px solid #334155; }
    .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
    .card h3 { font-size: 1.1rem; color: var(--accent); }
    .desc { color: var(--text); margin-bottom: 0.75rem; }
    .meta { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
             margin-bottom: 0.5rem; color: var(--muted); font-size: 0.85rem; }
    .meta code { background: #0f172a; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
    .meta a { color: var(--accent); text-decoration: none; }
    .tags { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .tag { padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem;
            color: white; font-weight: 500; }
    .status { padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.7rem;
               color: white; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .filter { margin-bottom: 1.5rem; }
    .filter input { background: var(--card); border: 1px solid #334155; border-radius: 8px;
                     padding: 0.5rem 1rem; color: var(--text); width: 100%; font-size: 1rem; }
    .filter input::placeholder { color: var(--muted); }
    .count { color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>\U0001f985 Aar Extensions</h1>
  <p class="subtitle">Curated extensions for the Aar agent framework</p>
  <div class="filter">
    <input type="text" id="search" placeholder="Search extensions..." oninput="filterCards()">
  </div>
  <p class="count">""" + str(ext_count) + """ extension(s) registered</p>
  <div id="cards">""" + cards_html + """
  </div>
  <script>
    function filterCards() {
      const q = document.getElementById('search').value.toLowerCase();
      document.querySelectorAll('.card').forEach(c => {
        c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
      });
    }
  </script>
</body>
</html>"""

    out = root / "site" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Built {out} ({ext_count} extensions)")


if __name__ == "__main__":
    build()
