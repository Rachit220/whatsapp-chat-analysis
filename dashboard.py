# =============================================================================
# dashboard.py — HTML Report Generator
# Produces a single self-contained HTML file with embedded charts and stats
# =============================================================================

import base64
import pandas as pd
from pathlib import Path
from jinja2 import Template
from datetime import datetime

OUT_DIR = Path("outputs/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHARTS_DIR = Path("outputs/charts")


def img_b64(filename: str) -> str:
    """Convert a PNG to base64 string for HTML embedding."""
    path = CHARTS_DIR / filename
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).to_html(
        index=False, border=0,
        classes="data-table",
        na_rep="—",
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WhatsApp Chat Analysis Report</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f4f6fb;
    color: #1a1a2e;
    line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white;
    padding: 2.5rem 2rem;
    text-align: center;
  }
  .header h1 { font-size: 2rem; font-weight: 600; margin-bottom: .4rem; }
  .header p  { font-size: .95rem; opacity: .75; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  section { margin-bottom: 2.5rem; }
  h2 {
    font-size: 1.2rem; font-weight: 600;
    border-left: 4px solid #3B8BD4;
    padding-left: .8rem;
    margin-bottom: 1.2rem;
    color: #1a1a2e;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid #e8ecf4;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
  }
  .stat-card .num {
    font-size: 2rem; font-weight: 700;
    color: #3B8BD4;
  }
  .stat-card .lbl {
    font-size: .78rem; color: #6b7280;
    margin-top: .3rem;
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .chart-img {
    width: 100%; border-radius: 10px;
    border: 1px solid #e8ecf4;
    margin-bottom: 1rem;
    background: white;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  @media (max-width: 640px) { .two-col { grid-template-columns: 1fr; } }
  .data-table {
    width: 100%; border-collapse: collapse;
    background: white; border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e8ecf4;
    font-size: .88rem;
  }
  .data-table th {
    background: #f0f4ff; padding: .65rem 1rem;
    text-align: left; font-weight: 600;
    color: #374151; border-bottom: 1px solid #e8ecf4;
    font-size: .8rem; text-transform: uppercase; letter-spacing: .4px;
  }
  .data-table td {
    padding: .55rem 1rem;
    border-top: 1px solid #f3f4f6;
    color: #4b5563;
  }
  .data-table tr:hover td { background: #fafbff; }
  .badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: .75rem; font-weight: 600;
  }
  .badge-blue   { background: #dbeafe; color: #1d4ed8; }
  .badge-green  { background: #dcfce7; color: #15803d; }
  .badge-orange { background: #fef3c7; color: #b45309; }
  .footer {
    text-align: center; padding: 2rem;
    font-size: .8rem; color: #9ca3af;
    border-top: 1px solid #e8ecf4;
    margin-top: 2rem;
  }
  iframe.plotly { width:100%; height:420px; border:none; border-radius:10px; border:1px solid #e8ecf4; }
</style>
</head>
<body>

<div class="header">
  <h1>📱 WhatsApp Chat Analysis</h1>
  <p>{{ participants }} participants · {{ date_range }} · Generated {{ generated_at }}</p>
</div>

<div class="container">

  <!-- ── Summary Stats ── -->
  <section>
    <h2>Overview</h2>
    <div class="stat-grid">
      <div class="stat-card"><div class="num">{{ total_messages }}</div><div class="lbl">Total Messages</div></div>
      <div class="stat-card"><div class="num">{{ participants }}</div><div class="lbl">Participants</div></div>
      <div class="stat-card"><div class="num">{{ total_days }}</div><div class="lbl">Days Active</div></div>
      <div class="stat-card"><div class="num">{{ avg_per_day }}</div><div class="lbl">Avg / Day</div></div>
      <div class="stat-card"><div class="num">{{ peak_hour }}</div><div class="lbl">Peak Hour</div></div>
      <div class="stat-card"><div class="num">{{ busiest_day }}</div><div class="lbl">Busiest Day</div></div>
      <div class="stat-card"><div class="num">{{ total_media }}</div><div class="lbl">Media Shared</div></div>
      <div class="stat-card"><div class="num">{{ streak_days }}</div><div class="lbl">Longest Streak</div></div>
    </div>
  </section>

  <!-- ── Per Sender Stats ── -->
  <section>
    <h2>Per-Participant Statistics</h2>
    {{ stats_table }}
  </section>

  <!-- ── Activity Heatmap ── -->
  <section>
    <h2>Activity Heatmap</h2>
    {% if heatmap_b64 %}
    <img class="chart-img" src="data:image/png;base64,{{ heatmap_b64 }}" alt="Activity Heatmap">
    {% endif %}
  </section>

  <!-- ── Hourly + Message Share ── -->
  <section>
    <h2>Message Distribution</h2>
    <div class="two-col">
      {% if hourly_b64 %}
      <img class="chart-img" src="data:image/png;base64,{{ hourly_b64 }}" alt="Hourly Distribution">
      {% endif %}
      {% if share_b64 %}
      <img class="chart-img" src="data:image/png;base64,{{ share_b64 }}" alt="Message Share">
      {% endif %}
    </div>
  </section>

  <!-- ── Word Clouds ── -->
  <section>
    <h2>Word Clouds</h2>
    {% if wordcloud_all_b64 %}
    <img class="chart-img" src="data:image/png;base64,{{ wordcloud_all_b64 }}" alt="Word Cloud All">
    {% endif %}
  </section>

  <!-- ── Sentiment ── -->
  <section>
    <h2>Sentiment Analysis</h2>
    {% if sentiment_b64 %}
    <img class="chart-img" src="data:image/png;base64,{{ sentiment_b64 }}" alt="Sentiment">
    {% endif %}
    {{ sentiment_table }}
  </section>

  <!-- ── Response Time ── -->
  <section>
    <h2>Response Times</h2>
    {% if resp_time_b64 %}
    <img class="chart-img" src="data:image/png;base64,{{ resp_time_b64 }}" alt="Response Time">
    {% endif %}
    {{ resp_table }}
  </section>

  <!-- ── Top Emojis ── -->
  <section>
    <h2>Top Emojis</h2>
    {{ emoji_table }}
  </section>

  <!-- ── Top Bigrams ── -->
  <section>
    <h2>Most Common Word Pairs (Bigrams)</h2>
    {{ bigram_table }}
  </section>

</div>

<div class="footer">
  WhatsApp Chat Analysis Report · Generated {{ generated_at }} · Python Project
</div>
</body>
</html>"""


def build_report(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    sent_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    emoji_df: pd.DataFrame,
    bigram_df: pd.DataFrame,
    streak: dict,
) -> Path:
    total_days  = df["date"].nunique()
    avg_per_day = round(len(df) / max(total_days, 1), 1)
    peak_hour   = int(df["hour"].value_counts().idxmax())
    busiest_day = df["day"].value_counts().idxmax()

    html = Template(TEMPLATE).render(
        total_messages  = f"{len(df):,}",
        participants    = df["sender"].nunique(),
        total_days      = f"{total_days:,}",
        avg_per_day     = avg_per_day,
        peak_hour       = f"{peak_hour:02d}:00",
        busiest_day     = busiest_day[:3],
        total_media     = f"{df['is_media'].sum():,}",
        streak_days     = streak.get("streak", "—"),
        date_range      = f"{df['date'].min()} → {df['date'].max()}",
        generated_at    = datetime.now().strftime("%d %b %Y %H:%M"),
        stats_table     = html_table(stats_df),
        sentiment_table = html_table(sent_df),
        resp_table      = html_table(resp_df),
        emoji_table     = html_table(emoji_df),
        bigram_table    = html_table(bigram_df),
        heatmap_b64     = img_b64("heatmap.png"),
        hourly_b64      = img_b64("hourly_distribution.png"),
        share_b64       = img_b64("message_share.png"),
        wordcloud_all_b64 = img_b64("wordcloud_all.png"),
        sentiment_b64   = img_b64("sentiment.png"),
        resp_time_b64   = img_b64("response_time.png"),
    )

    out_path = OUT_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
