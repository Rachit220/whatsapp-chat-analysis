# =============================================================================
# visualise.py — Chart & Visualisation Builder
# Generates PNG charts (matplotlib/seaborn) and interactive HTML (plotly)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("outputs/charts")
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = ["#3B8BD4", "#1D9E75", "#E8593C", "#BA7517", "#534AB7",
           "#D4537E", "#888780", "#639922", "#E24B4A"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})


# ── Activity Heatmap ──────────────────────────────────────────────────────────

def plot_heatmap(pivot_df: pd.DataFrame):
    """Hour × Day activity heatmap saved as PNG."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        pivot_df, cmap="YlOrRd", linewidths=0.3, linecolor="white",
        ax=ax, cbar_kws={"label": "Messages", "shrink": 0.8},
        annot=True, fmt="d", annot_kws={"size": 8},
    )
    ax.set_title("Message Activity Heatmap (Hour × Day)", fontsize=14, pad=14)
    ax.set_xlabel("Day of Week", labelpad=8)
    ax.set_ylabel("Hour of Day (24h)", labelpad=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ heatmap.png")


# ── Word Cloud ────────────────────────────────────────────────────────────────

EXTRA_STOPWORDS = STOPWORDS | {
    "media", "omitted", "message", "deleted", "this", "that",
    "will", "just", "like", "ok", "okay", "yeah", "yes", "no",
    "hey", "hi", "hello", "lol", "haha", "im", "ive", "dont",
    "ill", "its", "ur", "u", "r", "bhai", "yaar", "kya", "hai",
    "na", "nahi", "haan", "aur", "karo", "kar", "tha", "thi",
}

def plot_wordcloud(df: pd.DataFrame, sender: str = None):
    """Word cloud for a specific sender or all messages."""
    sub  = df if sender is None else df[df["sender"] == sender]
    text = " ".join(sub[~sub["is_media"]]["message"].dropna().astype(str))

    if len(text.strip()) < 50:
        print(f"  ⚠ Skipping word cloud for '{sender}' — not enough text.")
        return

    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        stopwords=EXTRA_STOPWORDS,
        colormap="viridis",
        max_words=200,
        collocations=True,
        prefer_horizontal=0.85,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    label = sender if sender else "All Participants"
    ax.set_title(f"Word Cloud — {label}", fontsize=14, pad=10)
    name  = (sender or "all").replace(" ", "_")
    fig.savefig(OUT / f"wordcloud_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ wordcloud_{name}.png")


# ── Bar: Messages per Sender ──────────────────────────────────────────────────

def plot_messages_per_sender(stats_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        stats_df["sender"], stats_df["messages"],
        color=PALETTE[:len(stats_df)], edgecolor="none"
    )
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=10)
    ax.set_title("Total Messages per Participant", fontsize=14, pad=12)
    ax.set_xlabel("Message Count")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / "messages_per_sender.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ messages_per_sender.png")


# ── Monthly Trend (Plotly interactive) ────────────────────────────────────────

def plot_monthly(trend_df: pd.DataFrame):
    fig = px.line(
        trend_df, x="month", y="count", color="sender",
        title="Monthly Message Trend",
        markers=True, template="plotly_white",
        labels={"count": "Messages", "month": "Month", "sender": "Participant"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        hovermode="x unified",
        legend_title="Participant",
        title_font_size=15,
        xaxis_tickangle=-35,
    )
    fig.update_traces(line_width=2.2, marker_size=5)
    fig.write_html(str(OUT / "monthly_trend.html"))
    print("  ✓ monthly_trend.html")


# ── Emoji Bar Chart (Plotly interactive) ──────────────────────────────────────

def plot_emoji_bar(emoji_df: pd.DataFrame):
    top = emoji_df.head(20)
    fig = px.bar(
        top, x="emoji", y="count",
        title="Top 20 Emojis Used",
        template="plotly_white",
        color="count",
        color_continuous_scale="Viridis",
        labels={"count": "Count", "emoji": "Emoji"},
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=15,
        coloraxis_showscale=False,
    )
    fig.update_traces(marker_line_width=0)
    fig.write_html(str(OUT / "emoji_bar.html"))
    print("  ✓ emoji_bar.html")


# ── Sentiment Chart ───────────────────────────────────────────────────────────

def plot_sentiment(sent_df: pd.DataFrame):
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in sent_df["avg_polarity"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(sent_df) * 0.8)))
    bars = ax.barh(sent_df["sender"], sent_df["avg_polarity"],
                   color=colors, edgecolor="none")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.axvline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax.set_title("Average Sentiment Score per Participant", fontsize=14, pad=12)
    ax.set_xlabel("Polarity  (−1 = very negative  ·  0 = neutral  ·  +1 = very positive)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / "sentiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ sentiment.png")


# ── Hourly Distribution ───────────────────────────────────────────────────────

def plot_hourly(df: pd.DataFrame):
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(hourly["hour"], hourly["count"], alpha=0.3, color="#3B8BD4")
    ax.plot(hourly["hour"], hourly["count"], color="#3B8BD4", linewidth=2.2, marker="o", markersize=4)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.set_title("Messages by Hour of Day", fontsize=14, pad=12)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Messages")
    plt.tight_layout()
    fig.savefig(OUT / "hourly_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ hourly_distribution.png")


# ── Response Time Bar ─────────────────────────────────────────────────────────

def plot_response_time(resp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, max(4, len(resp_df) * 0.8)))
    bars = ax.barh(
        resp_df["sender"], resp_df["median_min"],
        color="#534AB7", edgecolor="none"
    )
    ax.bar_label(bars, fmt="%.1f min", padding=4, fontsize=10)
    ax.set_title("Median Response Time per Participant", fontsize=14, pad=12)
    ax.set_xlabel("Minutes")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / "response_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ response_time.png")


# ── Pie: Share of messages ────────────────────────────────────────────────────

def plot_message_share(stats_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        stats_df["messages"],
        labels=stats_df["sender"],
        autopct="%1.1f%%",
        colors=PALETTE[:len(stats_df)],
        startangle=140,
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Message Share by Participant", fontsize=14, pad=14)
    plt.tight_layout()
    fig.savefig(OUT / "message_share.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ message_share.png")
