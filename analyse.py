# =============================================================================
# analyse.py — Core Statistics & Analysis Engine
# =============================================================================

import re
import emoji
import collections
import pandas as pd
from textblob import TextBlob


# ── Basic per-sender stats ────────────────────────────────────────────────────

def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Message count, word totals, averages, media sent per sender."""
    stats = df.groupby("sender").agg(
        messages   =("message",    "count"),
        words      =("word_count", "sum"),
        avg_words  =("word_count", "mean"),
        media_sent =("is_media",   "sum"),
        chars      =("char_count", "sum"),
    ).reset_index()
    stats["avg_words"]  = stats["avg_words"].round(1)
    stats["share_%"]    = (stats["messages"] / stats["messages"].sum() * 100).round(1)
    return stats.sort_values("messages", ascending=False)


# ── Activity heatmap ──────────────────────────────────────────────────────────

def activity_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = hour (0-23), cols = day of week, values = message count."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.groupby(["hour", "day"]).size().unstack(fill_value=0)
    return pivot.reindex(columns=[d for d in days if d in pivot.columns], fill_value=0)


# ── Emoji analysis ────────────────────────────────────────────────────────────

def top_emojis(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Extract and count all emojis across all messages."""
    all_emojis = []
    for msg in df["message"].dropna():
        all_emojis += [c for c in msg if c in emoji.EMOJI_DATA]
    counter = collections.Counter(all_emojis)
    return pd.DataFrame(counter.most_common(n), columns=["emoji", "count"])


def emojis_per_sender(df: pd.DataFrame, n: int = 10) -> dict:
    """Top emojis used by each sender."""
    result = {}
    for sender, group in df.groupby("sender"):
        all_emojis = []
        for msg in group["message"].dropna():
            all_emojis += [c for c in msg if c in emoji.EMOJI_DATA]
        counter = collections.Counter(all_emojis)
        result[sender] = pd.DataFrame(counter.most_common(n), columns=["emoji", "count"])
    return result


# ── Sentiment analysis ────────────────────────────────────────────────────────

def sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """TextBlob polarity score per sender (−1 = negative, +1 = positive)."""
    def score(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0

    text_df = df[~df["is_media"]].copy()
    text_df["polarity"] = text_df["message"].apply(score)
    result = text_df.groupby("sender")["polarity"].agg(["mean", "std", "count"]).reset_index()
    result.columns = ["sender", "avg_polarity", "std_polarity", "msg_count"]
    result["avg_polarity"] = result["avg_polarity"].round(3)
    return result


# ── Monthly / weekly trend ────────────────────────────────────────────────────

def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Message count per sender per month."""
    return df.groupby(["month", "sender"]).size().reset_index(name="count")


def weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Message count per week."""
    df = df.copy()
    df["week_start"] = df["datetime"].dt.to_period("W").apply(lambda r: r.start_time)
    return df.groupby("week_start").size().reset_index(name="count")


# ── Response time ─────────────────────────────────────────────────────────────

def response_time(df: pd.DataFrame) -> pd.DataFrame:
    """Median response time in minutes per sender (replies only, <24h gap)."""
    df = df.sort_values("datetime").copy()
    df["prev_sender"] = df["sender"].shift(1)
    df["prev_time"]   = df["datetime"].shift(1)
    replies = df[df["sender"] != df["prev_sender"]].copy()
    replies["resp_sec"] = (replies["datetime"] - replies["prev_time"]).dt.total_seconds()
    replies = replies[(replies["resp_sec"] > 0) & (replies["resp_sec"] < 86400)]
    result = replies.groupby("sender")["resp_sec"].agg(
        median_min=lambda x: round(x.median() / 60, 1),
        mean_min  =lambda x: round(x.mean()   / 60, 1),
    ).reset_index()
    return result


# ── Conversation starters ─────────────────────────────────────────────────────

def conversation_starters(df: pd.DataFrame, gap_hours: float = 6.0) -> pd.DataFrame:
    """Count how many conversations each sender started (gap > gap_hours)."""
    df = df.sort_values("datetime").copy()
    df["gap_hrs"] = df["datetime"].diff().dt.total_seconds().div(3600)
    starters = df[(df["gap_hrs"] > gap_hours) | df["gap_hrs"].isna()]
    return starters["sender"].value_counts().reset_index().rename(
        columns={"index": "sender", "sender": "count"}
    )


# ── Longest streak ────────────────────────────────────────────────────────────

def longest_streak(df: pd.DataFrame) -> dict:
    """Find the longest consecutive day streak of activity."""
    dates = sorted(df["date"].unique())
    if not dates:
        return {"streak": 0, "start": None, "end": None}

    best_streak, best_start, best_end = 1, dates[0], dates[0]
    streak, start = 1, dates[0]

    for i in range(1, len(dates)):
        delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
        if delta == 1:
            streak += 1
            if streak > best_streak:
                best_streak = streak
                best_start  = start
                best_end    = dates[i]
        else:
            streak = 1
            start  = dates[i]

    return {"streak": best_streak, "start": best_start, "end": best_end}


if __name__ == "__main__":
    df = pd.read_csv("data/processed/chat_clean.csv", parse_dates=["datetime"])
    df["is_media"] = df["is_media"].astype(bool)

    print("\n📊 Basic Stats:")
    print(basic_stats(df).to_string(index=False))

    print("\n⏱  Response Times (mins):")
    print(response_time(df).to_string(index=False))

    print("\n🏆 Conversation Starters:")
    print(conversation_starters(df).to_string(index=False))

    streak = longest_streak(df)
    print(f"\n🔥 Longest Streak: {streak['streak']} days ({streak['start']} → {streak['end']})")
