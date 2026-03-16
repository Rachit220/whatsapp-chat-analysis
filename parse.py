# =============================================================================
# parse.py — WhatsApp Chat Parser
# Supports Android & iOS export formats (single & group chats)
# =============================================================================

import re
import pandas as pd
from pathlib import Path


# ── Regex patterns ────────────────────────────────────────────────────────────
ANDROID_PATTERN = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[ap]m)?)\s-\s([^:]+):\s(.+)$',
    re.IGNORECASE
)
IOS_PATTERN = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}(?:\s?[AP]M)?)\]\s([^:]+):\s(.+)$',
    re.IGNORECASE
)
SYSTEM_MSGS = [
    'Messages and calls are end-to-end encrypted',
    'changed the group name',
    'added',
    'removed',
    'left',
    'changed this group',
    'created group',
    'changed the subject',
    'pinned a message',
    'turned on disappearing',
]


def is_system_message(text: str) -> bool:
    return any(kw.lower() in text.lower() for kw in SYSTEM_MSGS)


def parse_chat(filepath: str) -> pd.DataFrame:
    """
    Parse a WhatsApp exported .txt file into a clean DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the exported WhatsApp chat .txt file.

    Returns
    -------
    pd.DataFrame with columns:
        datetime, sender, message, date, hour, day, month,
        is_media, char_count, word_count
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Chat file not found: {filepath}")

    rows = []
    current = None

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = ANDROID_PATTERN.match(line) or IOS_PATTERN.match(line)

            if m:
                if current:
                    rows.append(current)
                date, time, sender, msg = (
                    m.group(1), m.group(2),
                    m.group(3).strip(), m.group(4).strip()
                )
                current = {
                    "date_str": date,
                    "time_str": time,
                    "sender": sender,
                    "message": msg,
                }
            elif current:
                # Multi-line message continuation
                current["message"] += "\n" + line

    if current:
        rows.append(current)

    if not rows:
        raise ValueError(
            "No messages parsed. Check the file format matches Android or iOS export."
        )

    df = pd.DataFrame(rows)

    # ── Parse datetime ────────────────────────────────────────────────────────
    df["datetime"] = pd.to_datetime(
        df["date_str"] + " " + df["time_str"],
        dayfirst=True,
        errors="coerce",
        infer_datetime_format=True,
    )
    df = df.dropna(subset=["datetime"]).copy()

    # ── Feature engineering ───────────────────────────────────────────────────
    df["date"]       = df["datetime"].dt.date
    df["hour"]       = df["datetime"].dt.hour
    df["day"]        = df["datetime"].dt.day_name()
    df["month"]      = df["datetime"].dt.to_period("M").astype(str)
    df["year"]       = df["datetime"].dt.year
    df["week"]       = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_media"]   = df["message"].str.contains(
        r"<Media omitted>|<image omitted>|<video omitted>|<audio omitted>",
        na=False, flags=re.IGNORECASE
    )
    df["is_system"]  = df["message"].apply(is_system_message)
    df["char_count"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()

    # ── Drop system messages ──────────────────────────────────────────────────
    df = df[~df["is_system"]].reset_index(drop=True)
    df = df.drop(columns=["date_str", "time_str"])

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/chat.txt"
    df = parse_chat(path)
    df.to_csv("data/processed/chat_clean.csv", index=False)
    print(f"✅ Parsed {len(df):,} messages from {df['sender'].nunique()} participants")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Participants: {', '.join(df['sender'].unique())}")
