#!/usr/bin/env python3
# =============================================================================
# main.py — WhatsApp Chat Analysis Pipeline (Full Entry Point)
# Usage: python main.py --file data/raw/chat.txt
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ── Import project modules ────────────────────────────────────────────────────
from parse import parse_chat
from analyse import (
    basic_stats,
    activity_heatmap,
    top_emojis,
    sentiment_analysis,
    monthly_trend,
    response_time,
    conversation_starters,
    longest_streak,
)
from visualise import (
    plot_heatmap,
    plot_wordcloud,
    plot_monthly,
    plot_emoji_bar,
    plot_sentiment,
    plot_messages_per_sender,
    plot_hourly,
    plot_response_time,
    plot_message_share,
)
from nlp_extras import (
    vader_summary,
    top_bigrams,
    question_rate,
    late_night_msgs,
)
from dashboard import build_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def step(msg: str):
    print(f"\n{'─'*50}\n▶  {msg}")


def done(msg: str = ""):
    print(f"   ✓  {msg}" if msg else "   ✓  Done")


def separator():
    print("\n" + "═" * 55)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(filepath: str, skip_wordcloud: bool = False):
    t0 = time.time()

    separator()
    print("  📱  WhatsApp Chat Analysis Pipeline")
    print(f"       File: {filepath}")
    separator()

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    step("Parsing chat export …")
    df = parse_chat(filepath)
    df.to_csv("data/processed/chat_clean.csv", index=False)
    done(f"{len(df):,} messages · {df['sender'].nunique()} participants · "
         f"{df['date'].nunique()} days")

    # ── 2. Core Analysis ──────────────────────────────────────────────────────
    step("Computing statistics …")
    stats_df   = basic_stats(df)
    heatmap_df = activity_heatmap(df)
    emoji_df   = top_emojis(df)
    sent_df    = sentiment_analysis(df)
    trend_df   = monthly_trend(df)
    resp_df    = response_time(df)
    starters   = conversation_starters(df)
    streak     = longest_streak(df)
    done()

    # ── 3. NLP ────────────────────────────────────────────────────────────────
    step("Running NLP analysis …")
    vader_df   = vader_summary(df)
    bigram_df  = top_bigrams(df)
    q_rate     = question_rate(df)
    lnm        = late_night_msgs(df)
    done()

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    step("Generating charts …")
    plot_heatmap(heatmap_df)
    plot_messages_per_sender(stats_df)
    plot_hourly(df)
    plot_message_share(stats_df)
    plot_monthly(trend_df)
    plot_emoji_bar(emoji_df)
    plot_sentiment(sent_df)
    plot_response_time(resp_df)

    if not skip_wordcloud:
        plot_wordcloud(df)                                  # all combined
        for sender in df["sender"].unique():
            plot_wordcloud(df, sender)
    done()

    # ── 5. HTML Report ────────────────────────────────────────────────────────
    step("Building HTML report …")
    out_path = build_report(
        df, stats_df, vader_df, resp_df, emoji_df, bigram_df, streak
    )
    done(f"Report saved → {out_path}")

    # ── 6. Console Summary ────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    separator()
    print(f"\n✅  Pipeline complete in {elapsed}s")
    print(f"\n{'─'*55}")
    print("📊  Per-Participant Stats:")
    print(stats_df.to_string(index=False))

    print(f"\n⏱   Median Response Times:")
    print(resp_df.to_string(index=False))

    print(f"\n🔥  Longest Activity Streak: {streak['streak']} days "
          f"({streak['start']} → {streak['end']})")

    print(f"\n🚀  Conversation Starters:")
    print(starters.to_string(index=False))

    print(f"\n🧠  VADER Sentiment:")
    print(vader_df.to_string(index=False))

    print(f"\n❓  Question Rate:")
    print(q_rate.to_string(index=False))

    print(f"\n🌙  Late Night Messages (11pm–4am):")
    print(lnm.to_string(index=False))

    print(f"\n📂  Outputs saved to:")
    print(f"    charts  → outputs/charts/")
    print(f"    report  → outputs/reports/report.html")
    print(f"    data    → data/processed/chat_clean.csv")
    separator()
    print(f"\n   Open outputs/reports/report.html in your browser to view the full report!\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse a WhatsApp exported chat .txt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file data/raw/chat.txt
  python main.py --file data/raw/group_chat.txt --no-wordcloud
        """
    )
    parser.add_argument(
        "--file", "-f",
        default="data/raw/chat.txt",
        help="Path to WhatsApp exported .txt file (default: data/raw/chat.txt)"
    )
    parser.add_argument(
        "--no-wordcloud",
        action="store_true",
        help="Skip word cloud generation (faster run)"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"\n❌  File not found: {args.file}")
        print("    Export your WhatsApp chat:")
        print("    Android: Open chat → ⋮ Menu → More → Export Chat → Without Media")
        print("    iOS:     Open chat → Contact/Group name → Export Chat")
        print(f"\n    Then place the .txt file at: {args.file}\n")
        sys.exit(1)

    run(args.file, skip_wordcloud=args.no_wordcloud)
