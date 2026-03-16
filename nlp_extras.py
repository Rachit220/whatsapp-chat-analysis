# =============================================================================
# nlp_extras.py — Advanced NLP: VADER Sentiment, Bigrams, Topics, Language
# =============================================================================

import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams
from collections import Counter

# Download required NLTK data (only first run)
for resource in ["vader_lexicon", "stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else resource)
    except LookupError:
        nltk.download(resource, quiet=True)

SIA = SentimentIntensityAnalyzer()

CUSTOM_STOP = {
    "media", "omitted", "ok", "okay", "yeah", "yes", "yep", "nope",
    "lol", "haha", "hahaha", "lmao", "omg", "tbh", "btw", "imo",
    "idk", "ngl", "fr", "rn", "bc", "w", "u", "r", "ur", "im",
    "ive", "dont", "ill", "its", "gonna", "wanna", "gotta", "kinda",
    "bhai", "yaar", "kya", "hai", "na", "nahi", "haan", "aur",
    "kar", "tha", "thi", "ke", "ki", "ko", "se", "pe", "bhi",
}

try:
    STOP = set(stopwords.words("english")) | CUSTOM_STOP
except Exception:
    STOP = CUSTOM_STOP


# ── VADER Sentiment ───────────────────────────────────────────────────────────

def vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply VADER sentiment to every text message.
    Returns df with extra columns: vader_compound, vader_pos, vader_neg, vader_neu, sentiment_label
    """
    def score_row(text):
        try:
            s = SIA.polarity_scores(str(text))
            return s["compound"], s["pos"], s["neg"], s["neu"]
        except Exception:
            return 0.0, 0.0, 0.0, 1.0

    df = df[~df["is_media"]].copy()
    scores = df["message"].apply(score_row)
    df[["vader_compound", "vader_pos", "vader_neg", "vader_neu"]] = pd.DataFrame(
        scores.tolist(), index=df.index
    )
    df["sentiment_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.1, -0.05, 0.05, 1.1],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


def vader_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average VADER scores per sender."""
    df_v = vader_sentiment(df)
    return df_v.groupby("sender").agg(
        avg_compound =("vader_compound", "mean"),
        pct_positive =("sentiment_label", lambda x: round((x == "Positive").mean() * 100, 1)),
        pct_neutral  =("sentiment_label", lambda x: round((x == "Neutral").mean()  * 100, 1)),
        pct_negative =("sentiment_label", lambda x: round((x == "Negative").mean() * 100, 1)),
    ).reset_index()


# ── Token cleaning ────────────────────────────────────────────────────────────

def clean_tokens(text: str) -> list:
    """Lowercase, strip URLs & punctuation, remove stopwords."""
    text = re.sub(r"http\S+|www\S+", "", str(text).lower())
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP and len(t) > 2 and not t.isdigit()]


# ── Bigrams & Trigrams ────────────────────────────────────────────────────────

def top_bigrams(df: pd.DataFrame, sender: str = None, n: int = 20) -> pd.DataFrame:
    """Most common word-pair bigrams for a sender (or all)."""
    sub = df if sender is None else df[df["sender"] == sender]
    all_tokens = []
    for msg in sub[~sub["is_media"]]["message"].dropna():
        all_tokens += clean_tokens(msg)
    bi = [" ".join(b) for b in bigrams(all_tokens)]
    return pd.DataFrame(Counter(bi).most_common(n), columns=["bigram", "count"])


def top_trigrams(df: pd.DataFrame, sender: str = None, n: int = 15) -> pd.DataFrame:
    """Most common trigrams."""
    sub = df if sender is None else df[df["sender"] == sender]
    all_tokens = []
    for msg in sub[~sub["is_media"]]["message"].dropna():
        all_tokens += clean_tokens(msg)
    tri = [" ".join(t) for t in trigrams(all_tokens)]
    return pd.DataFrame(Counter(tri).most_common(n), columns=["trigram", "count"])


# ── Word frequency per sender ─────────────────────────────────────────────────

def word_freq(df: pd.DataFrame, sender: str = None, n: int = 30) -> pd.DataFrame:
    """Top N words for a sender or overall."""
    sub = df if sender is None else df[df["sender"] == sender]
    all_tokens = []
    for msg in sub[~sub["is_media"]]["message"].dropna():
        all_tokens += clean_tokens(msg)
    return pd.DataFrame(Counter(all_tokens).most_common(n), columns=["word", "count"])


# ── Message length distribution ───────────────────────────────────────────────

def message_length_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats of word count per sender."""
    return df.groupby("sender")["word_count"].describe().round(1).reset_index()


# ── Question detection ────────────────────────────────────────────────────────

def question_rate(df: pd.DataFrame) -> pd.DataFrame:
    """% of messages containing a question mark per sender."""
    df = df[~df["is_media"]].copy()
    df["is_question"] = df["message"].str.contains(r"\?", na=False)
    result = df.groupby("sender")["is_question"].agg(
        questions  ="sum",
        total      ="count",
        question_pct=lambda x: round(x.mean() * 100, 1),
    ).reset_index()
    return result


# ── Late night messages ───────────────────────────────────────────────────────

def late_night_msgs(df: pd.DataFrame, start: int = 23, end: int = 4) -> pd.DataFrame:
    """Messages sent between start and end hour (late night / early morning)."""
    mask = (df["hour"] >= start) | (df["hour"] <= end)
    return df[mask].groupby("sender").size().reset_index(name="late_night_count")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/chat_clean.csv", parse_dates=["datetime"])
    df["is_media"] = df["is_media"].astype(bool)

    print("\n🧠 VADER Sentiment Summary:")
    print(vader_summary(df).to_string(index=False))

    print("\n📝 Top Bigrams (All):")
    print(top_bigrams(df, n=10).to_string(index=False))

    print("\n❓ Question Rate:")
    print(question_rate(df).to_string(index=False))

    print("\n🌙 Late Night Messages:")
    print(late_night_msgs(df).to_string(index=False))
