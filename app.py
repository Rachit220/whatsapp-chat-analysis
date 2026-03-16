# =============================================================================
# app.py — WhatsApp Chat Analyzer (Polished Edition)
# Features: Banner, Dark Mode Toggle, Hindi/Gujarati Language Support
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
import emoji, re, os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

from parse import parse_chat
from analyse import (basic_stats, activity_heatmap, top_emojis,
                     sentiment_analysis, monthly_trend, response_time,
                     conversation_starters, longest_streak)
from nlp_extras import vader_summary, top_bigrams, question_rate, late_night_msgs

# ═════════════════════════════════════════════════════════════════════════════
# LANGUAGE STRINGS
# ═════════════════════════════════════════════════════════════════════════════
LANGUAGES = {
    "English 🇬🇧": {
        "app_title":       "💬 WhatsApp Chat Analyzer",
        "app_sub":         "Upload your WhatsApp chat export and get instant insights!",
        "upload_label":    "Choose your WhatsApp .txt file",
        "how_export":      "📖 How to export?",
        "settings":        "⚙️ Settings",
        "show_wc":         "Show Word Clouds",
        "top_emojis":      "Top N Emojis",
        "min_words":       "Min words filter",
        "dark_mode":       "🌙 Dark Mode",
        "parsed_ok":       "✅ Parsed **{msgs}** messages from **{parts}** participants across **{days}** days!",
        "tab_stats":       "📊 Stats",
        "tab_activity":    "🗺 Activity",
        "tab_words":       "☁️ Words",
        "tab_emojis":      "😀 Emojis",
        "tab_sentiment":   "🧠 Sentiment",
        "tab_trends":      "📈 Trends",
        "total_messages":  "💬 Total Messages",
        "participants":    "👥 Participants",
        "days_active":     "📅 Days Active",
        "avg_day":         "📨 Avg / Day",
        "peak_hour":       "⏰ Peak Hour",
        "longest_streak":  "🔥 Longest Streak",
        "per_stats":       "### 👥 Per-Participant Statistics",
        "msg_count":       "### 📊 Message Count",
        "msg_share":       "### 🥧 Message Share",
        "resp_times":      "### ⏱ Response Times",
        "conv_start":      "### 🚀 Conversation Starters",
        "heatmap":         "### 🗺 Activity Heatmap (Hour × Day)",
        "hourly":          "### ⏰ Messages by Hour",
        "by_day":          "### 📅 Messages by Day of Week",
        "wc_all":          "### ☁️ Word Cloud — All Messages",
        "wc_per":          "### ☁️ Word Cloud per Participant",
        "bigrams":         "### 📝 Top Bigrams (Word Pairs)",
        "top_emoji":       "### 😀 Top Emojis Used",
        "q_rate":          "### ❓ Question Rate per Participant",
        "late_night":      "### 🌙 Late Night Messages (11pm–4am)",
        "sentiment":       "### 🧠 VADER Sentiment Analysis",
        "sent_time":       "### 📊 Sentiment Over Time",
        "monthly":         "### 📈 Monthly Message Trend",
        "monthly_all":     "### 📆 Messages Per Month",
        "streak":          "### 🔥 Streak Info",
        "download":        "### 💾 Download Data",
        "dl_chat":         "📥 Download Cleaned Chat CSV",
        "dl_stats":        "📥 Download Stats CSV",
        "footer":          "Built with ❤️ by Rachit Prajapati — Turning conversations into insights, one chat at a time. ✨",
        "welcome":         "👈 Upload your WhatsApp .txt file from the sidebar to get started!",
        "android":         "**Android:**\n1. Open any chat\n2. Tap ⋮ (3 dots)\n3. More → Export Chat\n4. Without Media\n5. Save the .txt file",
        "iphone":          "**iPhone:**\n1. Open any chat\n2. Tap contact name\n3. Export Chat\n4. Without Media",
    },
    "हिंदी 🇮🇳": {
        "app_title":       "💬 व्हाट्सएप चैट विश्लेषक",
        "app_sub":         "अपनी व्हाट्सएप चैट अपलोड करें और तुरंत जानकारी पाएं!",
        "upload_label":    "अपनी व्हाट्सएप .txt फ़ाइल चुनें",
        "how_export":      "📖 एक्सपोर्ट कैसे करें?",
        "settings":        "⚙️ सेटिंग्स",
        "show_wc":         "वर्ड क्लाउड दिखाएं",
        "top_emojis":      "शीर्ष N इमोजी",
        "min_words":       "न्यूनतम शब्द फ़िल्टर",
        "dark_mode":       "🌙 डार्क मोड",
        "parsed_ok":       "✅ **{parts}** प्रतिभागियों से **{msgs}** संदेश, **{days}** दिनों में पार्स किए!",
        "tab_stats":       "📊 आँकड़े",
        "tab_activity":    "🗺 गतिविधि",
        "tab_words":       "☁️ शब्द",
        "tab_emojis":      "😀 इमोजी",
        "tab_sentiment":   "🧠 भावना",
        "tab_trends":      "📈 रुझान",
        "total_messages":  "💬 कुल संदेश",
        "participants":    "👥 प्रतिभागी",
        "days_active":     "📅 सक्रिय दिन",
        "avg_day":         "📨 औसत / दिन",
        "peak_hour":       "⏰ सर्वाधिक समय",
        "longest_streak":  "🔥 सबसे लंबा क्रम",
        "per_stats":       "### 👥 प्रतिभागी आँकड़े",
        "msg_count":       "### 📊 संदेश गणना",
        "msg_share":       "### 🥧 संदेश हिस्सा",
        "resp_times":      "### ⏱ प्रतिक्रिया समय",
        "conv_start":      "### 🚀 बातचीत शुरू करने वाले",
        "heatmap":         "### 🗺 गतिविधि हीटमैप",
        "hourly":          "### ⏰ घंटे के अनुसार संदेश",
        "by_day":          "### 📅 दिन के अनुसार संदेश",
        "wc_all":          "### ☁️ वर्ड क्लाउड — सभी संदेश",
        "wc_per":          "### ☁️ प्रतिभागी वर्ड क्लाउड",
        "bigrams":         "### 📝 शीर्ष शब्द जोड़े",
        "top_emoji":       "### 😀 शीर्ष इमोजी",
        "q_rate":          "### ❓ प्रश्न दर",
        "late_night":      "### 🌙 रात के संदेश (11pm–4am)",
        "sentiment":       "### 🧠 भावना विश्लेषण",
        "sent_time":       "### 📊 समय के साथ भावना",
        "monthly":         "### 📈 मासिक संदेश रुझान",
        "monthly_all":     "### 📆 प्रति माह संदेश",
        "streak":          "### 🔥 क्रम जानकारी",
        "download":        "### 💾 डेटा डाउनलोड करें",
        "dl_chat":         "📥 चैट CSV डाउनलोड करें",
        "dl_stats":        "📥 आँकड़े CSV डाउनलोड करें",
        "footer":          "💬 व्हाट्सएप चैट विश्लेषक · Python & Streamlit · आपका डेटा 100% सुरक्षित 🔒",
        "welcome":         "👈 शुरू करने के लिए साइडबार से अपनी .txt फ़ाइल अपलोड करें!",
        "android":         "**Android:**\n1. कोई भी चैट खोलें\n2. ⋮ टैप करें\n3. More → Export Chat\n4. Without Media\n5. .txt फ़ाइल सेव करें",
        "iphone":          "**iPhone:**\n1. चैट खोलें\n2. संपर्क नाम टैप करें\n3. Export Chat\n4. Without Media",
    },
    "ગુજરાતી 🇮🇳": {
        "app_title":       "💬 વૉટ્સએપ ચૅટ વિશ્લેષક",
        "app_sub":         "તમારી વૉટ્સએપ ચૅટ અપલોડ કરો અને તરત જ માહિતી મેળવો!",
        "upload_label":    "તમારી વૉટ્સએપ .txt ફાઇલ પસંદ કરો",
        "how_export":      "📖 એક્સપોર્ટ કેવી રીતે કરવું?",
        "settings":        "⚙️ સેટિંગ્સ",
        "show_wc":         "વર્ડ ક્લાઉડ બતાવો",
        "top_emojis":      "ટોચના N ઇમોજી",
        "min_words":       "લઘુત્તમ શબ્દ ફિલ્ટર",
        "dark_mode":       "🌙 ડાર્ક મોડ",
        "parsed_ok":       "✅ **{parts}** સભ્યો પાસેથી **{msgs}** સંદેશા, **{days}** દિવસોમાં પાર્સ થયા!",
        "tab_stats":       "📊 આંકડા",
        "tab_activity":    "🗺 પ્રવૃત્તિ",
        "tab_words":       "☁️ શબ્દો",
        "tab_emojis":      "😀 ઇમોજી",
        "tab_sentiment":   "🧠 લાગણી",
        "tab_trends":      "📈 વલણ",
        "total_messages":  "💬 કુલ સંદેશા",
        "participants":    "👥 સભ્યો",
        "days_active":     "📅 સક્રિય દિવસો",
        "avg_day":         "📨 સરેરાશ / દિવસ",
        "peak_hour":       "⏰ મહત્તમ સમય",
        "longest_streak":  "🔥 સૌથી લાંબો ક્રમ",
        "per_stats":       "### 👥 સભ્ય આંકડા",
        "msg_count":       "### 📊 સંદેશ ગણતરી",
        "msg_share":       "### 🥧 સંદેશ હિસ્સો",
        "resp_times":      "### ⏱ પ્રતિભાવ સમય",
        "conv_start":      "### 🚀 વાર્તાલાપ શરૂ કરનાર",
        "heatmap":         "### 🗺 પ્રવૃત્તિ હીટમૅપ",
        "hourly":          "### ⏰ કલાક મુજબ સંદેશા",
        "by_day":          "### 📅 દિવસ મુજબ સંદેશા",
        "wc_all":          "### ☁️ વર્ડ ક્લાઉડ — બધા સંદેશા",
        "wc_per":          "### ☁️ સભ્ય વર્ડ ક્લાઉડ",
        "bigrams":         "### 📝 ટોચના શબ્દ જોડી",
        "top_emoji":       "### 😀 ટોચના ઇમોજી",
        "q_rate":          "### ❓ પ્રશ્ન દર",
        "late_night":      "### 🌙 મોડી રાત્રિના સંદેશા",
        "sentiment":       "### 🧠 લાગણી વિશ્લેષણ",
        "sent_time":       "### 📊 સમય સાથે લાગણી",
        "monthly":         "### 📈 માસિક સંદેશ વલણ",
        "monthly_all":     "### 📆 દર મહિને સંદેશા",
        "streak":          "### 🔥 ક્રમ માહિતી",
        "download":        "### 💾 ડેટા ડાઉનલોડ કરો",
        "dl_chat":         "📥 ચૅટ CSV ડાઉનલોડ કરો",
        "dl_stats":        "📥 આંકડા CSV ડાઉનલોડ કરો",
        "footer":          "💬 વૉટ્સએપ ચૅટ વિશ્લેષક · Python & Streamlit · તમારો ડેટા 100% સુરક્ષિત 🔒",
        "welcome":         "👈 શરૂ કરવા માટે સાઇડબારથી .txt ફાઇલ અપલોડ કરો!",
        "android":         "**Android:**\n1. કોઈ પણ ચૅટ ખોલો\n2. ⋮ ટૅપ કરો\n3. More → Export Chat\n4. Without Media\n5. .txt ફાઇલ સેવ કરો",
        "iphone":          "**iPhone:**\n1. ચૅટ ખોલો\n2. સંપર્ક નામ ટૅપ કરો\n3. Export Chat\n4. Without Media",
    },
}

LANG_STOP = {
    "English 🇬🇧": set(),
    "हिंदी 🇮🇳": {
        "hai","hain","tha","thi","ko","ki","ka","ke","se","me","mein",
        "par","aur","ya","bhi","to","na","nahi","nhi","kya","koi","kuch",
        "ab","bas","haan","ok","okay","bhai","yaar","toh","woh","yeh",
        "ye","vo","jo","jab","tab","phir","fir","mat","sab","sabhi",
        "sirf","bahut","bohot","thoda","accha","acha","theek",
    },
    "ગુજરાતી 🇮🇳": {
        "che","chhe","htu","hati","hata","mate","ane","pan","ke","jo",
        "tya","tyare","pachi","have","nathi","nahi","aa","tame","hun",
        "ame","tamne","mane","enu","ena","eni","tenu","tema","karo",
        "kari","karu","thay","thashe","thayu","hoy","bahu","thodu",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Language & Dark Mode FIRST
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / भाषा / ભાષા", list(LANGUAGES.keys()))
    T = LANGUAGES[lang_choice]
    dark_mode = st.toggle(T["dark_mode"], value=False)

# ═════════════════════════════════════════════════════════════════════════════
# THEME
# ═════════════════════════════════════════════════════════════════════════════
if dark_mode:
    BG, CARD, TEXT, MUTED = "#0e1117", "#1a1d27", "#e8eaf0", "#9097a8"
    BORDER, ACCENT        = "#2e3244", "#25D366"
    HDR = "linear-gradient(135deg,#0d2137 0%,#0a3d2e 100%)"
    PLOT_TMPL, WC_BG, WC_CM = "plotly_dark", "#1a1d27", "cool"
else:
    BG, CARD, TEXT, MUTED = "#ffffff", "#f8f9fa", "#1a1a2e", "#6b7280"
    BORDER, ACCENT        = "#e9ecef", "#25D366"
    HDR = "linear-gradient(135deg,#075E54 0%,#128C7E 60%,#25D366 100%)"
    PLOT_TMPL, WC_BG, WC_CM = "plotly_white", "white", "viridis"

st.markdown(f"""<style>
.stApp{{background:{BG};color:{TEXT}}}
section[data-testid="stSidebar"]{{background:{CARD};border-right:1px solid {BORDER}}}
.stMarkdown,.stText,p,h1,h2,h3,h4,label{{color:{TEXT}!important}}
.wa-banner{{background:{HDR};border-radius:18px;padding:2.2rem 2rem 1.8rem;
  text-align:center;margin-bottom:1.5rem;position:relative;overflow:hidden}}
.wa-banner::before{{content:'';position:absolute;top:-40px;right:-40px;
  width:180px;height:180px;background:rgba(255,255,255,0.06);border-radius:50%}}
.wa-logo{{font-size:3.5rem;display:block;margin-bottom:.4rem}}
.wa-title{{font-size:2.2rem;font-weight:800;color:white!important;
  text-shadow:0 2px 8px rgba(0,0,0,0.3)}}
.wa-sub{{font-size:1rem;color:rgba(255,255,255,0.82)!important;margin-top:.4rem}}
.wa-badges{{margin-top:1rem;display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap}}
.wa-badge{{background:rgba(255,255,255,0.18);color:white!important;padding:4px 14px;
  border-radius:20px;font-size:.78rem;font-weight:600;border:1px solid rgba(255,255,255,0.25)}}
div[data-testid="metric-container"]{{background:{CARD};border:1px solid {BORDER};
  border-radius:12px;padding:1rem;text-align:center}}
div[data-testid="metric-container"] label{{color:{MUTED}!important;font-size:.8rem!important}}
div[data-testid="metric-container"] [data-testid="metric-value"]{{color:{ACCENT}!important;
  font-size:1.8rem!important;font-weight:700!important}}
.stTabs [data-baseweb="tab-list"]{{gap:6px}}
.stTabs [data-baseweb="tab"]{{border-radius:8px;padding:8px 16px;
  background:{CARD};color:{TEXT}!important;border:1px solid {BORDER}}}
.stTabs [aria-selected="true"]{{background:{ACCENT}!important;color:white!important;
  border-color:{ACCENT}!important}}
.stButton>button,.stDownloadButton>button{{border-radius:10px;background:{ACCENT};
  color:white;border:none;font-weight:600}}
</style>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# BANNER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="wa-banner">
  <span class="wa-logo">💬</span>
  <div class="wa-title">{T['app_title']}</div>
  <div class="wa-sub">{T['app_sub']}</div>
  <div class="wa-badges">
    <span class="wa-badge">📊 Statistics</span>
    <span class="wa-badge">🧠 NLP Sentiment</span>
    <span class="wa-badge">☁️ Word Cloud</span>
    <span class="wa-badge">😀 Emoji Analysis</span>
    <span class="wa-badge">🔒 100% Private</span>
  </div>
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — rest of controls
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    uploaded_file = st.file_uploader(T["upload_label"], type=["txt"])
    st.markdown("---")
    st.markdown(f"### {T['how_export']}")
    st.markdown(T["android"])
    st.markdown(T["iphone"])
    st.markdown("---")
    st.markdown(f"### {T['settings']}")
    show_wc      = st.checkbox(T["show_wc"], value=True)
    top_n        = st.slider(T["top_emojis"], 5, 30, 15)
    min_w        = st.slider(T["min_words"], 1, 10, 1)
    st.markdown("---")
    st.caption(f"🌐 {lang_choice}  |  {'🌙 Dark' if dark_mode else '☀️ Light'}")

PALETTE = ["#25D366","#128C7E","#075E54","#34B7F1","#FF6B6B",
           "#4ECDC4","#45B7D1","#96CEB4","#F7B731"]
EXTRA_SW = STOPWORDS | LANG_STOP.get(lang_choice, set()) | {
    "media","omitted","message","deleted","ok","okay","yeah","yes",
    "lol","haha","im","ive","dont","ill","its","ur","bhai","yaar","aur",
}

# ═════════════════════════════════════════════════════════════════════════════
# WELCOME
# ═════════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.info(T["welcome"])
    c1, c2, c3 = st.columns(3)
    for col, title, items in [
        (c1, "📊 Statistics",     ["Message counts","Response times","Activity heatmaps","Conversation starters"]),
        (c2, "🧠 NLP Analysis",   ["VADER sentiment","Top bigrams","Question rate","Late night msgs"]),
        (c3, "📈 Visualisations", ["Word clouds","Emoji charts","Monthly trends","Sentiment graphs"]),
    ]:
        items_html = "".join(f"<li>{i}</li>" for i in items)
        col.markdown(
            f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;"
            f"padding:1.2rem'><h4 style='color:{ACCENT}'>{title}</h4>"
            f"<ul style='color:{MUTED};font-size:.9rem'>{items_html}</ul></div>",
            unsafe_allow_html=True
        )
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# PARSE
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

path = "data/raw/uploaded_chat.txt"
with open(path, "wb") as f:
    f.write(uploaded_file.getbuffer())

with st.spinner("🔍 Parsing..."):
    try:
        df = parse_chat(path)
        if min_w > 1:
            df = df[df["word_count"] >= min_w]
    except Exception as e:
        st.error(f"❌ {e}"); st.stop()

st.success(T["parsed_ok"].format(
    msgs=f"{len(df):,}", parts=df["sender"].nunique(), days=df["date"].nunique()
))

# ═════════════════════════════════════════════════════════════════════════════
# STATS
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("📊 Computing statistics..."):
    stats_df   = basic_stats(df)
    hm_df      = activity_heatmap(df)
    emoji_df   = top_emojis(df, n=top_n)
    sent_df    = sentiment_analysis(df)
    trend_df   = monthly_trend(df)
    resp_df    = response_time(df)
    starters   = conversation_starters(df)
    streak     = longest_streak(df)
    vader_df   = vader_summary(df)
    bigram_df  = top_bigrams(df)
    q_df       = question_rate(df)
    lnm_df     = late_night_msgs(df)

starters_c = starters.copy()
starters_c.columns = ["sender", "conversations_started"]

# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════
ph = int(df["hour"].value_counts().idxmax())
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric(T["total_messages"], f"{len(df):,}")
c2.metric(T["participants"],   df["sender"].nunique())
c3.metric(T["days_active"],    df["date"].nunique())
c4.metric(T["avg_day"],        round(len(df)/df["date"].nunique(),1))
c5.metric(T["peak_hour"],      f"{ph:02d}:00")
c6.metric(T["longest_streak"], f"{streak['streak']} days")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    T["tab_stats"],T["tab_activity"],T["tab_words"],
    T["tab_emojis"],T["tab_sentiment"],T["tab_trends"]
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(T["per_stats"])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown(T["msg_count"])
        fig = px.bar(stats_df,x="sender",y="messages",color="sender",
                     template=PLOT_TMPL,color_discrete_sequence=PALETTE,
                     labels={"messages":"Messages","sender":"Participant"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown(T["msg_share"])
        fig2 = px.pie(stats_df,values="messages",names="sender",
                      color_discrete_sequence=PALETTE,hole=0.4)
        fig2.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown(T["resp_times"])
    col3,col4 = st.columns(2)
    with col3:
        st.dataframe(resp_df,use_container_width=True,hide_index=True)
    with col4:
        fig3 = px.bar(resp_df,x="sender",y="median_min",color="sender",
                      template=PLOT_TMPL,color_discrete_sequence=PALETTE,
                      labels={"median_min":"Median Minutes","sender":"Participant"})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3,use_container_width=True)
    st.markdown(T["conv_start"])
    st.dataframe(starters_c,use_container_width=True,hide_index=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(T["heatmap"])
    fh,ah = plt.subplots(figsize=(12,6))
    fh.patch.set_facecolor(WC_BG); ah.set_facecolor(WC_BG)
    sns.heatmap(hm_df,cmap="YlOrRd",linewidths=0.3,linecolor="white",
                ax=ah,annot=True,fmt="d",annot_kws={"size":8},
                cbar_kws={"label":"Messages"})
    ah.tick_params(colors=TEXT); ah.set_xlabel("Day",color=TEXT); ah.set_ylabel("Hour",color=TEXT)
    plt.tight_layout(); st.pyplot(fh); plt.close()

    st.markdown(T["hourly"])
    hourly = df.groupby("hour").size().reset_index(name="count")
    fhr = px.area(hourly,x="hour",y="count",template=PLOT_TMPL,
                  color_discrete_sequence=[ACCENT],labels={"count":"Messages","hour":"Hour"})
    fhr.update_layout(xaxis=dict(tickmode="linear",tick0=0,dtick=1))
    st.plotly_chart(fhr,use_container_width=True)

    st.markdown(T["by_day"])
    days_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ddf = df.groupby("day").size().reset_index(name="count")
    ddf["day"] = pd.Categorical(ddf["day"],categories=days_ord,ordered=True)
    ddf = ddf.sort_values("day")
    fd = px.bar(ddf,x="day",y="count",template=PLOT_TMPL,
                color="count",color_continuous_scale="Greens",
                labels={"count":"Messages","day":"Day"})
    st.plotly_chart(fd,use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if show_wc:
        st.markdown(T["wc_all"])
        tall = " ".join(df[~df["is_media"]]["message"].dropna().astype(str))
        if len(tall.strip())>50:
            wc = WordCloud(width=1400,height=600,background_color=WC_BG,
                           stopwords=EXTRA_SW,colormap=WC_CM,max_words=200).generate(tall)
            fw,aw = plt.subplots(figsize=(14,6))
            fw.patch.set_facecolor(WC_BG); aw.imshow(wc,interpolation="bilinear"); aw.axis("off")
            st.pyplot(fw); plt.close()

        st.markdown(T["wc_per"])
        wcols = st.columns(min(df["sender"].nunique(),2))
        for i,sender in enumerate(df["sender"].unique()):
            sub  = df[df["sender"]==sender]
            txt  = " ".join(sub[~sub["is_media"]]["message"].dropna().astype(str))
            if len(txt.strip())>30:
                wc2 = WordCloud(width=800,height=400,background_color=WC_BG,
                                stopwords=EXTRA_SW,colormap="plasma",max_words=100).generate(txt)
                f2,a2 = plt.subplots(figsize=(8,4))
                f2.patch.set_facecolor(WC_BG); a2.imshow(wc2,interpolation="bilinear")
                a2.axis("off"); a2.set_title(sender,fontsize=12,color=TEXT)
                with wcols[i%2]: st.pyplot(f2)
                plt.close()

        st.markdown(T["bigrams"])
        fbi = px.bar(bigram_df.head(15),x="count",y="bigram",orientation="h",
                     template=PLOT_TMPL,color="count",color_continuous_scale="Viridis",
                     labels={"count":"Count","bigram":"Word Pair"})
        fbi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fbi,use_container_width=True)
    else:
        st.info("Word clouds disabled. Enable in sidebar ⚙️")

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(T["top_emoji"])
    if not emoji_df.empty:
        e1,e2 = st.columns([2,1])
        with e1:
            fe = px.bar(emoji_df,x="emoji",y="count",template=PLOT_TMPL,
                        color="count",color_continuous_scale="Viridis",
                        labels={"count":"Count","emoji":"Emoji"})
            fe.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fe,use_container_width=True)
        with e2:
            st.dataframe(emoji_df,use_container_width=True,hide_index=True)
    else:
        st.warning("No emojis found.")
    st.markdown(T["q_rate"])
    st.dataframe(q_df,use_container_width=True,hide_index=True)
    st.markdown(T["late_night"])
    if not lnm_df.empty:
        fln = px.bar(lnm_df,x="sender",y="late_night_count",template=PLOT_TMPL,
                     color="sender",color_discrete_sequence=PALETTE,
                     labels={"late_night_count":"Late Night Messages","sender":"Participant"})
        fln.update_layout(showlegend=False)
        st.plotly_chart(fln,use_container_width=True)

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(T["sentiment"])
    st.dataframe(vader_df,use_container_width=True,hide_index=True)
    s1,s2 = st.columns(2)
    with s1:
        fs = px.bar(sent_df,x="sender",y="avg_polarity",color="avg_polarity",
                    color_continuous_scale=["#FF6B6B","#FFE66D","#4ECDC4"],
                    template=PLOT_TMPL,
                    labels={"avg_polarity":"Avg Polarity","sender":"Participant"},
                    title="Average Sentiment Polarity")
        st.plotly_chart(fs,use_container_width=True)
    with s2:
        if not vader_df.empty and "pct_positive" in vader_df.columns:
            fv = go.Figure()
            for cn,clr in [("pct_positive","#25D366"),("pct_neutral","#34B7F1"),("pct_negative","#FF6B6B")]:
                if cn in vader_df.columns:
                    fv.add_trace(go.Bar(name=cn.replace("pct_","").title(),
                                        x=vader_df["sender"],y=vader_df[cn],marker_color=clr))
            fv.update_layout(barmode="stack",template=PLOT_TMPL,
                              title="Sentiment Breakdown %",yaxis_title="Percentage")
            st.plotly_chart(fv,use_container_width=True)
    st.markdown(T["sent_time"])
    from textblob import TextBlob
    dcp = df[~df["is_media"]].copy()
    dcp["month"]    = pd.to_datetime(dcp["datetime"]).dt.to_period("M").astype(str)
    dcp["polarity"] = dcp["message"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    ms = dcp.groupby(["month","sender"])["polarity"].mean().reset_index()
    fms = px.line(ms,x="month",y="polarity",color="sender",template=PLOT_TMPL,
                  markers=True,color_discrete_sequence=PALETTE,
                  labels={"polarity":"Avg Sentiment","month":"Month"},
                  title="Monthly Sentiment Trend")
    fms.add_hline(y=0,line_dash="dash",line_color="gray",opacity=0.5)
    st.plotly_chart(fms,use_container_width=True)

# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown(T["monthly"])
    ft = px.line(trend_df,x="month",y="count",color="sender",template=PLOT_TMPL,
                 markers=True,color_discrete_sequence=PALETTE,
                 labels={"count":"Messages","month":"Month","sender":"Participant"})
    ft.update_layout(hovermode="x unified",xaxis_tickangle=-35)
    st.plotly_chart(ft,use_container_width=True)

    st.markdown(T["monthly_all"])
    mall = df.groupby("month").size().reset_index(name="count")
    fma = px.bar(mall,x="month",y="count",template=PLOT_TMPL,
                 color="count",color_continuous_scale="Greens",
                 labels={"count":"Messages","month":"Month"})
    fma.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fma,use_container_width=True)

    st.markdown(T["streak"])
    sc1,sc2,sc3 = st.columns(3)
    sc1.metric("Longest Streak", f"{streak['streak']} days")
    sc2.metric("Start", str(streak['start']))
    sc3.metric("End",   str(streak['end']))

# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(T["download"])
d1,d2 = st.columns(2)
with d1:
    st.download_button(T["dl_chat"],df.to_csv(index=False).encode("utf-8"),
                       "chat_clean.csv","text/csv",use_container_width=True)
with d2:
    st.download_button(T["dl_stats"],stats_df.to_csv(index=False).encode("utf-8"),
                       "chat_stats.csv","text/csv",use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:.85rem'>{T['footer']}</div>",
    unsafe_allow_html=True
)
