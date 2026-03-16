# 📱 WhatsApp Chat Analysis — End-to-End Python Project

A complete data analysis pipeline for WhatsApp exported chat files.
Produces charts, word clouds, sentiment analysis, and a full HTML dashboard report.

---

## 🗂 Project Structure

```
whatsapp_analysis/
├── data/
│   ├── raw/          ← Place your exported chat.txt here
│   └── processed/    ← Cleaned CSV output
├── outputs/
│   ├── charts/       ← PNG + interactive HTML charts
│   └── reports/      ← Final HTML report
├── parse.py          ← Chat parser (Android + iOS)
├── analyse.py        ← Statistics engine
├── visualise.py      ← Chart generator
├── nlp_extras.py     ← VADER sentiment, bigrams, NLP
├── dashboard.py      ← HTML report builder
├── main.py           ← Full pipeline entry point
├── requirements.txt  ← Dependencies
└── README.md         ← This file
```

---

## 🚀 Quickstart

### Step 1 — Get your WhatsApp export

**Android:**
Open any chat → tap ⋮ (3-dot menu) → More → Export Chat → Without Media → Save the `.txt` file

**iOS / iPhone:**
Open any chat → tap the contact/group name at the top → Export Chat → Without Media → Save to Files

Place the `.txt` file in `data/raw/chat.txt`

---

### Step 2 — Set up Python environment

```bash
# Clone / download this project, then:
cd whatsapp_analysis

python -m venv venv

# Activate:
# macOS / Linux:
source venv/bin/activate
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

### Step 3 — Run the pipeline

```bash
python main.py --file data/raw/chat.txt
```

Optional flags:
```bash
python main.py --file data/raw/chat.txt --no-wordcloud   # faster, skips word clouds
```

---

### Step 4 — View results

Open `outputs/reports/report.html` in any browser.

---

## 🖥 Which Platform to Use?

| Platform | Recommendation | Notes |
|---|---|---|
| **Local PC / Mac** ⭐ Best | ✅ Recommended | Full speed, all charts, open HTML directly |
| **VS Code** | ✅ Excellent | Best editor; use integrated terminal |
| **PyCharm** | ✅ Excellent | Good for debugging |
| **Google Colab** | ✅ Good | Free GPU, no local install needed |
| **Jupyter Notebook** | ✅ Good | Interactive cell-by-cell |
| **Replit** | ⚠️ Limited | Slow for matplotlib; storage limits |
| **GitHub Codespaces** | ✅ Good | Cloud VS Code, works well |

### Recommended: Local Python (VS Code)
- Python 3.10, 3.11, or 3.12
- Install VS Code + Python extension
- Run in integrated terminal

### Alternative: Google Colab (no install needed)
```python
# In a Colab cell:
!pip install -r requirements.txt -q
from google.colab import files
uploaded = files.upload()   # upload your chat.txt
import shutil
shutil.move(list(uploaded.keys())[0], 'data/raw/chat.txt')
!python main.py --file data/raw/chat.txt --no-wordcloud
```

---

## 📊 What Gets Generated

| Output | Description |
|---|---|
| `heatmap.png` | Hour × Day activity heatmap |
| `wordcloud_all.png` | Word cloud for all messages |
| `wordcloud_<name>.png` | Word cloud per participant |
| `messages_per_sender.png` | Bar chart of message counts |
| `message_share.png` | Pie chart of participation |
| `hourly_distribution.png` | Messages by hour of day |
| `monthly_trend.html` | Interactive monthly line chart |
| `emoji_bar.html` | Interactive top emojis chart |
| `sentiment.png` | Sentiment scores per participant |
| `response_time.png` | Median reply times |
| `report.html` | Full self-contained HTML dashboard |
| `chat_clean.csv` | Cleaned data as CSV |

---

## 🛠 Troubleshooting

**"No messages parsed" error:**
- Check the .txt file encoding — try opening in Notepad and saving as UTF-8
- Some exports use a different date format — check the top few lines

**Emoji library error:**
```bash
pip install emoji --upgrade
```

**NLTK data missing:**
```python
import nltk
nltk.download('all')
```

**Matplotlib "no display" error on Linux server:**
Already handled — `matplotlib.use('Agg')` is set in visualise.py

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| pandas | Data manipulation |
| matplotlib + seaborn | Static charts |
| plotly | Interactive charts |
| wordcloud | Word cloud images |
| emoji | Emoji detection |
| textblob | Basic sentiment |
| nltk + VADER | Advanced sentiment |
| jinja2 | HTML templating |
| tqdm | Progress bars |
| pillow | Image processing |

---

## 🔒 Privacy Note

Your chat data never leaves your machine. All processing is 100% local.

---

## 📄 License

MIT — free to use, modify, and share.
