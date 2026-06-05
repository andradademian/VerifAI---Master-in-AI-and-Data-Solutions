# Run with:
# python pipeline_full_api_default.py

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
import os, warnings, time
warnings.filterwarnings("ignore")

MODEL_DIR    = "DistilBERT"
MAX_ARTICLES = 200
BATCH_SIZE   = 16
MAX_LEN      = 256
OUTPUT_CSV   = ""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from App.news_fetcher import NewsDataFetcher
from App.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------
print(f"\nLoading model from '{MODEL_DIR}' ...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
print("Model loaded")

# ---------------------------------------------------------------------------
# 3. Load data from API
# ---------------------------------------------------------------------------
articles = None

print("\n[DATA] Using LIVE API")

fetcher = NewsDataFetcher(Config.NEWSDATA_API_KEY)

articles = fetcher.fetch_articles(
    country=Config.DEFAULT_COUNTRY,
    language=Config.DEFAULT_LANGUAGE,
    max_results=Config.DEFAULT_MAX_ARTICLES
)

X_test = [(a["title"] + " " + a["content"]).strip() for a in articles]
y_test = [None] * len(X_test)

print(f"Fetched {len(X_test)} articles")

# ---------------------------------------------------------------------------
# 4. Risk system
# ---------------------------------------------------------------------------
FAKE_THRESHOLD = 0.70
UNCERTAIN_LOW = 0.40

def compute_risk_score(fake_prob):
    return round(min(max(fake_prob, 0.0), 1.0), 3)

def classify_with_threshold(fake_prob):
    if fake_prob >= FAKE_THRESHOLD:
        return 1, "fake"
    if fake_prob >= UNCERTAIN_LOW:
        return -1, "uncertain"
    return 0, "real"

# ---------------------------------------------------------------------------
# 5. Dataset class
# ---------------------------------------------------------------------------
class ArticleDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

# ---------------------------------------------------------------------------
# 6. Inference
# ---------------------------------------------------------------------------
def classify(texts):
    loader = DataLoader(ArticleDataset(texts, tokenizer), batch_size=BATCH_SIZE)
    probs_all = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probs_all.append(probs)

    return np.vstack(probs_all)

print("\n[1/3] Running model...")
t0 = time.time()
probs = classify(X_test)
print(f"Done in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 7. Build results
# ---------------------------------------------------------------------------
print("\n[2/3] Building results...")

rows = []
for i in range(len(X_test)):
    fake_prob = float(probs[i][0])
    real_prob = float(probs[i][1])
    argmax_label = int(np.argmax(probs[i]))
    pred_label, pred_name = classify_with_threshold(fake_prob)

    true_label = None

    rows.append({
        "text_snippet": X_test[i][:120],
        "pred_label": pred_label,
        "pred_label_name": pred_name,
        "argmax_label": argmax_label,
        "true_label": true_label,
        "correct": (argmax_label == true_label) if true_label is not None else None,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "risk_score": compute_risk_score(fake_prob),

        # API metadata
        "title": articles[i]["title"] if articles else "",
        "source": articles[i]["source_name"] if articles else "",
        "url": articles[i]["url"] if articles else "",
        "published_at": articles[i]["published_at"] if articles else "",
    })

results_df = pd.DataFrame(rows)

# Drop duplicates by title/url (NewsData often returns the same article multiple times)
before = len(results_df)
if "url" in results_df.columns and results_df["url"].notna().any():
    results_df = results_df.drop_duplicates(subset=["url"], keep="first")
results_df = results_df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
dupes_removed = before - len(results_df)
if dupes_removed:
    print(f"Removed {dupes_removed} duplicate articles")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"Articles processed: {len(results_df)}")

print("Accuracy: N/A (live data)")

print(f"Predicted fake     (>= {FAKE_THRESHOLD:.2f}): {(results_df['pred_label']==1).sum()}")
print(f"Uncertain ({UNCERTAIN_LOW:.2f}-{FAKE_THRESHOLD:.2f}):       {(results_df['pred_label']==-1).sum()}")
print(f"Predicted real     (< {UNCERTAIN_LOW:.2f}): {(results_df['pred_label']==0).sum()}")

# ---------------------------------------------------------------------------
# 8b. Risk report
# ---------------------------------------------------------------------------
def generate_risk_report(results_df: pd.DataFrame) -> dict:
    scores = results_df["risk_score"]
    n = len(scores)

    tiers = [
        ("Low",      scores < 0.4),
        ("Medium",   (scores >= 0.4) & (scores < 0.6)),
        ("High",     (scores >= 0.6) & (scores < 0.8)),
        ("Critical", scores >= 0.8),
    ]

    sep = "=" * 60

    print(f"\n{sep}")
    print("RISK REPORT")
    print(sep)
    print(
        "NOTE: Ground-truth labels are unavailable in live environments.\n"
        "Evaluation is based on confidence scores, risk levels, and\n"
        "detection patterns rather than traditional accuracy metrics."
    )

    print(f"\n{'- Risk Distribution ':-<60}")
    print(f"  {'Tier':<10} {'Count':>6}  {'Pct':>6}")
    print(f"  {'-'*10} {'-'*6}  {'-'*6}")
    for label, mask in tiers:
        count = int(mask.sum())
        pct = count / n * 100
        print(f"  {label:<10} {count:>6}  {pct:>5.1f}%")

    print(f"\n{'- Confidence Score Distribution ':-<60}")
    print(f"  Mean  : {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Std   : {scores.std():.4f}")

    print(f"\n{'- Top 10 Flagged Articles (by risk score) ':-<60}")
    top = results_df.nlargest(10, "risk_score")[["title", "risk_score"]].reset_index(drop=True)
    for rank, row in top.iterrows():
        title = str(row["title"])[:80] if row["title"] else "(no title)"
        title = title.encode("ascii", errors="replace").decode("ascii")
        print(f"  {rank+1:>2}. [{row['risk_score']:.3f}]  {title}")

    print(sep)

    # Stats returned for downstream use (e.g. AI recommendations)
    return {
        "n": n,
        "tier_counts": {label: int(mask.sum()) for label, mask in tiers},
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "std": float(scores.std()),
        "top_flagged": [
            (str(r["title"])[:80] if r["title"] else "(no title)", float(r["risk_score"]))
            for _, r in top.iterrows()
        ],
    }

risk_stats = generate_risk_report(results_df)

# ---------------------------------------------------------------------------
# 8c. AI recommendations (Cohere chat API)
# ---------------------------------------------------------------------------
def generate_ai_recommendations(risk_stats: dict):
    """Generate AI recommendations from the risk report using Cohere's chat API.

    Returns (risk_assessment, reader_guidance). On any failure (no API key,
    package missing, network error) returns (None, None) so the pipeline keeps
    running without AI output.
    """
    api_key = Config.COHERE_API_KEY
    if not api_key:
        print("\n[AI] Skipped: COHERE_API_KEY not set")
        return None, None

    try:
        import cohere
    except ImportError:
        print("\n[AI] Skipped: 'cohere' package not installed (pip install cohere)")
        return None, None

    tiers = risk_stats["tier_counts"]
    top_lines = "\n".join(
        f"  - [{score:.3f}] {title}" for title, score in risk_stats["top_flagged"]
    )

    stats_block = (
        f"Articles analysed: {risk_stats['n']}\n"
        f"Risk tiers -> Low: {tiers['Low']}, Medium: {tiers['Medium']}, "
        f"High: {tiers['High']}, Critical: {tiers['Critical']}\n"
        f"Risk score -> mean: {risk_stats['mean']:.3f}, "
        f"median: {risk_stats['median']:.3f}, std: {risk_stats['std']:.3f}\n"
        f"Top flagged headlines:\n{top_lines}"
    )

    risk_prompt = (
        "You are a misinformation-risk analyst. Based on the fake-news detection "
        "results below, write a short risk assessment (3-4 sentences) summarising "
        "the overall reliability of this news batch.\n\n" + stats_block
    )
    reader_prompt = (
        "You are a media-literacy advisor. Based on the fake-news detection results "
        "below, give a reader 2-3 concrete, actionable tips for verifying the "
        "flagged stories before trusting or sharing them.\n\n" + stats_block
    )

    try:
        cohere_client = cohere.Client(api_key)

        # Generated recommendations using Cohere's chat API with command-a-03-2025
        risk_response = cohere_client.chat(
            model=Config.COHERE_MODEL,   # the specified model
            message=risk_prompt,
            max_tokens=200,
            temperature=0.7,             # Optional: adjust creativity
        )
        risk_assessment = risk_response.text.strip()

        reader_response = cohere_client.chat(
            model=Config.COHERE_MODEL,   # the specified model
            message=reader_prompt,
            max_tokens=200,
            temperature=0.7,             # Optional: adjust creativity
        )
        reader_guidance = reader_response.text.strip()
    except Exception as e:
        print(f"\n[AI] Skipped: Cohere request failed ({e})")
        return None, None

    sep = "=" * 60
    print(f"\n{sep}")
    print("AI RECOMMENDATIONS (Cohere)")
    print(sep)
    print("\n- Risk Assessment -")
    print(risk_assessment)
    print("\n- Reader Guidance -")
    print(reader_guidance)
    print(sep)

    return risk_assessment, reader_guidance

ai_risk_assessment, ai_reader_guidance = generate_ai_recommendations(risk_stats)

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
print("\n[3/3] Saving results...")

if OUTPUT_CSV:
    out_path = OUTPUT_CSV
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_{ts}.csv"

results_df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")
