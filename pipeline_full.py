#run with py pipeline_full.py --use_api

# ---------------------------------------------------------------------------
# 0. Argument parsing
# ---------------------------------------------------------------------------
import argparse, os, warnings, time
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Fake-news detection pipeline")
parser.add_argument("--model_dir", default="distilbert_finetuned")
parser.add_argument("--max_articles", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--output_csv", default="")
parser.add_argument("--use_api", action="store_true",
                    help="Use live news API instead of dataset")
args = parser.parse_args()

MODEL_DIR    = args.model_dir
MAX_ARTICLES = args.max_articles
BATCH_SIZE   = args.batch_size
MAX_LEN      = 256

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
print("✅ Model loaded")

# ---------------------------------------------------------------------------
# 3. Load data (API OR dataset)
# ---------------------------------------------------------------------------
articles = None

if args.use_api:
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

else:
    print("\n[DATA] Using LOCAL dataset")

    X_test = np.load(os.path.join(MODEL_DIR, "test_texts.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(MODEL_DIR, "test_labels.npy"), allow_pickle=True)

    if MAX_ARTICLES and MAX_ARTICLES < len(X_test):
        X_test = X_test[:MAX_ARTICLES]
        y_test = y_test[:MAX_ARTICLES]

    print(f"Loaded {len(X_test)} test articles")

# ---------------------------------------------------------------------------
# 4. Risk system
# ---------------------------------------------------------------------------
def compute_risk_score(fake_prob):
    return round(min(max(fake_prob * 0.5, fake_prob * 0.3), 1.0), 3)

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
    pred_label = int(np.argmax(probs[i]))

    true_label = None if args.use_api else int(y_test[i])

    rows.append({
        "text_snippet": X_test[i][:120],
        "pred_label": pred_label,
        "pred_label_name": "fake" if pred_label == 1 else "real",
        "true_label": true_label,
        "correct": (pred_label == true_label) if true_label is not None else None,
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

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"Articles processed: {len(results_df)}")

if not args.use_api:
    accuracy = results_df["correct"].mean()
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("Accuracy: N/A (live data)")

print(f"Predicted fake: {(results_df['pred_label']==1).sum()}")
print(f"Predicted real: {(results_df['pred_label']==0).sum()}")

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
print("\n[3/3] Saving results...")

if args.output_csv:
    out_path = args.output_csv
else:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_{ts}.csv"

results_df.to_csv(out_path, index=False)

print(f"Saved to {out_path}")