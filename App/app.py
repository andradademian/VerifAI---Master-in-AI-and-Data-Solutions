from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import cohere

# -------------------------------------------------------------------
# AI MODEL IMPORTS
# -------------------------------------------------------------------
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

load_dotenv()

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
NEWSDATA_API_KEY = os.getenv('NEWSDATA_API_KEY', '')
NEWSDATA_API_URL = os.getenv(
    'NEWS_API_BASE_URL',
    'https://newsdata.io/api/1/latest'
)

# Cohere (AI-generated recommendations)
COHERE_API_KEY = os.getenv('COHERE_API_KEY', '')
COHERE_MODEL = os.getenv('COHERE_MODEL', 'command-a-03-2025')

cohere_client = None
if COHERE_API_KEY:
    try:
        import cohere
        cohere_client = cohere.Client(COHERE_API_KEY)
        print("Cohere client ready")
    except ImportError:
        print("Cohere not installed (pip install cohere) - AI recommendations disabled")
else:
    print("COHERE_API_KEY not set - AI recommendations disabled")

# -------------------------------------------------------------------
# LOAD AI MODEL
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "distilbert_frozen_hf")
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AI model...")

# Model weights (*.safetensors / *.bin) are gitignored and may be missing on a
# fresh checkout. Load defensively so the app still runs in DEMO mode, where
# /api/analyze returns a placeholder confidence score and Cohere still produces
# the friendly explanation.
tokenizer = None
model = None
MODEL_AVAILABLE = False
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(device)
    model.eval()
    MODEL_AVAILABLE = True
    print("AI model loaded successfully")
except Exception as e:
    print(f"WARNING: could not load model weights ({e})")
    print("Running in DEMO mode - /api/analyze will use a placeholder score")

# -------------------------------------------------------------------
# STORE ARTICLES
# -------------------------------------------------------------------
articles_cache = []


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/articles', methods=['GET'])
def get_articles():

    try:
        country = request.args.get('country', 'us')
        language = request.args.get('language', 'en')
        category = request.args.get('category', '')
        query = request.args.get('q', '')
        page = request.args.get('page', '')

        params = {
            'apikey': NEWSDATA_API_KEY,
            'country': country,
            'language': language,
        }

        if category:
            params['category'] = category

        if query:
            params['q'] = query

        if page:
            params['page'] = page

        print(f"Fetching articles with params: {params}")

        response = requests.get(
            NEWSDATA_API_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if data['status'] == 'success':

            articles = []

            for article in data.get('results', []):

                content = article.get('content', '')

                if content == "ONLY AVAILABLE IN PAID PLANS":
                    content = article.get('description', '')

                if not content or content == "ONLY AVAILABLE IN PAID PLANS":
                    continue

                processed_article = {
                    'id': article.get('article_id', ''),
                    'title': article.get('title', 'No title'),
                    'description': article.get(
                        'description',
                        'No description available'
                    ),
                    'content': content,
                    'source': article.get('source_id', 'Unknown'),
                    'source_name': article.get(
                        'source_name',
                        'Unknown Source'
                    ),
                    'url': article.get('link', ''),
                    'image_url': article.get('image_url', ''),
                    'published_at': article.get('pubDate', ''),
                    'category': article.get('category', []),
                    'country': article.get('country', []),
                }

                articles.append(processed_article)

            global articles_cache

            if not page:
                articles_cache = articles
            else:
                articles_cache.extend(articles)

            return jsonify({
                'status': 'success',
                'total_results': len(articles),
                'articles': articles,
                'nextPage': data.get('nextPage', None)
            })

        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to fetch articles'
            }), 500

    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'message': f'API request failed: {str(e)}'
        }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/api/article/<article_id>', methods=['GET'])
def get_article(article_id):

    article = next(
        (a for a in articles_cache if a['id'] == article_id),
        None
    )

    if article:
        return jsonify({
            'status': 'success',
            'article': article
        })

    return jsonify({
        'status': 'error',
        'message': 'Article not found'
    }), 404
# -------------------------------------------------------------------
# COHERE AI RECOMMENDATION
# -------------------------------------------------------------------
def generate_ai_recommendation(title, text, classification, uncertainty):
    if cohere_client is None:
        return None

    # Guide Cohere to focus purely on the "why" and "how" without data repetition
    ai_prompt = f"""
You are an analyst for VerifAI, an platform evaluating news credibility.

Your task is to write a short, friendly, and accessible summary explaining why our machine learning model reached its conclusion for the article below.

ARTICLE DETAILS
Title: {title}
Text: {text[:2000]}

MODEL TAKEAWAY
Overall Status: {classification}
Uncertainty Level: {"High" if uncertainty > 0.6 else "Normal"}

INSTRUCTIONS
Write a brief, easy-to-understand explanation (2-3 sentences max).

Guidelines:
- Use a helpful, conversational, yet professional tone (like a friendly editor).
- STRICT RULE: Do not include ANY numbers, percentages, or mathematical metrics. The user already sees them on their dashboard.
- Instead of metrics, describe the *linguistic style* of the text. Does it sound emotional, sensationalist, or speculative? Or does it use objective, well-structured language?
- If the model's uncertainty is high, gently advise the reader to cross-reference with other sources.
- Do not make definitive claims about whether the text is factually "true" or "false"—focus entirely on text style, tone, and framing.
- Output ONLY the final summary text. No introductory filler.
"""

    try:
        ai_response = cohere_client.chat(
            model=COHERE_MODEL,
            message=ai_prompt,
            max_tokens=150,
            temperature=0.4
        )
        return ai_response.text.strip()

    except Exception as e:
        print(f"Cohere request failed: {e}")
        return None

# -------------------------------------------------------------------
# AI ANALYSIS ENDPOINT
# -------------------------------------------------------------------
@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    try:
        data = request.get_json()

        title = data.get('title', '')
        text = data.get('text', '')
        article_id = data.get('article_id', '')

        if not title or not text:
            return jsonify({
                'status': 'error',
                'message': 'Title and text are required'
            }), 400

        # ---------------------------------------------------------
        # OPTIONAL TEXT NORMALISATION (reduces domain bias)
        # ---------------------------------------------------------
        def clean_text(t):
            import re
            return re.sub(
                r"\b(NBA|NFL|FIFA|election|senate|government|president)\b",
                "",
                t,
                flags=re.I
            )

        # Better structure than raw concatenation
        content = f"Title: {title}. Article: {text[:2000]}"
        content = clean_text(content)

        if MODEL_AVAILABLE:
            # -----------------------------------------------------
            # TOKENIZE
            # -----------------------------------------------------
            inputs = tokenizer(
                content,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # -----------------------------------------------------
            # MODEL INFERENCE (TEMPERATURE SCALING)
            # -----------------------------------------------------
            import torch.nn.functional as F

            TEMPERATURE = 2.0  # key fix for overconfidence

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits / TEMPERATURE
                probs = F.softmax(logits, dim=-1)[0]

            prob_fake = float(probs[0].item())
            prob_real = float(probs[1].item())
        else:
            # -----------------------------------------------------
            # DEMO MODE (model weights missing) - deterministic
            # placeholder score derived from the article text so
            # the UI + Cohere flow can still be exercised.
            # -----------------------------------------------------
            import hashlib
            h = int(hashlib.sha256(content.encode("utf-8")).hexdigest(), 16)
            prob_fake = 0.15 + (h % 1000) / 1000 * 0.70  # spread across 0.15-0.85
            prob_real = 1.0 - prob_fake

        # ---------------------------------------------------------
        # STABLE CONFIDENCE METRICS
        # ---------------------------------------------------------
        margin = abs(prob_fake - prob_real)

        confidence = margin
        uncertainty = 1 - margin

        risk_score = prob_fake

        # ---------------------------------------------------------
        # SOFT CLASSIFICATION (NO HARD THRESHOLDS)
        # ---------------------------------------------------------
        if uncertainty > 0.6:
            label = "uncertain"
            classification = "Low confidence prediction"
        elif prob_fake > prob_real:
            label = "fake"
            classification = "Likely misinformation"
        else:
            label = "real"
            classification = "Likely credible"

        # ---------------------------------------------------------
        # AI RECOMMENDATION (Cohere)
        # ---------------------------------------------------------
        ai_recommendation = generate_ai_recommendation(
            title=title,
            text=text,
            classification=classification,
            uncertainty=uncertainty
        )

        # ---------------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------------
        result = {
            "status": "success",
            "article_id": article_id,

            "analysis": {
                "label": label,
                "classification": classification,

                "prob_fake": round(prob_fake, 4),
                "prob_real": round(prob_real, 4),

                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty, 4),

                "risk_score": round(risk_score, 4),

                "decision_rule": "temperature-scaled softmax + margin-based uncertainty",

                "ai_recommendation": ai_recommendation,
                "demo_mode": not MODEL_AVAILABLE
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Analysis failed: {str(e)}"
        }), 500
# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------
if __name__ == '__main__':

    if not NEWSDATA_API_KEY:

        print("ERROR: NEWSDATA_API_KEY not found")
        print("Please set it in your .env file")

    else:

        print("Starting Flask server...")
        print("Frontend: http://localhost:5000")
        print("Articles API: http://localhost:5000/api/articles")

        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000
        )