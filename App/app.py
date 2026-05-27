from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

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

# -------------------------------------------------------------------
# LOAD AI MODEL
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "distilbert_frozen_hf")
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AI model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

print("AI model loaded successfully")

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
        # PREPARE ARTICLE
        # ---------------------------------------------------------
        content = f"{title} {text}"

        # ---------------------------------------------------------
        # TOKENIZE
        # ---------------------------------------------------------
        inputs = tokenizer(
            content,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # ---------------------------------------------------------
        # MODEL INFERENCE
        # ---------------------------------------------------------
        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # IMPORTANT:
        # class 0 = fake
        # class 1 = real

        prob_fake = float(probs[0].cpu().item())
        prob_real = float(probs[1].cpu().item())

        # ---------------------------------------------------------
        # CLASSIFICATION LOGIC
        # ---------------------------------------------------------
        if prob_fake >= 0.70:
            classification = "Most likely fake"

        elif prob_fake >= 0.40:
            classification = "Uncertain"

        else:
            classification = "Most likely real"

        # ---------------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------------
        result = {
            'status': 'success',
            'article_id': article_id,
            'analysis': {
                'classification': classification,
                'prob_real': round(prob_real, 4),
                'prob_fake': round(prob_fake, 4),
                'threshold_used': 0.70,
                'crisis_detected': False,
                'crisis_categories': [],
                'crisis_keywords': {},
                'crisis_intensity': round(prob_fake, 4)
            }
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
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