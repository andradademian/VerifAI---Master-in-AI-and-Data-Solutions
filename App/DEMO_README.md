# VerifAI — Demo UX/UI (branch: `demo-ux-redesign`)

This branch adds the **subscription, pricing, and onboarding experience** on top
of the existing article-scoring MVP — without changing how the MVP scores
articles. It's built to be demoed on a laptop with **no API keys and no ML
model installed**.

## What's new

A top **navigation with four tabs**:

| Tab | What it shows |
|---|---|
| **Explore** | The original working app — search live news and score any article. Unchanged behaviour. |
| **How it works** | Step-by-step of both channels (web app + newsletter) and a sample inbox digest. |
| **Pricing** | Free vs Premium plans (monthly/annual toggle) straight from the go-to-market deck. |
| **Subscribe** | Pick topics + keywords + frequency + email → instant newsletter preview. |

Plus: a **mock login / sign-up**, a **freemium usage gate** (5 free article
checks/month, then an upgrade prompt), a **mock Premium upgrade** that unlocks
the **five-dimension trust breakdown**, and a **sample newsletter** preview.

> **Demo mode:** all accounts, subscriptions and payments are simulated in the
> browser (localStorage). Nothing is stored on a server, and no emails are sent.

## Run it locally

```bash
cd App
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install Flask==3.0.0 flask-cors==4.0.0 requests==2.31.0 python-dotenv==1.0.0
PORT=5001 python app.py
```

Then open **http://localhost:5001**.

That's the full lightweight demo. To go beyond demo mode:

- **Live news in Explore:** copy `.env.example` to `.env` and set `NEWSDATA_API_KEY`.
- **AI explanation text:** also set `COHERE_API_KEY`.
- **Real ML score:** `pip install torch transformers` and make sure the model
  weights are present in `../distilbert_frozen_hf`.

## How the MVP stays safe

- The new UI lives in **new files** (`static/ui.css`, `static/ui.js`,
  `sample_data.py`) plus additive markup in `templates/index.html`.
- The only edits to the original scoring code (`static/script.js`, `app.py`)
  are **guarded add-ons** — heavy imports are wrapped in `try/except`, and the
  freemium/premium hooks no-op if the UI layer is removed. Delete `ui.js` and
  the original article-scoring app still runs exactly as before.
