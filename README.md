# period-pieces (v1)
Fetches TMDB "top K by revenue" per year, heuristically detects period pieces, extracts setting years/decades, and exports CSV/Parquet.

## TMDB access (how to obtain)
1. Create a free account at https://www.themoviedb.org/ (TMDB).
2. Go to Settings → API → "Create" (apply for an API key if prompted).
3. Under API v4, generate a **"Read Access Token (v4 auth)"**.
4. Copy the token into a local `.env` file as:
```

TMDB_BEARER=YOUR_LONG_BEARER_TOKEN

````

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env   # paste your token

# Run: 1980–2024 inclusive, top 200 per year, write CSV+Parquet under ./data
python -m period_pieces.pipeline --start 1980 --end 2024 --topk 200

# Options:
python -m period_pieces.pipeline -h
````