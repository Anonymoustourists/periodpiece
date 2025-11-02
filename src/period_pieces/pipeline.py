from __future__ import annotations
import os, time, re, json, argparse, pathlib, typing as T
import requests
import pandas as pd
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# ---------- Config & setup ----------
load_dotenv()
BASE = "https://api.themoviedb.org/3"
TMDB_BEARER = os.getenv("TMDB_BEARER", "").strip()
HEADERS = {"Authorization": f"Bearer {TMDB_BEARER}"} if TMDB_BEARER else {}

CACHE_DIR = pathlib.Path(".cache/tmdb")
DATA_DIR = pathlib.Path("data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Support broader historical years (1600–2099)
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
DECADE_CORE_RE = re.compile(r"\b(18|19|20)(\d)0s\b", re.I)
DECADE_QUAL_RE = re.compile(r"\b(early|mid|late)\s+((?:18|19|20)\d0s)\b", re.I)

EVENT_MAP: dict[str, tuple[int,int]] = {
    "vietnam war": (1955, 1975),
    "world war ii": (1939, 1945),
    "world war 2": (1939, 1945),
    "wwii": (1939, 1945),
    "world war i": (1914, 1918),
    "wwi": (1914, 1918),
    "prohibition": (1920, 1933),
    "great depression": (1929, 1939),
    "civil rights movement": (1954, 1968),
    "regency": (1811, 1820),
    # Added eras/events
    "cold war": (1947, 1991),
    "space race": (1957, 1975),
    "civil war": (1861, 1865),
    "jazz age": (1920, 1929),
    "roaring twenties": (1920, 1929),
    "gilded age": (1870, 1900),
    "edwardian": (1901, 1910),
    "y2k": (1999, 2000),
}

PERIOD_GENRES = {"history","war","western","biography"}
PERIOD_KEYWORDS = {
    "period piece","biopic","based on true story","historical drama",
    "world war ii","prohibition","victorian","regency"
}

# ---------- Data models ----------
@dataclass
class MovieRow:
    tmdb_id: int
    title: str | None
    release_year: int | None
    genres: list[str]
    overview: str
    keywords: list[str]
    country: list[str] | None
    revenue_usd: int | None
    vote_count: int | None
    in_top_k_by_revenue: bool
    is_period_piece: bool
    signals_used: list[str]
    setting_start_year: int | None
    setting_end_year: int | None
    setting_decade: str | None
    year_gap: int | None
    # new fields
    era_hits: list[str] | None = None
    setting_type: str | None = None
    setting_confidence: float | None = None
    sources_used: list[str] | None = None

# ---------- HTTP + cache ----------
def _cache_path(kind: str, key: str) -> pathlib.Path:
    return CACHE_DIR / f"{kind}-{key}.json"

def _read_cache(kind: str, key: str) -> dict | None:
    p = _cache_path(kind, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _write_cache(kind: str, key: str, obj: dict) -> None:
    p = _cache_path(kind, key)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def tmdb_get(path: str, params: dict | None = None, cache_key: str | None = None,
             retries: int = 4, backoff: float = 1.6) -> dict:
    if cache_key:
        c = _read_cache("GET", cache_key)
        if c is not None:
            return c
    if not TMDB_BEARER:
        raise RuntimeError("TMDB_BEARER missing. Put your v4 Bearer token in .env")

    for attempt in range(retries):
        r = requests.get(BASE + path, headers=HEADERS, params=params or {}, timeout=30)
        if r.status_code in (429,) or r.status_code >= 500:
            time.sleep(backoff ** (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if cache_key:
            _write_cache("GET", cache_key, data)
        return data
    raise RuntimeError(f"TMDB request failed after retries: {path} {params}")

# ---------- TMDB helpers ----------
def discover_top_by_revenue(year: int, topk: int, min_votes: int = 200) -> list[dict]:
    """Discover movies for a year sorted by revenue with sensible behavior for sparse eras.

    For years < 1980, use release_date.gte/lte instead of primary_release_year and automatically
    reduce the minimum vote threshold to surface older titles.
    """
    # Auto-tune min_votes for older years
    if year < 1960:
        min_votes_eff = min(min_votes, 25)
    elif year < 1980:
        min_votes_eff = min(min_votes, 50)
    else:
        min_votes_eff = min_votes

    use_date_range = year < 1980

    acc: list[dict] = []
    page = 1
    while len(acc) < topk:
        if use_date_range:
            params = {
                "sort_by": "revenue.desc",
                "vote_count.gte": min_votes_eff,
                "include_adult": "false",
                "release_date.gte": f"{year}-01-01",
                "release_date.lte": f"{year}-12-31",
                "page": page,
            }
            mode = "date"
        else:
            params = {
                "primary_release_year": year,
                "sort_by": "revenue.desc",
                "vote_count.gte": min_votes_eff,
                "include_adult": "false",
                "page": page,
            }
            mode = "year"

        resp = tmdb_get(
            "/discover/movie",
            params,
            cache_key=f"discover_{year}_{page}_mv{min_votes_eff}_{mode}",
        )
        acc.extend(resp.get("results", []))
        if page >= resp.get("total_pages", 1):
            break
        page += 1
    return acc[:topk]

def get_movie_details(movie_id: int) -> dict:
    return tmdb_get(
        f"/movie/{movie_id}",
        params={
            "append_to_response": "keywords,external_ids,translations,alternative_titles,release_dates",
        },
        cache_key=f"movie_{movie_id}_details",
    )

# ---------- Text parsing ----------
def find_years(text: str, release_year: int | None) -> list[int]:
    ys = [int(m.group(0)) for m in YEAR_RE.finditer(text or "")]
    if release_year:
        ys = [y for y in ys if y <= release_year]
    return ys

def decade_spans(text: str) -> list[tuple[int,int]]:
    t = (text or "")
    spans: list[tuple[int,int]] = []
    for m in DECADE_CORE_RE.finditer(t):
        start = int(m.group(1) + m.group(2) + "0")
        spans.append((start, start+9))
    for qual, base in DECADE_QUAL_RE.findall(t):
        y = int(base[:3]+"0")
        if qual.lower() == "early":
            spans.append((y, y+3))
        elif qual.lower() == "mid":
            spans.append((y+4, y+6))
        else:
            spans.append((y+7, y+9))
    return spans

def event_spans(text: str) -> list[tuple[int,int]]:
    t = (text or "").lower()
    out = []
    for k,(a,b) in EVENT_MAP.items():
        if k in t:
            out.append((a,b))
    return out

def event_hits(text: str) -> list[str]:
    t = (text or "").lower()
    return [k for k in EVENT_MAP.keys() if k in t]

def decade_label(y: int | None) -> str | None:
    return f"{(y//10)*10}s" if isinstance(y,int) else None

def infer_setting_type(release_year: int | None, span: tuple[int,int] | None, overview: str, genres: list[str], keywords: list[str]) -> tuple[str, float]:
    ry = release_year
    ov = (overview or "").lower()
    future_cues = ["set in the future", "post-apocalyptic", "dystopian", "far future"]
    has_future_phrase = any(c in ov for c in future_cues)

    if span and ry is not None:
        s,e = span
        if e <= ry - 5:
            return ("past", 0.9)
        if s >= ry + 5:
            return ("future", 0.9)
        mid = (s + e) // 2
        if abs(mid - ry) <= 5:
            return ("present", 0.8)
        return ("past", 0.7) if e < ry else ("future", 0.7)

    if decade_spans(ov) or event_spans(ov):
        if ry is not None and has_future_phrase:
            return ("future", 0.6)
        return ("past", 0.6)
    if has_future_phrase:
        return ("future", 0.5)
    return ("unknown", 0.3)

# ---------- Classification ----------
def classify_period(genres: list[str], keywords: list[str], overview: str, release_year: int | None) -> tuple[bool,list[str]]:
    gset = {g.lower() for g in genres}
    kset = {k.lower() for k in keywords}
    score = 0
    signals: list[str] = []

    if PERIOD_GENRES & gset:
        score += 3; signals.append("genre_period")

    if PERIOD_KEYWORDS & kset:
        score += 2; signals.append("keyword_period")

    yrs = find_years(overview, release_year)
    if release_year is not None and any(y <= release_year - 10 for y in yrs):
        score += 2; signals.append("explicit_past_year")

    if decade_spans(overview):
        score += 1; signals.append("decade_phrase")

    if {"fantasy","science fiction"} & gset and not yrs and not decade_spans(overview):
        score -= 2; signals.append("fantasy_penalty")

    return (score >= 3), signals

def extract_span(overview: str, keywords: list[str], release_year: int | None) -> tuple[int,int] | None:
    yrs = set(find_years(overview, release_year))
    if yrs:
        return (min(yrs), max(yrs))
    decs = decade_spans(overview)
    if decs:
        return (min(a for a,_ in decs), max(b for _,b in decs))
    evs = event_spans(overview)
    if evs:
        return (min(a for a,_ in evs), max(b for _,b in evs))
    return None

# ---------- Row builder ----------
def build_row(movie_id: int) -> MovieRow:
    d = get_movie_details(movie_id)
    title = d.get("title")
    rd = d.get("release_date") or "0000-01-01"
    release_year = int(rd.split("-")[0]) if rd[:4].isdigit() else None
    genres = [g["name"] for g in d.get("genres", [])]
    overview = d.get("overview") or ""
    revenue = d.get("revenue") or 0
    votes = d.get("vote_count") or 0
    country = d.get("origin_country") or [c.get("iso_3166_1") for c in d.get("production_countries", [])]
    kw = d.get("keywords", {})
    keywords = [k["name"] for k in (kw.get("keywords") or kw.get("results") or [])]

    is_period, signals = classify_period(genres, keywords, overview, release_year)
    span = extract_span(overview, keywords, release_year)
    s_start, s_end = (span if span else (None, None))
    dec = decade_label(s_start)
    gap = (release_year - s_start) if (release_year and s_start) else None
    eras = event_hits(overview)
    stype, sconf = infer_setting_type(release_year, span, overview, genres, keywords)
    sources = ["overview"] + (["keywords"] if keywords else [])

    return MovieRow(
        tmdb_id=d["id"], title=title, release_year=release_year, genres=genres,
        overview=overview, keywords=keywords, country=country,
        revenue_usd=revenue, vote_count=votes, in_top_k_by_revenue=True,
        is_period_piece=is_period, signals_used=signals,
        setting_start_year=s_start, setting_end_year=s_end,
        setting_decade=dec, year_gap=gap,
        era_hits=eras, setting_type=stype, setting_confidence=sconf, sources_used=sources,
    )

# ---------- CLI ----------
def run(start_year: int, end_year: int, topk: int, min_votes: int, sleep_s: float, out_basename: str, seventies_vote_min: int | None = None):
    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        # apply seventies override when applicable
        eff_min_votes = min_votes
        if seventies_vote_min is not None and 1970 <= year <= 1979:
            eff_min_votes = seventies_vote_min
        stubs = discover_top_by_revenue(year, topk, min_votes=eff_min_votes)
        ids = list({s["id"] for s in stubs})
        for mid in ids:
            row = build_row(mid)
            rows.append(asdict(row))
            if sleep_s > 0:
                time.sleep(sleep_s)
        # checkpoint per year
        df_ck = pd.DataFrame(rows)
        df_ck.to_parquet(DATA_DIR / f"{out_basename}_checkpoint_{year}.parquet", index=False)

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / f"{out_basename}.csv"
    pq_path  = DATA_DIR / f"{out_basename}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    # quick aggregates (saved for sanity-check)
    agg = pd.DataFrame({
        "count_by_setting_decade": df["setting_decade"].value_counts(dropna=False),
    })
    agg.to_csv(DATA_DIR / f"{out_basename}_quick_agg.csv")

    print(f"Wrote:\n- {csv_path}\n- {pq_path}\n- {DATA_DIR / f'{out_basename}_quick_agg.csv'}")

def main():
    ap = argparse.ArgumentParser(description="Pull TMDB top-K by year and label period pieces.")
    ap.add_argument("--start", type=int, default=1980, help="Start year (inclusive)")
    ap.add_argument("--end", type=int, default=2024, help="End year (inclusive)")
    ap.add_argument("--topk", type=int, default=200, help="Top K per year by revenue")
    ap.add_argument("--min-votes", type=int, default=200, help="Minimum vote_count to include in discover query")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between movie detail calls")
    ap.add_argument("--out", type=str, default="period_movies_v1", help="Output base filename (no extension)")
    ap.add_argument("--seventies-vote-min", type=int, default=None, help="Override min-votes for years 1970-1979")
    args = ap.parse_args()
    run(args.start, args.end, args.topk, args.min_votes, args.sleep, args.out, seventies_vote_min=args.seventies_vote_min)

if __name__ == "__main__":
    main()
