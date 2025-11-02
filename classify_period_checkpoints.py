#!/usr/bin/env python3
# classify_period_checkpoints.py
from __future__ import annotations
import argparse, re, json, math, os, numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

# Support broader historical years (1600–2099)
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b", re.I)
DECADE_CORE_RE = re.compile(r"\b(18|19|20)(\d)0s\b", re.I)
DECADE_QUAL_RE = re.compile(r"\b(early|mid|late)\s+((?:18|19|20)\d0s)\b", re.I)

EVENT_MAP = {
    "vietnam war": (1955, 1975),
    "world war ii": (1939, 1945), "world war 2": (1939, 1945), "wwii": (1939, 1945),
    "world war i": (1914, 1918), "wwi": (1914, 1918),
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

def to_list(x) -> list:
    """Normalize a variety of column cell types into a list of strings.

    Handles: list/tuple/set, pandas Series/Index, numpy arrays, JSON-like strings,
    and scalar values. Avoids calling pd.isna on array-like objects directly.
    """
    # already list-like
    if isinstance(x, (list, tuple, set)):
        return [str(v) for v in x if pd.notna(v)]

    # pandas containers
    if isinstance(x, (pd.Series, pd.Index)):
        return [str(v) for v in x.tolist() if pd.notna(v)]

    # numpy arrays
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist() if pd.notna(v)]

    # strings: maybe JSON-ish or bracketed
    if isinstance(x, str):
        s = x.strip()
        # try JSON list (['Drama','History'] or ["Drama","History"]) or scalar
        try:
            v = json.loads(s)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v if pd.notna(t)]
            if v is None:
                return []
            return [str(v)]
        except Exception:
            pass
        # try python-ish list string: "['Drama','History']"
        if s.startswith("[") and s.endswith("]"):
            s2 = s[1:-1]
            out = [t.strip().strip("'\"") for t in s2.split(",")]
            return [t for t in out if t]
        # plain scalar string
        return [s] if s else []

    # scalars (None/NaN/etc.)
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    return [str(x)]

def find_years(text: str, release_year: Optional[int]) -> List[int]:
    t = text or ""
    yrs = [int(m.group(0)) for m in YEAR_RE.finditer(t)]
    if release_year is not None:
        yrs = [y for y in yrs if y <= release_year]
    return yrs

def decade_spans(text: str) -> List[Tuple[int,int]]:
    t = text or ""
    spans = []
    for m in DECADE_CORE_RE.finditer(t):
        start = int(m.group(1) + m.group(2) + "0")
        spans.append((start, start+9))
    for qual, base in DECADE_QUAL_RE.findall(t):
        y = int(base[:3]+"0")
        qual = qual.lower()
        if qual == "early": spans.append((y, y+3))
        elif qual == "mid": spans.append((y+4, y+6))
        else: spans.append((y+7, y+9))
    return spans

def event_spans(text: str) -> List[Tuple[int,int]]:
    t = (text or "").lower()
    out = []
    for k,(a,b) in EVENT_MAP.items():
        if k in t:
            out.append((a,b))
    return out

def event_hits(text: str) -> List[str]:
    t = (text or "").lower()
    return [k for k in EVENT_MAP.keys() if k in t]

def decade_label(y: Optional[int]) -> Optional[str]:
    return f"{(y//10)*10}s" if isinstance(y, int) else None

def classify_period(genres: List[str], keywords: List[str], overview: str, release_year: Optional[int]) -> Tuple[bool,List[str]]:
    gset = {str(g).lower() for g in genres}
    kset = {str(k).lower() for k in keywords}
    score, signals = 0, []

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

def extract_span(overview: str, keywords: List[str], release_year: Optional[int]) -> Optional[Tuple[int,int]]:
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

def infer_setting_type(release_year: Optional[int], span: Optional[Tuple[int,int]], overview: str, genres: List[str], keywords: List[str]) -> Tuple[str, float]:
    """Infer setting type: past|present|future|unknown with a rough confidence.

    High confidence if explicit span is available; medium for era/decade phrases; low otherwise.
    """
    ry = release_year
    # Future textual cues
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
        # default: if span exists but ambiguous relative to ry
        return ("past", 0.7) if e < ry else ("future", 0.7)

    # No span: use phrases
    if decade_spans(ov) or event_spans(ov):
        if ry is not None and has_future_phrase:
            return ("future", 0.6)
        return ("past", 0.6)
    if has_future_phrase:
        return ("future", 0.5)
    return ("unknown", 0.3)


def infer_setting_type_explicit(release_year: Optional[int], setting_start_year: Optional[int], setting_end_year: Optional[int], era_hits: List[str], text_blob: Optional[str]) -> Tuple[str, float]:
    """New explicit signature requested: infer setting_type from start/end years, era hits, and text.

    Returns (setting_type, confidence)
    """
    span = None
    try:
        if setting_start_year is not None and setting_end_year is not None:
            span = (int(setting_start_year), int(setting_end_year))
    except Exception:
        span = None
    ov = (text_blob or "").lower() if isinstance(text_blob, str) else ""
    # use earlier helper for core logic
    return infer_setting_type(release_year, span, ov, [], [])

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # best-effort normalization of expected columns
    if "release_year" not in df.columns:
        # try derive from release_date
        if "release_date" in df.columns:
            df["release_year"] = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce").astype("Int64")
        else:
            df["release_year"] = pd.NA

    # unify genres & keywords into lists of strings
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(to_list)
    elif "genre_ids" in df.columns:
        df["genres"] = df["genre_ids"].apply(to_list)
    else:
        df["genres"] = [[] for _ in range(len(df))]

    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(to_list)
    else:
        df["keywords"] = [[] for _ in range(len(df))]

    if "overview" not in df.columns:
        df["overview"] = ""

    if "tmdb_id" not in df.columns and "id" in df.columns:
        df["tmdb_id"] = df["id"]

    return df

def process(df: pd.DataFrame, future_handling: str = "separate") -> pd.DataFrame:
    """Process and enrich dataframe.

    future_handling: one of 'separate' (default), 'exclude', 'include'
    'separate' => mark setting_type='future' and leave is_period_piece=False
    'include' => count futures as period pieces when other signals indicate period
    'exclude' => ensure futures are not counted as period pieces
    """
    df = normalize_columns(df)

    # compute classification + spans
    def _row(r):
        genres = [str(g) for g in (r.get("genres") or [])]
        keywords = [str(k) for k in (r.get("keywords") or [])]
        overview = r.get("overview") or ""
        ry = int(r["release_year"]) if pd.notna(r["release_year"]) else None

        is_period, signals = classify_period(genres, keywords, overview, ry)
        span = extract_span(overview, keywords, ry)
        eras = event_hits(overview)
        # compute setting type using explicit signature helper
        s_start, s_end = (span if span else (None, None))
        stype, sconf = infer_setting_type_explicit(ry, s_start, s_end, eras, overview)
        s_start, s_end = (span if span else (None, None))
        dlabel = decade_label(s_start)
        gap = (ry - s_start) if (ry is not None and s_start is not None) else None
        # carry through any noted sources_used if present on the input row
        sources_used = r.get("sources_used") if isinstance(r.get("sources_used"), list) else []
        # enforce separate-but-include policy: if setting_type is future and handling is 'separate' or 'exclude', do not mark as period
        if stype == "future" and future_handling in ("separate", "exclude"):
            is_period = False

        return pd.Series({
            "is_period_piece": is_period,
            "signals_used": signals,
            "era_hits": eras,
            "setting_type": stype,
            "setting_confidence": sconf,
            "sources_used": sources_used,
            "setting_start_year": s_start,
            "setting_end_year": s_end,
            "setting_decade": dlabel,
            "year_gap": gap
        })

    enrich = df.apply(_row, axis=1)
    # remove any existing enrichment columns to avoid duplicates (from prior runs)
    enrich_cols = list(enrich.columns)
    df_no_enrich = df.drop(columns=enrich_cols, errors='ignore')
    out = pd.concat([df_no_enrich, enrich], axis=1)

    # order columns nicely if present
    cols = [c for c in [
        "tmdb_id", "title", "release_year", "overview", "genres", "keywords",
        "is_period_piece", "signals_used", "era_hits", "setting_type", "setting_confidence", "sources_used",
        "setting_start_year", "setting_end_year", "setting_decade", "year_gap",
        "vote_count", "revenue_usd", "in_top_k_by_revenue", "country"
    ] if c in out.columns]
    return out[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Parquet checkpoint files (e.g., data/period_movies_v1_checkpoint_1995.parquet data/period_movies_v1_checkpoint_1996.parquet)")
    ap.add_argument("--out-base", default="data/period_movies_checked", help="output base path (no extension)")
    ap.add_argument("--future-handling", choices=["separate","include","exclude"], default="separate", help="How to handle future-set films for period counts (default: separate)")
    args = ap.parse_args()

    dfs = [pd.read_parquet(p) for p in args.inputs]
    df = pd.concat(dfs, ignore_index=True)
    out = process(df, future_handling=args.future_handling)

    out_csv = f"{args.out_base}.csv"
    out_parq = f"{args.out_base}.parquet"
    out.to_csv(out_csv, index=False)
    out.to_parquet(out_parq, index=False)

    # tiny sanity print
    n = len(out)
    pc = int(out["is_period_piece"].fillna(False).sum())
    miss = int(out["setting_start_year"].isna().sum())
    print(f"[ok] rows={n}  period_piece_true={pc}  missing_setting_years={miss}")
    print(f"wrote:\n- {out_csv}\n- {out_parq}")

if __name__ == "__main__":
    main()
