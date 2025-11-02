from __future__ import annotations
import argparse, glob, os, pathlib, typing as T, time, json
import pandas as pd

# Reuse the enrichment logic from the existing classifier script
# Import the classifier module (we call process via the module name to preserve explicit call site)
try:
    import classify_period_checkpoints as classify_period_checkpoints
except Exception:
    import sys
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    import classify_period_checkpoints as classify_period_checkpoints

WRITES_DIR = pathlib.Path("data") / "reports"
WRITES_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoints(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No checkpoint files matched pattern: {pattern}")
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    # drop duplicates if reruns occurred
    if "tmdb_id" in df.columns:
        df = df.drop_duplicates(subset=["tmdb_id"])  # keep first
    return df, paths, dfs


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # classify_process orders columns and computes period features
    return classify_period_checkpoints.process(df)


def _import_tmdb_getter():
    """Import get_movie_details from the pipeline, accommodating running by path."""
    try:
        from period_pieces.pipeline import get_movie_details as get_details
        return get_details
    except Exception:
        import sys
        ROOT = pathlib.Path(__file__).resolve().parents[2]
        SRC = ROOT / "src"
        # ensure project src on path
        if str(SRC) not in sys.path:
            sys.path.append(str(SRC))
        from period_pieces.pipeline import get_movie_details as get_details
        return get_details


def details_pass(df: pd.DataFrame, limit: int | None = 300) -> pd.DataFrame:
    """Fetch extra TMDB fields for likely period pieces with unknown spans and enrich overview/keywords.

    Only touches rows where is_period_piece == True and setting_start_year is NA.
    """
    if "tmdb_id" not in df.columns:
        return df
    mask = df.get("is_period_piece").fillna(False) & df.get("setting_start_year").isna()
    idxs = df.index[mask].tolist()
    if not idxs:
        return df

    get_details = _import_tmdb_getter()

    # cap work for safety in one-off runs
    if limit is not None:
        idxs = idxs[:limit]

    for i in idxs:
        mid = int(df.at[i, "tmdb_id"]) if pd.notna(df.at[i, "tmdb_id"]) else None
        if not mid:
            continue
        try:
            d = get_details(mid)
        except Exception:
            continue
        # collect extra text
        tagline = (d.get("tagline") or "").strip()
        coll = d.get("belongs_to_collection") or {}
        coll_name = (coll.get("name") or "").strip() if isinstance(coll, dict) else ""
        # alternative titles
        alt_titles = []
        try:
            for t in (d.get("alternative_titles", {}) or {}).get("titles", []):
                name = (t.get("title") or "").strip()
                if name:
                    alt_titles.append(name)
        except Exception:
            pass
        # translations: titles + overviews
        trans_texts = []
        try:
            for tr in (d.get("translations", {}) or {}).get("translations", []):
                data = tr.get("data") or {}
                tt = (data.get("title") or "").strip()
                ov = (data.get("overview") or "").strip()
                if tt:
                    trans_texts.append(tt)
                if ov:
                    trans_texts.append(ov)
        except Exception:
            pass
        # enrich overview by appending extra context
        base_overview = (df.at[i, "overview"] if "overview" in df.columns else "") or ""
        pieces = [base_overview]
        if tagline:
            pieces.append(tagline)
        if coll_name:
            pieces.append(coll_name)
        pieces.extend(alt_titles[:5])  # cap noise
        pieces.extend(trans_texts[:8])
        combined = ". ".join([p for p in pieces if p]).strip()
        if combined:
            df.at[i, "overview"] = combined

        # union keywords
        kw = d.get("keywords", {})
        new_kws = [k.get("name") for k in (kw.get("keywords") or kw.get("results") or [])]
        cur_kws = df.at[i, "keywords"] if "keywords" in df.columns else []
        cur_kws = cur_kws if isinstance(cur_kws, list) else []
        union = sorted({*(k for k in cur_kws if k), *(k for k in new_kws if k)})
        df.at[i, "keywords"] = union

        # tag sources used for transparency
        used = []
        if tagline:
            used.append("tagline")
        if coll_name:
            used.append("collection")
        if alt_titles:
            used.append("alt_titles")
        if trans_texts:
            used.append("translations")
        prev = df.at[i, "sources_used"] if "sources_used" in df.columns else []
        prev = prev if isinstance(prev, list) else []
        df.at[i, "sources_used"] = sorted(set(prev + used))
        # mark that we ran TMDB details enrichment on this row
        df.at[i, "details_pass_ran"] = True

    return df


def analyze(df: pd.DataFrame, out_base: str) -> None:
    # 0) setting_type counts
    if "setting_type" in df.columns:
        st_counts = df["setting_type"].value_counts(dropna=False).rename_axis("setting_type").reset_index(name="count")
        st_counts.to_csv(WRITES_DIR / f"{out_base}_setting_type_counts.csv", index=False)

    # 1) counts by setting_decade
    dec_counts = df["setting_decade"].value_counts(dropna=False).rename_axis("setting_decade").reset_index(name="count")
    dec_counts.to_csv(WRITES_DIR / f"{out_base}_counts_by_setting_decade.csv", index=False)

    # 2) signal usage counts
    sigs = df[["signals_used"]].explode("signals_used")
    sig_counts = sigs["signals_used"].value_counts(dropna=False).rename_axis("signal").reset_index(name="count")
    sig_counts.to_csv(WRITES_DIR / f"{out_base}_signal_counts.csv", index=False)

    # 3) top genres among classified period pieces
    if "genres" in df.columns:
        gg = df[df["is_period_piece"]].copy()
        gg = gg.explode("genres")
        gg["genres"] = gg["genres"].astype(str).str.strip().str.lower()
        genre_counts = gg["genres"].value_counts(dropna=True).rename_axis("genre").reset_index(name="count")
        genre_counts.to_csv(WRITES_DIR / f"{out_base}_top_genres_period_pieces.csv", index=False)

    # 4) release decade vs setting decade crosstab
    def decade_label(y):
        return f"{(int(y)//10)*10}s" if pd.notna(y) else None
    rel_decade = df["release_year"].apply(decade_label)
    ctab = pd.crosstab(rel_decade, df["setting_decade"], dropna=False)
    ctab.to_csv(WRITES_DIR / f"{out_base}_release_vs_setting_decade.csv")

    # 5) coverage metrics
    coverage = pd.DataFrame({
        "rows": [len(df)],
        "period_piece_true": [int(df["is_period_piece"].fillna(False).sum())],
        "missing_setting_years": [int(df["setting_start_year"].isna().sum())],
        "with_revenue": [int((df.get("revenue_usd", 0) > 0).sum()) if "revenue_usd" in df.columns else None],
        "with_votes_ge_200": [int((df.get("vote_count", 0) >= 200).sum()) if "vote_count" in df.columns else None],
    })
    coverage.to_csv(WRITES_DIR / f"{out_base}_coverage.csv", index=False)

    # 6) nostalgia gap stats (mode and median) among period pieces; overall and by release decade
    # exclude explicit futures from gap stats
    pp = df[(df["is_period_piece"]) & (df.get("setting_type") != "future")].copy()
    # ensure numeric
    pp["year_gap"] = pd.to_numeric(pp["year_gap"], errors="coerce")
    # by release decade
    def rel_dec(y):
        try:
            return f"{(int(y)//10)*10}s"
        except Exception:
            return None
    pp["release_decade"] = pp["release_year"].apply(rel_dec)
    def mode_safe(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else None
    gap_stats = pp.groupby("release_decade")["year_gap"].agg(median="median").reset_index()
    modes = pp.groupby("release_decade")["year_gap"].apply(mode_safe).reset_index(name="mode")
    gap_stats = gap_stats.merge(modes, on="release_decade", how="outer").sort_values("release_decade")
    gap_stats.to_csv(WRITES_DIR / f"{out_base}_nostalgia_gap_by_release_decade.csv", index=False)

    # 7) favorites shares: share of period pieces by setting_decade (exclude Unknown/NA)
    fav = pp[pp["setting_decade"].notna()].copy()
    if not fav.empty:
        shares = (fav["setting_decade"].value_counts(normalize=True) * 100.0).rename_axis("setting_decade").reset_index(name="percent")
        shares.to_csv(WRITES_DIR / f"{out_base}_favorites_setting_decade_shares.csv", index=False)

    # 8) release vs setting decade shares (row-normalized)
    if not pp.empty:
        rdec = pp["release_decade"]
        sdec = pp["setting_decade"]
        ctab_counts = pd.crosstab(rdec, sdec, dropna=False)
        ctab_rowshare = ctab_counts.div(ctab_counts.sum(axis=1).replace(0, pd.NA), axis=0) * 100.0
        ctab_rowshare.to_csv(WRITES_DIR / f"{out_base}_release_vs_setting_decade_rowshares.csv")

    # 9) genre shares among confirmed period pieces
    if "genres" in pp.columns:
        gg = pp.explode("genres").copy()
        gg["genres"] = gg["genres"].astype(str).str.strip().str.lower()
        targets = ["history","war","western","biography"]
        total_pp = pp.shape[0]
        data = []
        for g in targets:
            cnt = int((gg["genres"] == g).sum())
            pct = (cnt / total_pp * 100.0) if total_pp else 0.0
            data.append({"genre": g, "count": cnt, "percent": pct})
        pd.DataFrame(data).to_csv(WRITES_DIR / f"{out_base}_genre_shares_period_pieces.csv", index=False)

    print("Wrote summaries to:")
    for name in [
        f"{out_base}_counts_by_setting_decade.csv",
        f"{out_base}_signal_counts.csv",
        f"{out_base}_top_genres_period_pieces.csv",
        f"{out_base}_release_vs_setting_decade.csv",
        f"{out_base}_coverage.csv",
    ]:
        print("-", WRITES_DIR / name)


def main():
    ap = argparse.ArgumentParser(description="Analyze TMDB checkpoints and derive period-piece insights.")
    ap.add_argument("--glob", default="data/period_movies_v1_checkpoint_*.parquet", help="Glob pattern of checkpoint parquet files")
    ap.add_argument("--out-base", default="period_analysis_v1", help="Base name for outputs in writes/")
    ap.add_argument("--details-pass", action="store_true", help="Fetch extra TMDB details for likely period pieces with unknown spans to improve detection")
    ap.add_argument("--wiki-fallback", action="store_true", help="Enable capped wikipedia fallback to fetch first paragraph for hard cases")
    ap.add_argument("--wiki-limit", type=int, default=150, help="Maximum wikipedia pages to attempt")
    ap.add_argument("--wiki-timeout", type=float, default=6.0, help="HTTP timeout seconds for wiki requests")
    ap.add_argument("--wiki-concurrency", type=int, default=2, help="Max concurrent wiki fetches")
    ap.add_argument("--confidence-threshold", type=float, default=0.6, help="Confidence threshold for wiki fallback selection")
    ap.add_argument("--include-future", action="store_true", help="Include future-set films into period-piece KPIs")
    ap.add_argument("--persist-enrichment", action="store_true", help="Persist details_pass enrichment back into checkpoint parquet (writes merged enriched file)")
    ap.add_argument("--overwrite-checkpoints", action="store_true", help="Overwrite original checkpoint parquet files with enriched fields (dangerous)")
    ap.add_argument("--apply-fixes", default=None, help="Path to fixes CSV to apply before classification")
    ap.add_argument("--fixes-dry-run", action="store_true", help="When --apply-fixes is set, do not write output files (dry-run)")
    ap.add_argument("--review-out", default=None, help="Write low-confidence rows to CSV for manual review")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.glob.replace("*","all")))[0]
    out_base = args.out_base or f"analysis_{base}"

    raw, paths, dfs = load_checkpoints(args.glob)
    if args.details_pass:
        raw = details_pass(raw)

    # optional apply fixes before classification
    if args.apply_fixes:
        try:
            from period_pieces.apply_fixes import apply_fixes as _apply_fixes_func, _load_input
            fixes_df = pd.read_csv(args.apply_fixes, dtype=object)
            # apply on the current raw dataframe
            fixed_df, fix_summary = _apply_fixes_func(raw, fixes_df, force_downgrade=False)
            print('Applied fixes summary:', fix_summary)
            if not args.fixes_dry_run:
                # write merged fixed checkpoint
                outp = pathlib.Path('data') / 'period_movies_v1_checkpoint_fixed.parquet'
                fixed_df.to_parquet(outp, index=False)
                print('Wrote fixed merged checkpoint:', outp)
                # swap raw to fixed for classification
                raw = fixed_df
            else:
                print('Fixes dry-run: no files written; classification will proceed on current raw data')
        except Exception as e:
            print('Failed applying fixes:', e)

    # summary of enrichment so far
    details_count = int(raw.get("details_pass_ran").sum()) if "details_pass_ran" in raw.columns else 0
    print(f"details_pass applied to {details_count} rows")
    # optional wikipedia fallback
    if args.wiki_fallback:
        # lazy import to avoid adding requests dependency earlier
        from period_pieces.wiki_enrich import fetch_wikipedia_first_paragraph
        # select rows to attempt wiki enrichment: is_period_piece True and (no setting_start_year or unknown or low confidence)
        sel = raw[(raw.get("is_period_piece").fillna(False)) & ((raw.get("setting_start_year").isna()) | (raw.get("setting_type") == "unknown") | (raw.get("setting_confidence").fillna(0) < args.confidence_threshold))]
        attempted = 0
        successes = 0
        for i, row in sel.iterrows():
            if attempted >= args.wiki_limit:
                break
            # fetch details to get wikidata id
            try:
                mid = int(row["tmdb_id"])
            except Exception:
                continue
            d = None
            try:
                from period_pieces.pipeline import get_movie_details
                d = get_movie_details(mid)
            except Exception:
                d = None
            wikidata_id = None
            if d:
                ext = d.get("external_ids") or {}
                wikidata_id = ext.get("wikidata_id") or ext.get("wikidata")
            if not wikidata_id:
                continue
            attempted += 1
            para = fetch_wikipedia_first_paragraph(str(wikidata_id), timeout=args.wiki_timeout)
            if not para:
                continue
            # append to overview and re-run extraction heuristics by updating fields and marking
            prev_ov = raw.at[i, "overview"] if "overview" in raw.columns else ""
            raw.at[i, "overview"] = (prev_ov or "") + "\n\n" + para
            prev = raw.at[i, "sources_used"] if "sources_used" in raw.columns else []
            prev = prev if isinstance(prev, list) else []
            raw.at[i, "sources_used"] = sorted(set(prev + ["wikipedia"]))
            raw.at[i, "details_pass_ran"] = True
            successes += 1
            time.sleep(max(0.5, args.wiki_timeout / max(1, args.wiki_concurrency)))
        print(f"wiki fallback: attempted={attempted} success={successes}")

    # optionally persist enrichment back into checkpoint files (merged or overwrite originals)
    if args.persist_enrichment:
        # write a merged enriched checkpoint file to data/
        merged_path = pathlib.Path('data') / 'period_movies_v1_checkpoint_enriched.parquet'
        try:
            pathlib.Path('data').mkdir(parents=True, exist_ok=True)
            # sanitize list-like columns before writing parquet
            list_cols = ['sources_used', 'keywords', 'signals_used']
            for c in list_cols:
                if c in raw.columns:
                    try:
                        raw[c] = raw[c].apply(lambda v: json.dumps(list(v)) if (v is not None and not (isinstance(v, (str, bytes))) and hasattr(v, '__iter__')) else v)
                    except Exception:
                        raw[c] = raw[c].astype(str)

            raw.to_parquet(merged_path, index=False)
            print('Wrote merged enriched checkpoint:', merged_path)
        except Exception as e:
            print('Failed writing merged enriched checkpoint:', e)

        if args.overwrite_checkpoints:
            # update each original checkpoint file by merging enriched columns on tmdb_id
            cols_to_persist = [c for c in ['overview', 'keywords', 'sources_used', 'details_pass_ran'] if c in raw.columns]
            if not cols_to_persist:
                print('No enrichment columns present to persist back to originals.')
            else:
                enriched_index = raw.set_index('tmdb_id')
                for pth, original_df in zip(paths, dfs):
                    try:
                        left = original_df.copy()
                        # ensure tmdb_id type alignment
                        left['tmdb_id'] = left['tmdb_id'].astype(str)
                        # join
                        joined = left.merge(raw[ ['tmdb_id'] + cols_to_persist ], on='tmdb_id', how='left', suffixes=(None, '_enr'))
                        # for each col, replace with enriched value when present
                        for c in cols_to_persist:
                            enr = c + '_enr'
                            if enr in joined.columns:
                                joined[c] = joined[enr].where(pd.notna(joined[enr]), joined[c])
                                joined = joined.drop(columns=[enr])
                        joined.to_parquet(pth, index=False)
                        print('Overwrote checkpoint with enriched fields:', pth)
                    except Exception as e:
                        print('Failed to overwrite', pth, e)

    # call classifier with future handling per args
    future_handling = "include" if args.include_future else "separate"
    enriched = classify_period_checkpoints.process(raw, future_handling=future_handling)
    analyze(enriched, out_base)

    # write low-confidence review CSV for manual inspection (optional path or default)
    review_path = pathlib.Path(args.review_out) if args.review_out else (WRITES_DIR / f"{out_base}_low_confidence.csv")
    # select rows with low confidence or unknown setting_type
    low_mask = (enriched.get("setting_confidence").fillna(0) < args.confidence_threshold) | (enriched.get("setting_type") == "unknown")
    review_df = enriched[low_mask].copy()
    if not review_df.empty:
        # pick useful columns if present
        cols = ["tmdb_id", "title", "release_year", "is_period_piece", "setting_type", "setting_confidence", "overview", "sources_used"]
        cols = [c for c in cols if c in review_df.columns]
        review_df[cols].to_csv(review_path, index=False)
        print("Wrote low-confidence review CSV:", review_path)
    else:
        print("No low-confidence rows found; skipping review CSV.")


if __name__ == "__main__":
    main()
