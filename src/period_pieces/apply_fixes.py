from __future__ import annotations
import argparse
import glob
import pathlib
import time
import json
from typing import Optional
import pandas as pd


def _load_input(in_pattern: Optional[str], in_enriched: Optional[str]):
    if in_enriched:
        p = pathlib.Path(in_enriched)
        if not p.exists():
            raise SystemExit(f"Enriched input not found: {in_enriched}")
        df = pd.read_parquet(p)
        paths = [str(p)]
        dfs = [df.copy()]
        return df, paths, dfs
    if in_pattern:
        paths = sorted(glob.glob(in_pattern))
        if not paths:
            raise SystemExit(f"No checkpoint files matched pattern: {in_pattern}")
        dfs = [pd.read_parquet(p) for p in paths]
        df = pd.concat(dfs, ignore_index=True)
        return df, paths, dfs
    raise SystemExit("Either --in or --in-enriched must be provided")


def _coerce_tmdb(df: pd.DataFrame):
    if 'tmdb_id' in df.columns:
        df['tmdb_id'] = df['tmdb_id'].astype(str)
    return df


def apply_fixes(df: pd.DataFrame, fixes: pd.DataFrame, force_downgrade: bool = False):
    """Apply fixes (DataFrame) onto df (DataFrame) by tmdb_id. Returns (new_df, summary).

    Rules:
    - Only update columns when fixes has a non-null value.
    - Do not downgrade setting_confidence unless force_downgrade True.
    - Mark needs_reclass=True when is_period_piece or any setting_* changed.
    """
    df = df.copy()
    df = _coerce_tmdb(df)
    fixes = fixes.copy()
    fixes = fixes.astype(object)
    if 'tmdb_id' not in fixes.columns:
        raise SystemExit('fixes.csv must include tmdb_id column')
    fixes['tmdb_id'] = fixes['tmdb_id'].astype(str)

    updatable = [
        'is_period_piece', 'setting_start_year', 'setting_end_year', 'setting_decade',
        'setting_type', 'setting_confidence', 'signals_used', 'era_hits', 'sources_used', 'overview', 'keywords', 'notes'
    ]

    touched = 0
    per_col = {c: 0 for c in updatable}
    samples = []

    df_index = df.set_index('tmdb_id') if 'tmdb_id' in df.columns else df

    for _, f in fixes.iterrows():
        tid = str(f['tmdb_id'])
        if tid not in df_index.index:
            continue
        row = df_index.loc[tid]
        changed = False
        for c in updatable:
            if c not in f.index:
                continue
            val = f[c]
            if pd.isna(val):
                continue
            # coerce common types: booleans and numbers from string inputs
            if isinstance(val, str):
                vlow = val.strip().lower()
                if vlow in ('true', '1', 'yes', 'y', 't'):
                    val = True
                elif vlow in ('false', '0', 'no', 'n', 'f'):
                    val = False
                else:
                    # try numeric
                    try:
                        if '.' in vlow:
                            val = float(vlow)
                        else:
                            val = int(vlow)
                    except Exception:
                        pass
            old = row.get(c) if isinstance(row, (pd.Series, dict)) else None
            # handle confidence special case
            if c == 'setting_confidence' and old is not None and not force_downgrade:
                try:
                    oldf = float(old)
                    newf = float(val)
                    if newf < oldf:
                        continue
                except Exception:
                    pass
            # apply
            df_index.at[tid, c] = val
            per_col[c] = per_col.get(c, 0) + 1
            changed = True
        if changed:
            touched += 1
            samples.append({'tmdb_id': tid, 'title': df_index.at[tid, 'title'] if 'title' in df_index.columns else None})
            # set needs_reclass
            df_index.at[tid, 'needs_reclass'] = True

    new_df = df_index.reset_index()
    summary = {'touched_rows': touched, 'per_column': per_col, 'samples': samples[:10]}
    return new_df, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Apply curated fixes to checkpoint parquet(s)')
    ap.add_argument('--fixes', required=True, help='CSV of fixes with tmdb_id and columns to update')
    ap.add_argument('--in', dest='in_pattern', help="Input checkpoint glob pattern (e.g. data/period_movies_v1_checkpoint_*.parquet)")
    ap.add_argument('--in-enriched', help='Single enriched parquet to read')
    ap.add_argument('--dry-run', action='store_true', default=True, help='Do not write output, just print summary')
    ap.add_argument('--no-dry-run', dest='dry_run', action='store_false')
    ap.add_argument('--overwrite-checkpoints', action='store_true', help='Overwrite original checkpoint files (requires --in)')
    ap.add_argument('--backup-dir', default=None, help='Directory to write backups when overwriting')
    ap.add_argument('--force-downgrade', action='store_true', help='Allow lowering setting_confidence')
    ap.add_argument('--out', default='data/period_movies_v1_checkpoint_fixed.parquet', help='Output path for merged fixed parquet')
    args = ap.parse_args(argv)

    fixes = pd.read_csv(args.fixes, dtype=object)
    df, paths, dfs = _load_input(args.in_pattern, args.in_enriched)
    new_df, summary = apply_fixes(df, fixes, force_downgrade=args.force_downgrade)

    print('Apply fixes summary:')
    print(' Rows matched and changed:', summary['touched_rows'])
    print(' Columns updated counts:')
    for k, v in summary['per_column'].items():
        if v:
            print('  ', k, v)
    print(' Sample titles:', [s for s in summary['samples'][:5]])

    if args.dry_run:
        print('Dry-run: no files written')
        return 0

    # sanitize list-like columns before writing parquet (pyarrow dislikes numpy arrays)
    list_cols = ['sources_used', 'keywords', 'signals_used']
    for c in list_cols:
        if c in new_df.columns:
            try:
                new_df[c] = new_df[c].apply(lambda v: json.dumps(list(v)) if (v is not None and not (isinstance(v, (str, bytes))) and hasattr(v, '__iter__')) else v)
            except Exception:
                # fallback: stringify
                new_df[c] = new_df[c].astype(str)

    # write merged fixed parquet
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(outp, index=False)
    print('Wrote fixed merged parquet:', outp)

    if args.overwrite_checkpoints:
        if not args.in_pattern:
            raise SystemExit('--overwrite-checkpoints requires --in to know which files to update')
        # backup originals
        bdir = pathlib.Path(args.backup_dir) if args.backup_dir else pathlib.Path('data') / 'checkpoint_backups' / time.strftime('%Y%m%d-%H%M%S')
        bdir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            bp = bdir / pathlib.Path(p).name
            pathlib.Path(p).rename(bp)
            print('Backed up', p, '->', bp)
        # write per-file by splitting new_df to original segments by tmdb_id
        # naive approach: write single merged file into each original path
        for p in paths:
            new_df.to_parquet(p, index=False)
            print('Overwrote', p)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
