import pandas as pd
from period_pieces import apply_fixes


def make_fixture(tmp_path):
    rows = [
        {'tmdb_id': '10', 'title': 'Ten', 'is_period_piece': False},
        {'tmdb_id': '20', 'title': 'Twenty', 'is_period_piece': False},
        {'tmdb_id': '30', 'title': 'Thirty', 'is_period_piece': False},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / 'fixture.parquet'
    df.to_parquet(p, index=False)
    return str(p), df


def test_apply_fixes_dry_run(tmp_path):
    p, df = make_fixture(tmp_path)
    fixes = pd.DataFrame([
        {'tmdb_id': '10', 'setting_start_year': 1970},
        {'tmdb_id': '20', 'is_period_piece': True},
        {'tmdb_id': '30', 'setting_start_year': pd.NA},
    ])
    fixes_p = tmp_path / 'fixes.csv'
    fixes.to_csv(fixes_p, index=False)

    # dry-run: call main with --in-enriched and --dry-run
    rc = apply_fixes.main(['--fixes', str(fixes_p), '--in-enriched', p, '--dry-run'])
    assert rc == 0
    # ensure original fixture unchanged
    df2 = pd.read_parquet(p)
    assert df2.equals(df)


def test_apply_fixes_real(tmp_path):
    p, df = make_fixture(tmp_path)
    fixes = pd.DataFrame([
        {'tmdb_id': '10', 'setting_start_year': 1970},
        {'tmdb_id': '20', 'is_period_piece': True},
    ])
    fixes_p = tmp_path / 'fixes.csv'
    fixes.to_csv(fixes_p, index=False)

    outp = tmp_path / 'fixed.parquet'
    rc = apply_fixes.main(['--fixes', str(fixes_p), '--in-enriched', p, '--no-dry-run', '--out', str(outp)])
    assert rc == 0
    df_fixed = pd.read_parquet(outp)
    # 10 should have setting_start_year
    r10 = df_fixed[df_fixed['tmdb_id'] == '10'].iloc[0]
    assert int(r10['setting_start_year']) == 1970
    # 20 should be marked as period piece
    r20 = df_fixed[df_fixed['tmdb_id'] == '20'].iloc[0]
    assert str(r20['is_period_piece']).lower() in ('true', '1') or r20['is_period_piece'] in (True, 'True')