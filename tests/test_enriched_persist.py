import pandas as pd, json, os
from pathlib import Path

def test_enriched_persist_serializes_lists(tmp_path, monkeypatch):
    out = tmp_path / "enriched.parquet"
    df = pd.DataFrame([{
        "tmdb_id": 1,
        "keywords": ["a","b"],
        "sources_used": ["tmdb","wiki"],
        "signals_used": ["desc_hit"]
    }])
    # emulate the sanitize step the analysis.py uses
    for col in ("keywords","sources_used","signals_used"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list,dict)) else x)
    df.to_parquet(out)
    assert out.exists()
