from __future__ import annotations
import json, os, pathlib, requests, time
from typing import Optional

CACHE_DIR = pathlib.Path('.cache/wiki')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(wikidata_id: str) -> pathlib.Path:
    return CACHE_DIR / f"{wikidata_id}.json"


def fetch_wikipedia_first_paragraph(wikidata_id: str, timeout: int = 6) -> Optional[str]:
    """Given a Wikidata Q-id, resolve English Wikipedia sitelink and return first paragraph text.

    Returns None on failure. Caches result to .cache/wiki/{wikidata_id}.json
    """
    if not wikidata_id:
        return None
    p = _cache_path(wikidata_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            return data.get('extract')
        except Exception:
            pass

    # Resolve sitelink via Wikidata API
    try:
        res = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={'action': 'wbgetentities', 'ids': wikidata_id, 'props': 'sitelinks/urls', 'format': 'json'},
            timeout=timeout,
        )
        res.raise_for_status()
        jd = res.json()
        entities = jd.get('entities', {})
        ent = entities.get(wikidata_id, {})
        sitelinks = ent.get('sitelinks', {})
        enwiki = sitelinks.get('enwiki')
        if not enwiki:
            # No English Wikipedia link available
            p.write_text(json.dumps({'extract': None}), encoding='utf-8')
            return None
        title = enwiki.get('title')
    except Exception:
        return None

    # Fetch the extract (intro) from English Wikipedia
    try:
        wres = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={'action': 'query', 'prop': 'extracts', 'exintro': True, 'explaintext': True, 'titles': title, 'format': 'json'},
            timeout=timeout,
        )
        wres.raise_for_status()
        wj = wres.json()
        pages = wj.get('query', {}).get('pages', {})
        for pid, page in pages.items():
            extract = page.get('extract')
            # take only the first paragraph
            if extract:
                first_para = extract.split('\n\n')[0].strip()
                p.write_text(json.dumps({'extract': first_para}), encoding='utf-8')
                return first_para
    except Exception:
        return None

    return None


def _cache_path_imdb(imdb_id: str) -> pathlib.Path:
    return CACHE_DIR / f"imdb_{imdb_id}.json"


def get_wikidata_id_from_imdb(imdb_id: str, timeout: int = 6) -> Optional[str]:
    """Map an IMDb id (e.g. tt0123456 or 0123456) to a Wikidata Q-id using the SPARQL endpoint.

    Returns the Q-id string (e.g. 'Q12345') or None.
    """
    if not imdb_id:
        return None
    # normalize: ensure it starts with tt
    iid = imdb_id.strip()
    if not iid.startswith("tt"):
        iid = f"tt{iid}"

    # check cache first
    p = _cache_path_imdb(iid)
    if p.exists():
        try:
            jd = json.loads(p.read_text(encoding='utf-8'))
            return jd.get('wikidata_id')
        except Exception:
            pass

    # Use Wikidata SPARQL to find entity with property P345 = imdb id
    sparql = 'https://query.wikidata.org/sparql'
    query = 'SELECT ?item WHERE { ?item wdt:P345 "%s" } LIMIT 1' % iid
    headers = {"Accept": "application/sparql-results+json", "User-Agent": "period-pieces-bot/1.0 (github)"}
    try:
        r = requests.get(sparql, params={"query": query}, headers=headers, timeout=timeout)
        r.raise_for_status()
        jd = r.json()
        results = jd.get("results", {}).get("bindings", [])
        if results:
            uri = results[0].get("item", {}).get("value")
            if uri:
                # uri like https://www.wikidata.org/entity/Q12345
                qid = uri.rsplit("/", 1)[-1]
                p.write_text(json.dumps({"wikidata_id": qid}), encoding='utf-8')
                return qid
    except Exception:
        pass
    # cache miss
    try:
        p.write_text(json.dumps({"wikidata_id": None}), encoding='utf-8')
    except Exception:
        pass
    return None


def fetch_wikipedia_first_paragraph_via_imdb(imdb_id: str, timeout: int = 6) -> Optional[str]:
    """Try to resolve the Wikipedia intro via an IMDb id by first mapping to Wikidata.

    This caches results under .cache/wiki/imdb_{imdb_id}.json and returns the first paragraph or None.
    """
    if not imdb_id:
        return None
    iid = imdb_id.strip()
    p = _cache_path_imdb(iid if iid.startswith("tt") else f"tt{iid}")
    if p.exists():
        try:
            jd = json.loads(p.read_text(encoding='utf-8'))
            if jd.get('extract'):
                return jd.get('extract')
        except Exception:
            pass

    qid = get_wikidata_id_from_imdb(iid, timeout=timeout)
    if not qid:
        return None
    # delegate to existing fetcher which will also cache under the wikidata id
    para = fetch_wikipedia_first_paragraph(qid, timeout=timeout)
    # also cache under imdb key for faster future lookups
    try:
        if para is not None:
            p.write_text(json.dumps({"wikidata_id": qid, "extract": para}), encoding='utf-8')
        else:
            p.write_text(json.dumps({"wikidata_id": qid, "extract": None}), encoding='utf-8')
    except Exception:
        pass
    return para
