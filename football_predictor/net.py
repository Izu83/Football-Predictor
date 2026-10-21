"""Small HTTP helper with polite rate limiting and an on-disk JSON cache."""

import hashlib
import json
import time

import requests

from .config import CACHE_DIR

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
MIN_INTERVAL = 0.6  # seconds between live requests

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def get_text(url, params=None, max_age=None, suffix=".txt"):
    """GET a document as text.

    max_age: seconds a cached copy stays valid. None = always fetch fresh,
    float("inf") = cache forever (finished seasons never change).
    """
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}{suffix}"
    if max_age is not None and path.exists() and time.time() - path.stat().st_mtime < max_age:
        return path.read_text(encoding="utf-8")

    global _last_request
    wait = MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = _session.get(url, params=params, timeout=30)
    _last_request = time.time()
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    text = resp.text

    if max_age is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def get_json(url, params=None, max_age=None):
    return json.loads(get_text(url, params, max_age, suffix=".json"))
