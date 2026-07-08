"""Ngram feature extraction (network faked)."""

from __future__ import annotations

from concordance import ngram


class _Resp:
    def __init__(self, code=200, payload=None):
        self.status_code = code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError
        return self._payload


class _Session:
    def __init__(self, resp):
        self.resp = resp

    def get(self, url, params=None, timeout=None):
        return self.resp


def test_features_from_timeseries():
    # 520 years 1500..2019; a spike in 1600, near-zero recently
    ts = [0.0] * 520
    ts[100] = 1.0e-5           # year 1600
    ts[-1] = 1.0e-7            # 2019
    f = ngram.fetch("forsooth", _Session(_Resp(payload=[{"timeseries": ts}])))
    assert f["peak"] == 1.0e-5
    assert f["peak_year"] == 1600
    assert 0 < f["recency_ratio"] < 0.01     # faded


def test_absent_from_corpus_returns_zeros():
    f = ngram.fetch("zzz", _Session(_Resp(payload=[])))
    assert f == {"peak": 0.0, "recent": 0.0, "recency_ratio": None, "peak_year": None}


def test_network_failure_returns_none():
    assert ngram.fetch("x", _Session(_Resp(code=503))) is None
