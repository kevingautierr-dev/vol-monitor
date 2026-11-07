"""STOXX end-of-day file parser (VSTOXX source: Yahoo has no ^V2TX)."""
from volmon import config as C
from volmon import data as D

SAMPLE = "﻿Date;Symbol;Indexvalue\n04.01.1999;V2TX;18.2033\n05.11.2026;V2TX;20.6798\n06.11.2026;V2TX;21.2793\n"


def test_parse_stoxx_txt():
    h = D.parse_stoxx_txt(SAMPLE.lstrip("﻿"))
    assert list(h.columns) == ["Open", "High", "Low", "Close"]
    assert str(h.index[0].date()) == "1999-01-04" and str(h.index[-1].date()) == "2026-11-06"
    assert h["Close"].iloc[-1] == 21.2793
    assert (h["Open"] == h["Close"]).all()


def test_vstoxx_routed_to_stoxx(monkeypatch):
    seen = {}
    monkeypatch.setattr(D, "fetch_stoxx", lambda url: seen.setdefault("url", url))
    D.fetch_history("^V2TX")
    assert seen["url"] == C.STOXX_FILES["^V2TX"]
