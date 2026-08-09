#!/usr/bin/env python3
"""
Download 5 years of funding-rate history from Bybit v5 REST for BTCUSDT and ETHUSDT.
Outputs BTCUSDT_funding_5y.csv and ETHUSDT_funding_5y.csv in the current directory.

Range: 2021-08-08T00:00:00Z (inclusive) .. 2026-08-08T00:00:00Z (exclusive)
No third-party deps (uses urllib). Run:  python fetch_funding.py
"""

import urllib.request, json, time, csv, datetime

BASE = "https://api.bybit.com/v5/market/funding/history"


def iso_to_ms(s: str) -> int:
    dt = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


START_MS = iso_to_ms("2021-08-08T00:00:00Z")
END_MS = iso_to_ms("2026-08-08T00:00:00Z")  # exclusive


def fetch(symbol: str) -> dict:
    rows = {}  # timestamp -> funding_rate
    cur_end = END_MS - 1
    while cur_end >= START_MS:
        url = (
            f"{BASE}?category=linear&symbol={symbol}"
            f"&limit=200&startTime={START_MS}&endTime={cur_end}"
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
        if data.get("retCode") != 0:
            raise SystemExit(f"API error {symbol}: {data.get('retCode')} {data.get('retMsg')}")
        lst = data["result"]["list"]
        if not lst:
            break
        for item in lst:
            t = int(item["fundingRateTimestamp"])
            if START_MS <= t < END_MS:
                rows[t] = float(item["fundingRate"])
        oldest = min(int(i["fundingRateTimestamp"]) for i in lst)
        new_end = oldest - 1  # page backward
        if new_end >= cur_end:
            new_end = cur_end - 1
        cur_end = new_end
        time.sleep(0.15)
    return rows


def sanity(path: str, ts_sorted: list, rates: dict):
    assert ts_sorted == sorted(ts_sorted), f"{path}: not ascending"
    assert len(ts_sorted) == len(set(ts_sorted)), f"{path}: duplicates"
    for t in ts_sorted:
        v = rates[t]
        assert v == v, f"{path}: NaN funding_rate"  # NaN != NaN
    diffs = [b - a for a, b in zip(ts_sorted, ts_sorted[1:])]
    from collections import Counter

    common = Counter(diffs).most_common(3)
    print(f"  {path}: {len(ts_sorted)} rows, no dupes, no NaN")
    print(f"    spacing (ms) most common: {common}")


for sym in ["BTCUSDT", "ETHUSDT"]:
    rates = fetch(sym)
    ts_sorted = sorted(rates)
    path = f"{sym}_funding_5y.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_time_ms", "funding_rate"])
        for t in ts_sorted:
            w.writerow([int(t), float(rates[t])])
    sanity(path, ts_sorted, rates)

print("Done.")
