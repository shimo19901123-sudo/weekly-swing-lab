"""WEEKLY SWING LAB: daily personal-use OHLCV enrichment.

Best-effort unofficial yfinance source; does NOT certify investment decisions,
credit balances, or earnings. Does not alter the weekly ranking percentage.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from update_data import estimate_volume_profile, week_dates

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data.json'
JST = ZoneInfo('Asia/Tokyo')
PERIODS = (5, 25, 75, 100, 200, 250, 300)


def positive(x):
    try:
        return math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


def normalize_dates(df):
    dates = (df.index.tz_convert(JST).strftime('%Y-%m-%d')
             if getattr(df.index, 'tz', None) is not None
             else df.index.strftime('%Y-%m-%d'))
    df = df.copy()
    df['_date'] = dates
    return df[~df['_date'].duplicated(keep='last')].sort_values('_date')


def enrich_one(stock, asof, today, history_fn=None):
    """Fill only verified OHLC dates; keep stale/missing values conspicuous."""
    code = str(stock.get('code', ''))
    if not (5 <= float(stock.get('gainPct', -999)) <= 12
            and int(stock.get('volumeDay', 0)) >= 10000
            and positive(stock.get('close'))
            and float(stock.get('close')) >= 250
            and float(stock.get('turnoverThousand', 0)) >= 50000):
        return False
    if history_fn is None:
        import yfinance as yf
        history_fn = lambda symbol: yf.Ticker(symbol).history(
            period='2y', interval='1d', auto_adjust=False, actions=False)
    df = history_fn(f'{code}.T')
    if df is None or df.empty:
        raise ValueError('OHLCV empty')
    df = normalize_dates(df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']))
    changed = False
    cutoff = df.loc[df['_date'] <= asof]
    if len(cutoff) >= 27 and str(cutoff.iloc[-1]['_date']) == asof:
        closes = cutoff['Close'].astype(float)
        volumes = cutoff['Volume'].astype(float)
        tech = {'priceAsOf': asof,
                'priceSource': 'Yahoo Finance via yfinance (非公式・要照合)'}
        for period in PERIODS:
            if len(closes) >= period:
                v = float(closes.iloc[-period:].mean())
                if math.isfinite(v):
                    tech[f'ma{period}'] = round(v, 2)
        delta = closes.diff()
        ups = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        downs = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        up, down = float(ups.iloc[-1]), float(downs.iloc[-1])
        if math.isfinite(up) and math.isfinite(down):
            tech['rsi'] = round(100 if down == 0 and up > 0 else
                                50 if down == 0 else 100 - 100/(1 + up/down), 1)
        if len(volumes) >= 21:
            base = float(volumes.iloc[-21:-1].mean())
            if base > 0:
                tech['volumeRatio'] = round(float(volumes.iloc[-1])/base, 2)
        vp = estimate_volume_profile(cutoff, float(closes.iloc[-1]))
        if vp:
            tech['volumeProfile'] = vp
        if any(stock.get(k) != v for k, v in tech.items()):
            stock.update(tech)
            changed = True

    mon, fri = week_dates(asof)
    latest_allowed = min(today, fri)
    if latest_allowed >= mon:
        week = df.loc[(df['_date'] >= mon) & (df['_date'] <= latest_allowed)]
        monday = week.loc[week['_date'] == mon]
        # Never substitute a Tuesday opening price for a closed Monday.
        if not monday.empty:
            opening = float(monday.iloc[0]['Open'])
            if positive(opening):
                v = round(opening, 2)
                if stock.get('mondayOpen') != v or stock.get('mondayOpenDate') != mon:
                    stock.update(mondayOpen=v, mondayOpenDate=mon)
                    changed = True
        if not week.empty:
            last = week.iloc[-1]
            day, closing = str(last['_date']), float(last['Close'])
            if positive(closing):
                v = round(closing, 2)
                if stock.get('latestClose') != v or stock.get('latestCloseDate') != day:
                    stock.update(latestClose=v, latestCloseDate=day)
                    changed = True
                if day == fri and stock.get('fridayClose') != v:
                    stock['fridayClose'] = v
                    changed = True
    return changed


def main():
    raw = json.loads(DATA.read_text(encoding='utf-8'))
    asof = str(raw['asOf'])
    datetime.fromisoformat(asof)
    today = datetime.now(JST).date().isoformat()
    pool = [s for s in raw['candidates']
            if 5 <= float(s.get('gainPct', -999)) <= 12
            and int(s.get('volumeDay', 0)) >= 10000
            and positive(s.get('close')) and float(s['close']) >= 250
            and float(s.get('turnoverThousand', 0)) >= 50000][:25]
    succeeded = changed = 0
    for stock in pool:
        try:
            changed += int(enrich_one(stock, asof, today))
            succeeded += 1
        except Exception as e:
            print(f"[WARN] {stock.get('code')}: {e}", file=sys.stderr)
    if changed:
        raw['quotesCheckedAt'] = datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')
        raw['quoteSource'] = 'Yahoo Finance via unofficial yfinance (best effort)'
        DATA.write_text(json.dumps(raw, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'[INFO] eligible={len(pool)} fetched={succeeded} changed={changed}')
    if pool and succeeded == 0:
        raise RuntimeError('All quote downloads failed; saved data was kept unchanged')


if __name__ == '__main__':
    main()
