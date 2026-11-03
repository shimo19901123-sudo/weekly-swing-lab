"""Weekly Swing Lab: public-data best-effort updater.

Sources: traders.co.jp ranking (Friday close); Yahoo via yfinance (OHLCV).
Margin/earnings clearance are deliberately NEVER auto-certified.
If a live source fails or is stale, do not publish guessed prices.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
DATA = ROOT / "data.json"
ARCHIVE = ROOT / "archive.json"
RANK_URL = "https://www.traders.co.jp/market_jp/stock_ranking/price_up/week/all/all"
EARN_URL = "https://www.traders.co.jp/market_jp/earnings_calendar"
UA = {"User-Agent": "Mozilla/5.0 (compatible; WeeklySwingLab/0.1; research dashboard)", "Accept-Language": "ja-JP,ja;q=0.9"}
JST = ZoneInfo("Asia/Tokyo")


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=25)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    if len(r.content) < 3000:
        raise ValueError(f"Source too small: {url}")
    return r.text


def parse_ranking(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    # Select the date closest to the Japanese ranking heading when available.
    marker = re.search(r"表示条件[：:]\s*前週末比", text)
    focus = text[marker.end():marker.end()+300] if marker else text
    m = re.search(r"(20\d{2})[/\-](\d{1,2})[/\-](\d{1,2})\s+\d{1,2}:\d{2}", focus)
    if not m:
        raise ValueError("Could not verify the ranking as-of timestamp")
    asof = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    stocks, seen = [], set()
    for row in soup.select("tr"):
        cells = row.find_all(["td"], recursive=False)
        if len(cells) < 5:
            continue
        cell_text = [c.get_text(" ", strip=True) for c in cells]
        row_text = " ".join(cell_text)
        matched = re.search(r"\(([0-9]{4}|[0-9]{3}[A-Z])\s*/\s*東\s*([PSG])\)", row_text)
        if not matched:
            continue
        code,market = matched.groups()
        if code in seen:
            continue
        seen.add(code)
        idx = next((i for i,c in enumerate(cell_text) if code in c), None)
        if idx is None:
            continue
        name_text = cell_text[idx]
        name = re.sub(r"\s*\((?:[0-9]{4}|[0-9]{3}[A-Z])\s*/\s*東[PSG]\).*", "", name_text).strip()
        name = re.sub(r"\s+", " ", name)
        # The source columns are rank, name(code/market), price + change, pct, volume, traded value (thousand JPY).
        price_text = cell_text[idx+1] if len(cell_text) > idx+1 else ""
        pct_text = cell_text[idx+2] if len(cell_text) > idx+2 else ""
        volume_text = cell_text[idx+3] if len(cell_text) > idx+3 else ""
        turnover_text = cell_text[idx+4] if len(cell_text) > idx+4 else ""
        price_match = re.search(r"[\d,]+(?:\.\d+)?", price_text)
        pct_match = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*%", pct_text)
        volume_match = re.search(r"[\d,]+", volume_text)
        turnover_match = re.search(r"[\d,]+", turnover_text)
        if not all((name,price_match,pct_match,volume_match,turnover_match)):
            continue
        price = float(price_match.group().replace(",", ""))
        pct = float(pct_match.group(1))
        volume = int(volume_match.group().replace(",", ""))
        turnover = int(turnover_match.group().replace(",", ""))
        if price<=0 or turnover<0 or pct < -100:
            continue
        stocks.append({"code": code, "name": name, "close": price, "gainPct": pct, "turnoverThousand": turnover, "volumeDay": volume, "market": market})
    if len(stocks)<10:
        raise ValueError(f"Only {len(stocks)} rows extracted; likely source changed")
    return asof,stocks[:100]


def estimate_volume_profile(df, last_close: float, bars: int = 120, bins: int = 22) -> dict:
    """Daily-OHLCV **approximation** only: spread each day's volume uniformly
    over its high-low interval. NOT exchange-executed volume at each price.
    """
    import math

    recent = df.tail(bars)
    if len(recent) < 25:
        return {}
    lows = recent['Low'].astype(float)
    highs = recent['High'].astype(float)
    volumes = recent['Volume'].astype(float)
    floor = float(lows.min())
    ceiling = float(highs.max())
    if not math.isfinite(floor) or not math.isfinite(ceiling) or ceiling <= floor:
        return {}
    width = (ceiling - floor) / bins
    distributed = [0.0] * bins
    for low, high, vol in zip(lows, highs, volumes):
        low, high, vol = float(low), float(high), float(vol)
        if not all(math.isfinite(v) for v in (low, high, vol)) or vol <= 0:
            continue
        if high <= low:
            ix = min(bins - 1, max(0, int((low - floor) / width)))
            distributed[ix] += vol
            continue
        for i in range(bins):
            left = floor + width * i
            right = left + width
            overlap = max(0.0, min(high, right) - max(low, left))
            if overlap > 0:
                distributed[i] += vol * overlap / (high - low)
    result_bins = [
        {'low': round(floor + width * i, 2),
         'high': round(floor + width * (i + 1), 2),
         'volume': int(round(distributed[i]))}
        for i in range(bins)
    ]
    if max(distributed) <= 0:
        return {}
    centers = [floor + width * (i + .5) for i in range(bins)]
    max_vol = max(distributed)
    peak_index = max(range(bins), key=lambda i: distributed[i])
    # Strongest estimated dense bands above/below last close; these are NOT verified floors/ceilings.
    below = [i for i in range(bins) if centers[i] < last_close and distributed[i] >= .45 * max_vol]
    above = [i for i in range(bins) if centers[i] > last_close and distributed[i] >= .45 * max_vol]
    support = max(below, key=lambda i: distributed[i]) if below else None
    resistance = max(above, key=lambda i: distributed[i]) if above else None
    return {
        'method': 'daily-high-low-uniform-estimate', 'periodBars': len(recent),
        'poc': round(centers[peak_index], 2),
        'support': round(centers[support], 2) if support is not None else None,
        'resistance': round(centers[resistance], 2) if resistance is not None else None,
        'bins': result_bins,
    }


def maybe_price_history(code: str, until: str):
    import yfinance as yf
    import math

    ticker = yf.Ticker(f"{code}.T")
    # 300-day MA needs more than a year's daily candles.
    df = ticker.history(period="2y", interval="1d", auto_adjust=False, actions=False)
    if df.empty:
        return {}
    dates = df.index.tz_convert(JST).strftime("%Y-%m-%d") if df.index.tz is not None else df.index.strftime("%Y-%m-%d")
    df = df.assign(_date=dates)
    prior = df.loc[df._date <= until].copy()
    if len(prior) < 27:
        return {}
    last = prior.iloc[-1]
    if last["_date"] != until:
        return {}
    close = prior["Close"].astype(float)
    volumes = prior["Volume"].astype(float)
    delta = close.diff()
    avgup = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avgdown = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = float(avgdown.iloc[-1])
    up = float(avgup.iloc[-1])
    rsi = 100.0 if down == 0 and up > 0 else (50.0 if down == 0 else 100 - 100/(1+up/down))
    result = {
        'rsi': round(rsi, 1),
        'volumeRatio': round(float(volumes.iloc[-1]/volumes.iloc[-21:-1].mean()), 2) if volumes.iloc[-21:-1].mean() > 0 else None,
        'volumeAvg20': int(round(float(volumes.iloc[-20:].mean()))),
        'priceAsOf': until,
        'priceSource': 'Yahoo Finance via yfinance / auto, cross-check recommended',
    }
    for period in (5, 25, 75, 100, 200, 250, 300):
        if len(close) >= period:
            value = float(close.iloc[-period:].mean())
            if math.isfinite(value):
                result[f'ma{period}'] = round(value, 2)
    profile = estimate_volume_profile(prior, float(last['Close']))
    if profile:
        result['volumeProfile'] = profile
    return {k:v for k,v in result.items() if not isinstance(v, float) or math.isfinite(v)}


def week_dates(asof: str):
    d=datetime.fromisoformat(asof).date()
    days=(7-d.weekday())%7
    if days==0:days=7
    mon=d+timedelta(days=days)
    return mon.isoformat(),(mon+timedelta(days=4)).isoformat()


def add_sim_history(stocks:list[dict], asof:str, max_tickers=5):
    """Best-effort historical actual OHLC, only when both target dates have passed."""
    now=datetime.now(JST).date()
    mon,fri=week_dates(asof)
    if now < datetime.fromisoformat(fri).date():
        return
    import yfinance as yf
    for s in stocks[:max_tickers]:
        try:
            df=yf.Ticker(f"{s['code']}.T").history(start=mon,end=(datetime.fromisoformat(fri)+timedelta(days=1)).date().isoformat(),interval="1d",auto_adjust=False,actions=False)
            if df.empty:continue
            dd=df.index.tz_convert(JST).strftime('%Y-%m-%d') if df.index.tz is not None else df.index.strftime('%Y-%m-%d')
            dmap={str(d):i for i,d in enumerate(dd)}
            if mon in dmap and fri in dmap:
                s['mondayOpen']=round(float(df.iloc[dmap[mon]]['Open']),2)
                s['fridayClose']=round(float(df.iloc[dmap[fri]]['Close']),2)
        except Exception as exc:
            print(f"[WARN] OHLC for {s['code']}: {exc}",file=sys.stderr)


def enrich_earnings(stocks):
    """Flag positively identified announcements ONLY; never treat missing as confirmation."""
    try:
        soup=BeautifulSoup(fetch_html(EARN_URL),'html.parser')
        for r in soup.select('tr'):
            t=r.get_text(' ',strip=True)
            date=re.search(r'(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)',t)
            code=re.search(r'\(([0-9]{4}|[0-9]{3}[A-Z])/東[PSG]\)',t)
            if date and code:
                for s in stocks:
                    if s['code']==code.group(1):
                        year=datetime.now(JST).year
                        d=f'{year}-{int(date.group(1)):02d}-{int(date.group(2)):02d}'
                        s['earnings']=d
                        s['earningsSource']='Traders-web earnings calendar (partial listing); confirm company IR'
    except Exception as exc:
        print(f"[WARN] Earnings listing unavailable: {exc}",file=sys.stderr)


def friday_update():
    now=datetime.now(JST)
    if now.weekday()!=4:
        raise RuntimeError('Friday update must run on Japan Friday')
    asof,stocks=parse_ranking(fetch_html(RANK_URL))
    if asof!=now.date().isoformat():
        raise RuntimeError(f'Stale ranking {asof}; expected {now.date()} (no overwrite)')
    eligible = [s for s in stocks if 5 <= s['gainPct'] <= 12 and s['volumeDay'] >= 10000 and s['close'] >= 250 and s['turnoverThousand'] >= 50000]
    for s in eligible[:40]:
        try:s.update(maybe_price_history(s['code'],asof))
        except Exception as exc:print(f"[WARN] technicals for {s['code']}: {exc}",file=sys.stderr)
    enrich_earnings(stocks)
    out={"asOf":asof,"source":"traders.co.jp weekly ranking; Yahoo Finance charts where available", "sourceStatus":"automatic-partial","lastCheckedAt":now.strftime('%Y-%m-%d %H:%M JST'), "candidates":stocks}
    assert len(out['candidates'])>=10
    if DATA.exists():
        old=json.loads(DATA.read_text())
        if old.get('asOf') != asof:
            hist=json.loads(ARCHIVE.read_text()) if ARCHIVE.exists() else []
            hist=[v for v in hist if v.get('asOf')!=old.get('asOf')]
            hist.insert(0,old)
            ARCHIVE.write_text(json.dumps(hist[:52],ensure_ascii=False,indent=2)+'\n')
    DATA.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(f'[OK] Friday ranking updated: {asof}, {len(stocks)} stocks')


def daily_update():
    if not DATA.exists():return
    data=json.loads(DATA.read_text())
    if data.get('sourceStatus')=='historical-reference':
        print('[INFO] Seed reference data is preserved, until first Friday live update')
        return
    before=json.dumps(data,sort_keys=True)
    add_sim_history(data['candidates'],data['asOf'], max_tickers=35)
    if json.dumps(data,sort_keys=True)!=before:
        data['lastCheckedAt']=datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')
        DATA.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print('[OK] Filled historical Monday/Friday price outcomes')
    else:print('[INFO] No new comparable historical prices available')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['friday','daily'],required=True)
    args=p.parse_args()
    try:
        (friday_update if args.mode=='friday' else daily_update)()
    except Exception as exc:
        print(f'[ERROR] No data overwritten: {exc}',file=sys.stderr)
        sys.exit(1)
