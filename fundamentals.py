"""Optional WEEKLY SWING LAB financial reference for Sunday-locked five.

This file never changes ranking, picks, trading prices or performance metrics.
Yahoo Finance / yfinance fundamentals are unofficial and must be cross-checked
with company IR. Missing values stay null, never guessed.
"""
from __future__ import annotations
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data.json'
SUNDAY=ROOT/'sunday_picks.json'
PICKS=ROOT/'picks.json'
OUTPUT=ROOT/'fundamentals.json'
JST=ZoneInfo('Asia/Tokyo')


def finite(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError,OverflowError):
        return None


def round_or_none(v,digits=2):
    n=finite(v)
    return round(n,digits) if n is not None else None


def find_locked(asof):
    for path in (SUNDAY,PICKS):
        if not path.exists():
            continue
        try:
            obj=json.loads(path.read_text(encoding='utf-8'))
            for week in obj.get('weeks',[]):
                if week.get('asOf')==asof and week.get('selectionMode')=='pretrade_locked':
                    arr=week.get('pickedCodes',[])
                    if isinstance(arr,list):
                        return list(dict.fromkeys(str(x) for x in arr))[:5]
        except (OSError,ValueError,TypeError) as ex:
            print(f'[WARN] cannot read {path}: {ex}',file=sys.stderr)
    return []


def parse_info(stock, raw, price_basis):
    m=finite(raw.get('marketCap'))
    cap=round(m/1e8,2) if m is not None and m>0 else None
    div=finite(raw.get('dividendRate'))
    # Some feeds supply annual cash dividend, some don't. Never infer a forecast.
    dividend=round(div,2) if div is not None and div>=0 else None
    if dividend is not None and price_basis is not None and price_basis>0:
        y=round(dividend/price_basis*100,2)
    else:
        y=None
    payout=finite(raw.get('payoutRatio'))
    payout=round(100*payout,2) if payout is not None and payout>=0 else None
    per=round_or_none(raw.get('trailingPE'))
    pbr=round_or_none(raw.get('priceToBook'))
    return dict(code=stock['code'],name=stock['name'],status='ok',
                marketCapOku=cap,dividendYieldPct=y,per=per,pbr=pbr,
                payoutRatioPct=payout,annualDividendYen=dividend,
                priceBasisYen=round_or_none(price_basis),
                note='配当利回り参考＝配当額参考÷ランキング基準終値。配当額は会社予想と異なる場合がある。')


def collect(asof,picks,stocks,loader):
    rows=[]
    for code in picks:
        stock=next((s for s in stocks if str(s.get('code'))==str(code)),None)
        if stock is None:
            rows.append(dict(code=code,name='未取得',status='missing',error='ランキングに該当銘柄なし'))
            continue
        try:
            info=loader(f'{code}.T')
            if not isinstance(info,dict) or not info:
                raise ValueError('fundamental response empty')
            price=finite(stock.get('close'))
            rows.append(parse_info(stock,info,price))
        except Exception as ex:
            print(f'[WARN] {code}: {ex}',file=sys.stderr)
            rows.append(dict(code=code,name=stock['name'],status='missing',error='非公式財務情報の取得失敗。企業IRを要確認。',marketCapOku=None,dividendYieldPct=None,per=None,pbr=None,payoutRatioPct=None,annualDividendYen=None,priceBasisYen=round_or_none(stock.get('close'))))
    return rows


def main():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    asof=str(data.get('asOf',''))
    datetime.fromisoformat(asof)
    codes=find_locked(asof)
    if not codes:
        print('[INFO] No pretrade-locked candidates; fundamentals not generated')
        return
    import yfinance as yf
    rows=collect(asof,codes,data.get('candidates',[]),lambda symbol: yf.Ticker(symbol).info)
    now=datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')
    obj=dict(version=1,asOf=asof,checkedAtJst=now,source='Yahoo Finance via yfinance, unofficial and unverified',scope='five pretrade-locked candidates only',candidateCount=len(codes),candidates=rows,limitations=['Not a JPX full-universe top-50 yield ranking','Dividend data and market cap must be confirmed from corporate IR','No scoring or automatic candidate changes','5-year CAGR not calculated: referenced metric definition unverified'])
    OUTPUT.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[INFO] fundamentals reference saved: {len(rows)} picks / {sum(x["status"]=="ok" for x in rows)} nonempty responses')

if __name__=='__main__':main()
