"""Daily end-of-session review for pre-trade-locked WEEKLY SWING LAB picks.

Reads sunday_picks.json and writes daily_reviews.json only. No trading orders,
no rewriting historical picks, no speculation about investor motives.
Source: Yahoo Finance through yfinance, unofficial OHLCV, best effort.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
JST = ZoneInfo('Asia/Tokyo')
PICKS = ROOT / 'sunday_picks.json'
OUTPUT = ROOT / 'daily_reviews.json'
SOURCE = 'Yahoo Finance via yfinance (非公式、確定日足の再照合が必要)'


def good_number(n, positive=False):
    try:
        v = float(n)
        return math.isfinite(v) and (v > 0 if positive else True)
    except (TypeError, ValueError, OverflowError):
        return False


def number(n, digits=2):
    return round(float(n), digits) if good_number(n) else None


def read_json(path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding='utf-8'))


def normalize(df):
    if df is None or df.empty:
        raise ValueError('OHLCV empty')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).copy()
    if getattr(df.index, 'tz', None) is not None:
        df['_date'] = df.index.tz_convert(JST).strftime('%Y-%m-%d')
    else:
        df['_date'] = df.index.strftime('%Y-%m-%d')
    return df.drop_duplicates(subset=['_date'], keep='last').sort_values('_date')


def make_history_fetcher():
    import yfinance as yf
    return lambda code: yf.Ticker(f'{code}.T').history(
        period='4mo', interval='1d', auto_adjust=False, actions=False)


def make_stock_record(stock, df, day, monday):
    code = str(stock['code'])
    prev = df.loc[df['_date'] < day]
    current = df.loc[df['_date'] == day]
    if current.empty or prev.empty:
        return {'code': code, 'name': stock.get('name', code), 'status': 'missing',
                'reason': '対象日または前営業日の確定日足を取得できません'}
    row, before = current.iloc[-1], prev.iloc[-1]
    o, h, lo, cl, volume = map(float, (row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
    previous_close = float(before['Close'])
    if not all(good_number(x, positive=True) for x in (o, h, lo, cl, previous_close)) or not good_number(volume) or not lo <= min(o,cl) <= max(o,cl) <= h:
        return {'code':code, 'name':stock.get('name',code), 'status':'missing',
                'reason':'取得した日足の高安値・始値・終値に不整合'}
    monday_rows = df.loc[df['_date'] == monday]
    monday_open = (float(monday_rows.iloc[-1]['Open']) if not monday_rows.empty
                   and good_number(monday_rows.iloc[-1]['Open'], positive=True) else None)
    prev_volumes = prev['Volume'].astype(float).tail(20)
    volume_ratio = (volume / float(prev_volumes.mean()) if len(prev_volumes) >= 20
                    and float(prev_volumes.mean()) > 0 else None)
    history_closes = df.loc[df['_date'] <= day, 'Close'].astype(float)
    ma5 = float(history_closes.tail(5).mean()) if len(history_closes)>=5 else None
    ma25 = float(history_closes.tail(25).mean()) if len(history_closes)>=25 else None
    span = h - lo
    upper = h - max(o,cl)
    lower = min(o,cl) - lo
    body = abs(cl - o)
    if span == 0:
        candle = '値幅なし'
    elif body <= span * 0.1:
        candle = '小実体・寄せ線に近い形'
    elif cl>o:
        candle = '陽線'
    else:
        candle = '陰線'
    if span>0 and upper/span>=0.4:
        candle += '・長い上ヒゲ'
    if span>0 and lower/span>=0.4:
        candle += '・長い下ヒゲ'
    support = (stock.get('volumeProfile') or {}).get('support')
    resistance = (stock.get('volumeProfile') or {}).get('resistance')
    support = number(support) if good_number(support, positive=True) else None
    resistance = number(resistance) if good_number(resistance, positive=True) else None
    observations=[]
    if cl>previous_close:observations.append('前営業日の終値より上昇')
    elif cl<previous_close:observations.append('前営業日の終値より下落')
    else:observations.append('前営業日の終値と同値')
    if volume_ratio is not None:
        if volume_ratio>=1.5:observations.append('出来高が直前20営業日平均の1.5倍以上')
        elif volume_ratio<=0.7:observations.append('出来高が直前20営業日平均の70%以下')
    if ma5 is not None:
        observations.append('終値が当日5MA以上' if cl>=ma5 else '終値が当日5MAを下回る')
    if ma25 is not None and cl<ma25:observations.append('終値が当日25MAを下回る')
    if support is not None and cl<support:observations.append('選定時の推計支持帯を終値で下回る（支持帯は実測ではない）')
    if resistance is not None and cl>resistance:observations.append('選定時の推計抵抗帯を終値で上回る（抵抗帯は実測ではない）')
    if span>0 and upper/span>=0.4:observations.append('当日の日足に長い上ヒゲ')
    if span>0 and lower/span>=0.4:observations.append('当日の日足に長い下ヒゲ')
    return {
        'code':code, 'name':stock.get('name',code), 'status':'observed',
        'date':day, 'source':SOURCE,
        'selectionClose':number(stock.get('close')),
        'open':number(o), 'high':number(h), 'low':number(lo), 'close':number(cl),
        'previousClose':number(previous_close), 'volume':int(volume),
        'volumeRatio20':number(volume_ratio), 'ma5':number(ma5), 'ma25':number(ma25),
        'mondayOpen':number(monday_open),
        'dayChangePct':number((cl/previous_close-1)*100),
        'dayProfit100':number((cl-previous_close)*100,0),
        'weekChangePct':number((cl/monday_open-1)*100) if monday_open else None,
        'weekOpenProfit100':number((cl-monday_open)*100,0) if monday_open else None,
        'candle':candle, 'selectionSupportEst':support,
        'selectionResistanceEst':resistance,
        'observations':observations,
    }


def make_day(week, day, tables, now):
    monday=week['tradeMonday']
    universe={str(s.get('code')):s for s in week.get('universe', [])}
    rows=[]
    for code in [str(c) for c in week.get('pickedCodes', [])]:
        stock=universe.get(code,{'code':code, 'name':code})
        df=tables.get(code)
        if df is None:
            rows.append({'code':code,'name':stock.get('name',code),'status':'missing',
                         'reason':'日足の取得失敗または未提供'} )
        else:
            rows.append(make_stock_record(stock, df, day, monday))
    n=[x for x in rows if x['status']=='observed']
    daily_total=sum(float(x['dayProfit100']) for x in n if x['dayProfit100'] is not None)
    week_eligible=[x for x in n if x['weekOpenProfit100'] is not None]
    return {'date':day,'sourceCheckedAtJST':now.isoformat(timespec='minutes'),
            'recordedMode':'historical_reconstruction' if day<now.date().isoformat() else 'day_of_capture',
            'expectedCount':len(rows),'obtainedCount':len(n),
            'weekPnlAvailableCount':len(week_eligible),
            'dayGross100Total':number(daily_total,0) if n else None,
            'weekOpenGross100Total':number(sum(float(x['weekOpenProfit100']) for x in week_eligible),0) if week_eligible else None,
            'dayUpCount':sum(x['dayChangePct']>0 for x in n),
            'dayDownCount':sum(x['dayChangePct']<0 for x in n),
            'complete':len(n)==len(rows) and len(rows)>0,
            'stocks':rows}


def generate(raw, weeks, now, history_fn=None):
    if history_fn is None:history_fn=make_history_fetcher()
    today=now.date().isoformat()
    # Only the most recently pre-trade-locked week can receive new reviews.
    locked=sorted((w for w in weeks if w.get('selectionMode')=='pretrade_locked'
                   and w.get('tradeMonday') and w.get('tradeFriday')
                   and w.get('pickedCodes') and w.get('tradeMonday')<=today),
                  key=lambda w:w['tradeMonday'])
    if not locked:
        print('[INFO] No pre-trade-locked trading week available')
        return raw,0
    week=locked[-1]
    if date.fromisoformat(today)>date.fromisoformat(week['tradeFriday'])+timedelta(days=2):
        print('[INFO] The most recent selected week ended over two days ago; no retroactive auto-review')
        return raw,0
    if now.weekday()<5 and now.time()<time(16,0):
        # Ensure a manual run before close cannot be mistaken for a completed session.
        review_end=(now.date()-timedelta(days=1)).isoformat()
    else:
        review_end=today
    review_end=min(review_end,week['tradeFriday'])
    prior_weeks=raw.setdefault('weeks',[])
    record=next((w for w in prior_weeks if w.get('asOf')==week['asOf']),None)
    if record is None:
        record={'asOf':week['asOf'],'tradeMonday':week['tradeMonday'],
                'tradeFriday':week['tradeFriday'], 'selectionMode':'pretrade_locked',
                'pickedCodes':[str(x) for x in week['pickedCodes']], 'days':[]}
        prior_weeks.append(record)
    # Same week already published entries are preserved to avoid rewriting history.
    stored_days={d['date'] for d in record.get('days',[]) if d.get('complete')}
    if review_end<week['tradeMonday']:
        return raw,0
    start=date.fromisoformat(week['tradeMonday']);end=date.fromisoformat(review_end)
    dates=[(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)
           if (start+timedelta(days=i)).weekday()<5]
    missing_days=[d for d in dates if d not in stored_days]
    if not missing_days:return raw,0
    tables={}
    for code in [str(c) for c in week['pickedCodes']]:
        try:
            tables[code]=normalize(history_fn(code))
        except Exception as err:
            print(f'[WARN] {code}: {err}',file=sys.stderr)
    generated=0
    for day in missing_days:
        result=make_day(week, day, tables, now)
        if result['obtainedCount']==0:
            print(f'[WARN] {day}: no confirmed price bars, not publishing empty review',file=sys.stderr)
            continue
        record['days']=[d for d in record.get('days',[]) if d.get('date')!=day]
        record['days'].append(result)
        generated+=1
    record['days'].sort(key=lambda x:x['date'])
    prior_weeks.sort(key=lambda x:x.get('tradeMonday',''))
    if generated:
        raw['generatedAtJST']=now.isoformat(timespec='minutes')
        raw['source']=SOURCE
    return raw,generated


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--date',help='Local JST date YYYY-MM-DD; tests/manual replay only, default today')
    args=parser.parse_args()
    now=datetime.now(JST)
    if args.date:
        date.fromisoformat(args.date)
        now=datetime.combine(date.fromisoformat(args.date),time(17,15),JST)
    saved=read_json(OUTPUT,{'version':1,'weeks':[]})
    snapshot=read_json(PICKS,{'weeks':[]})
    updated,n=generate(saved,snapshot.get('weeks',[]),now)
    if n:
        OUTPUT.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[INFO] daily reviews created/updated: {n}')


if __name__=='__main__':
    main()
