"""WEEKLY SWING LAB v7: immutable pre-trade paper-pick snapshots, Friday reviews,
conservative walk-forward learning. No orders placed, no source code rewriting.

Data from unofficial yfinance. Missing Monday/Fri prices are never fabricated.
A historical seed is labeled retrospective and excluded from strategy learning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
JST = ZoneInfo('Asia/Tokyo')
PICKS = ROOT / 'picks.json'
SUNDAY_PICKS = ROOT / 'sunday_picks.json'
REPORT_DIR = ROOT / 'reports'
INDEX = REPORT_DIR / 'index.json'
LEARNING = ROOT / 'learning.json'
WEEKEND = ROOT / 'weekend_news.json'
BASE_FEE_100 = 0  # fee/tax free gross simulation; never conflate with actual trades
FEATURES = ('rsi_balanced','rsi_hot','volume_expansion','ma_uptrend','above_ma200')


def read_json(path, default):
    if not Path(path).exists():
        return copy.deepcopy(default)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if not path.exists() or path.read_text(encoding='utf-8') != txt:
        path.write_text(txt, encoding='utf-8')


def finite(n):
    try:
        v=float(n)
        return math.isfinite(v)
    except (TypeError,ValueError):
        return False


def val(s,k):
    x=s.get(k)
    return float(x) if finite(x) else None


def eligible(s):
    g,v,p,t=val(s,'gainPct'),val(s,'volumeDay'),val(s,'close'),val(s,'turnoverThousand')
    return all(vv is not None for vv in (g,v,p,t)) and 5<=g<=12 and v>=10000 and p>=250 and t>=50000


def week_dates(asof):
    day=date.fromisoformat(asof)
    mon=day+timedelta(days=(7-day.weekday()) % 7 or 7)
    return mon.isoformat(),(mon+timedelta(days=4)).isoformat()


def features(s):
    r,vr,c,m5,m25,m200=(val(s,k) for k in ('rsi','volumeRatio','close','ma5','ma25','ma200'))
    return {
        'rsi_balanced':None if r is None else 45<=r<=70,
        'rsi_hot':None if r is None else r>70,
        'volume_expansion':None if vr is None else vr>=1.5,
        'ma_uptrend':None if None in (c,m5,m25) else c>m5>m25,
        'above_ma200':None if None in (c,m200) else c>m200,
    }


def base_score(s):
    # Preserve the scoring rule applied by the currently published v6.1 frontend.
    p=0; c=val(s,'close'); t=val(s,'turnoverThousand'); vol=val(s,'volumeDay'); g=val(s,'gainPct')
    if g is not None and 5<=g<=12:p+=20
    if t is not None and t>=50000:p+=6
    if vol is not None and vol>=10000:p+=8
    if t is not None and t>=1000000:p+=5
    m5,m25=val(s,'ma5'),val(s,'ma25')
    if c is not None and m5 is not None and m25 is not None:
        if c>m5:p+=5
        if c>m25:p+=5
        if m5>m25:p+=5
    for a,b in [('ma25','ma75'),('ma75','ma100'),('ma100','ma200'),('ma200','ma250'),('ma250','ma300')]:
        aa,bb=val(s,a),val(s,b)
        if aa is not None and bb is not None and aa>bb:p+=2
    r=val(s,'rsi')
    if r is not None:
        p+=9 if 45<=r<=70 else 3 if 70<r<=80 else 0
    vr=val(s,'volumeRatio')
    if vr is not None:p+=8 if vr>=1.5 else 5 if vr>=1 else 0
    prof=s.get('volumeProfile') or {}
    sup=val(prof,'support');res=val(prof,'resistance')
    if c is not None and c>0:
        if sup is not None and sup<c and (c-sup)/c<=.08:p+=5
        if res is not None and res>c and (res-c)/c<=.025:p-=5
        elif res is not None and res>c and (res-c)/c>=.05:p+=4
    buy,sell,change,short=(val(s,k) for k in ('creditBuy','creditSell','creditDelta','shortDelta'))
    if buy is not None and sell is not None:
        ratio=buy/sell if sell>0 else float('inf')
        p+=8 if ratio<=5 else 3 if ratio<=10 else 0
    if change is not None and change<=0:p+=5
    if short is not None and short<0:p+=3
    return max(0,min(100,p))


def adjusted_score(s, learn):
    base=base_score(s)
    if learn.get('status')!='active':return base
    f=features(s);bonus=sum(int(learn.get('weights',{}).get(k,0)) for k in FEATURES if f.get(k) is True)
    return max(0,min(100,base+max(-6,min(6,bonus))))




# v7.4: Multi-candle structures in user-provided 酒田五法 diagrams.
# Strictly historical, shape-based candidates: no buyer/seller identity inferred.
# Head-and-shoulders and triple formations are recorded only after a neckline break.
def structure_shapes(good, flag):
    if len(good)<18:return
    n=len(good)
    highs=[b['high'] for b in good]
    lows=[b['low'] for b in good]
    closes=[b['close'] for b in good]
    current=closes[-1]
    # Local turning points need two completed bars on both sides.
    hi=[i for i in range(2,n-2) if all(highs[i]>highs[j] for j in (i-2,i-1,i+1,i+2))]
    lo=[i for i in range(2,n-2) if all(lows[i]<lows[j] for j in (i-2,i-1,i+1,i+2))]
    recent_highs=[i for i in hi if i>=n-60]
    recent_lows=[i for i in lo if i>=n-60]
    def same(a,b,tol):return max(a,b)>0 and abs(a-b)/max(a,b)<=tol
    def triples(ids, mode):
        # No foresight: only pivots visible by the last (Friday) bar.
        from itertools import combinations
        for i,j,k in reversed(list(combinations(ids,3))):
            if k<n-12 or j-i<3 or k-j<3 or j-i>24 or k-j>24:continue
            if mode=='shoulder_top':
                if not (same(highs[i],highs[k],.065) and highs[j]>max(highs[i],highs[k])*1.025):continue
                neck=(min(lows[i+1:j+1])+min(lows[j+1:k+1]))/2
                if current<neck*.995:
                    flag('三尊天井','高値圏警戒','左右の山より中央の山が高く、終値が仮ネックラインを下回った形（過去60本内）')
                    return True
            elif mode=='shoulder_bottom':
                if not (same(lows[i],lows[k],.065) and lows[j]<min(lows[i],lows[k])*.975):continue
                neck=(max(highs[i+1:j+1])+max(highs[j+1:k+1]))/2
                if current>neck*1.005:
                    flag('逆三尊','反発候補','左右の谷より中央の谷が深く、終値が仮ネックラインを上回った形（過去60本内）')
                    return True
            elif mode=='triple_top':
                if not (same(highs[i],highs[j],.045) and same(highs[j],highs[k],.045)):continue
                neck=max(min(lows[i+1:j+1]), min(lows[j+1:k+1]))
                if current<neck*.995 and neck<min(highs[i],highs[j],highs[k])*.97:
                    flag('三山（トリプルトップ）','高値圏警戒','近い水準の3つの山と仮ネックライン割れを検出')
                    return True
            elif mode=='triple_bottom':
                if not (same(lows[i],lows[j],.045) and same(lows[j],lows[k],.045)):continue
                neck=min(max(highs[i+1:j+1]), max(highs[j+1:k+1]))
                if current>neck*1.005 and neck>max(lows[i],lows[j],lows[k])*1.03:
                    flag('三川（トリプルボトム）','反発候補','近い水準の3つの谷と仮ネックライン超えを検出')
                    return True
        return False
    triples(recent_highs,'shoulder_top')
    triples(recent_lows,'shoulder_bottom')
    triples(recent_highs,'triple_top')
    triples(recent_lows,'triple_bottom')



# v7.5: Additional patterns based on ten user-provided annotated reference images.
# Simplified reproducible OHLCV checks, not a one-to-one implementation of the
# original book's rules. The original infographic does not provide fixed
# numerical tolerances; these thresholds are exploratory and must be backtested.
# No lookahead beyond the Friday snapshot date; no automatic score adjustment.
EXTRA_PATTERN_NAMES_V75 = (
    '明けの明星','宵の明星','上昇二ツ星','上昇三ツ星','下落二ツ星','下落三ツ星',
    'たくり線','勢力線','首吊り線','放れ五手黒一本底',
    '化け線（下落相場）','化け線（上昇相場）',
    '差し込み線（上昇局面）','差し込み線（下落局面）',
    '切り込み線','入り首線','アテ首線',
    'やぐら底','鍋底','団子天井','陽線五本（下位）','陰線五本（上位）',
    '押さえ込み線','カブセの上抜け','ツタイの打ち返し',
)

def extra_patterns_v75(good, flag, context):
    """Evaluate candidate formations ending on the supplied last candle only."""
    n=len(good)
    if n<3:return
    last=good[-1]
    def rng(x):return max(1e-9,x['high']-x['low'])
    def body(x):return abs(x['close']-x['open'])
    def up(x):return x['close']>x['open']
    def down(x):return x['close']<x['open']
    def avg(seq, key):return sum(x[key] for x in seq)/len(seq)
    def near(a,b,rel=.025):return abs(a-b)<=max(abs(a),abs(b),1)*rel
    def downtrend(k=5):
        if n<k+1:return False
        return good[-k-1]['close']>good[-1]['close']*1.025
    def uptrend(k=5):
        if n<k+1:return False
        return good[-1]['close']>good[-k-1]['close']*1.025
    islow=context=='20日レンジ安値付近'
    ishigh=context=='20日レンジ高値付近'
    a,b,d=good[-3:]
    dbody=body(d)
    # Stars: large first candle, small second, third moves beyond first midpoint.
    if body(a)>=.53*rng(a) and body(b)<=.28*rng(b) and dbody>=.48*rng(d):
        middle=(a['open']+a['close'])/2
        if down(a) and up(d) and d['close']>=middle and b['close']<=a['close']*1.012:
            flag('明けの明星','安値圏参考','大陰線→小実体→陽線が前々日実体の中間以上に回復した簡易形')
        if up(a) and down(d) and d['close']<=middle and b['close']>=a['close']*.988:
            flag('宵の明星','高値圏警戒','大陽線→小実体→陰線が前々日実体の中間以下に下落した簡易形')
    # Two/three stars: following a strong directional candle, small real bodies
    # cluster in a narrow band. A breakout is deliberately not inferred.
    for stars in (3,2):
        if n<stars+1:continue
        leader=good[-stars-1]
        tails=good[-stars:]
        if body(leader)<.58*rng(leader) or any(body(x)>.27*rng(x) for x in tails):continue
        closes=[x['close'] for x in tails]
        if max(closes)-min(closes)>.065*max(closes):continue
        if up(leader) and min(x['low'] for x in tails)>=leader['close']*.965:
            flag(('上昇三ツ星' if stars==3 else '上昇二ツ星'),'上昇継続参考',f'上昇大実体の後に小実体{stars}本が高値近辺で休止。上抜けは未確認')
            break
        if down(leader) and max(x['high'] for x in tails)<=leader['close']*1.035:
            flag(('下落三ツ星' if stars==3 else '下落二ツ星'),'下落警戒',f'下落大実体の後に小実体{stars}本が安値近辺で休止。下抜けは未確認')
            break
    # Long lower shadow = reversal attempt only if price is already low.
    lower=min(d['open'],d['close'])-d['low']
    upper=d['high']-max(d['open'],d['close'])
    if islow and lower>=max(2.3*dbody,.55*rng(d)) and upper<=.18*rng(d):
        flag('たくり線','安値圏参考','20日レンジ安値付近の長い下ヒゲ。翌日以降の回復確認が必要')
    if ishigh and lower>=max(2.1*dbody,.53*rng(d)) and upper<=.20*rng(d):
        flag('首吊り線','高値圏警戒','20日レンジ高値付近の長い下ヒゲ。高値圏で売り圧力を警戒')
    if up(d) and dbody>=.72*rng(d) and d['close']>b['high']:
        if n>=6 and good[-6]['close']>b['close']*1.03:
            flag('勢力線','安値圏参考','下落後、直前高値を上抜く長い陽線（独自の簡易判定）')
    # Countertrend oversized candle ('化け線') signals surprise, NOT reversal.
    if n>=8:
        prev=good[-7:-1]
        prior_mean_range=sum(rng(x) for x in prev)/len(prev)
        if prior_mean_range>0 and rng(d)>=1.55*prior_mean_range and dbody>=.63*rng(d):
            if up(d) and prev[0]['close']>prev[-1]['close']*1.055:
                flag('化け線（下落相場）','要監視','下落基調中の大陽線。これだけで上昇転換とは判断できない')
            if down(d) and prev[0]['close']<prev[-1]['close']*.945:
                flag('化け線（上昇相場）','要監視','上昇基調中の大陰線。これだけで下落転換とは判断できない')
    # The four price levels distinguish the close's penetration of yesterday's
    # real body. These Japanese formations are simplified, not canonical.
    prior=b
    pb=body(prior)
    if pb>0 and pb>=.38*rng(prior):
        if down(prior) and up(d) and d['open']<=prior['close']:
            recovery=(d['close']-prior['close'])/pb
            if d['open']<prior['low'] and .5<=recovery<1:
                flag('切り込み線','安値圏参考','前日陰線の安値より下で寄り、実体の中間以上まで切り返した近似形')
            elif .23<=recovery<.50:
                flag('差し込み線（下落局面）','要監視','下落後の反発陽線が前日陰線実体の中間までは届かない形')
            elif .08<=recovery<.23:
                flag('入り首線','下落警戒','前日陰線の実体に陽線が浅く入り込む。反発力は限定的')
            elif abs(recovery)<.08:
                flag('アテ首線','下落警戒','前日陰線の終値近辺で反発陽線が終了。戻りの弱さに注意')
        if up(prior) and down(d) and d['open']>=prior['close']:
            retracement=(prior['close']-d['close'])/pb
            if .22<=retracement<.5:
                flag('差し込み線（上昇局面）','上昇継続参考','陽線後の陰線が前日実体上部に収まった近似形。次の上抜けが必要')
    # Five black after downward gap then rebound, extremely strict/rare.
    if n>=7:
        five=good[-6:-1]
        if all(down(x) for x in five) and all(five[i]['close']<five[i-1]['close'] for i in range(1,5)) and up(d):
            if dbody>=.60*rng(d) and d['close']>five[-1]['high'] and five[0]['close']>five[-1]['close']*1.06:
                flag('放れ五手黒一本底','安値圏参考','5本連続の下落陰線後の強い陽線。窓・古典定義まで一致するとは限らない')
    # Bottom/top structures: require breakout / breakdown on the current candle.
    if n>=12:
        prev=good[-12:-1]
        lows=[x['low'] for x in prev]
        highs=[x['high'] for x in prev]
        minlo=min(lows); maxhi=max(highs)
        base=good[-9:-1]
        base_w=max(x['high'] for x in base)-min(x['low'] for x in base)
        if up(d) and d['close']>max(x['high'] for x in base)*1.004 and body(d)>=.52*rng(d):
            if base_w/max(min(x['low'] for x in base),1e-9)<=.085 and good[-12]['close']>=avg(base,'close')*1.045:
                flag('鍋底','安値圏参考','下落後に比較的狭いレンジで底固めし、その上限を終値で上抜いた近似形')
        cap=good[-9:-1]
        cap_w=max(x['high'] for x in cap)-min(x['low'] for x in cap)
        if down(d) and d['close']<min(x['low'] for x in cap)*.995 and body(d)>=.52*rng(d):
            if cap_w/max(min(x['low'] for x in cap),1e-9)<=.085 and good[-12]['close']<=avg(cap,'close')*.955:
                flag('団子天井','高値圏警戒','上昇後に狭いレンジで上値が重くなり、下限を終値で割った近似形')
    if n>=7 and up(d) and body(d)>=.62*rng(d) and d['close']>max(x['high'] for x in good[-4:-1]):
        if good[-7]['close']>good[-2]['close']*1.065:
            flag('やぐら底','安値圏参考','短期間の下落後に直近3本の高値を大陽線で上抜く底打ち候補')
    # Five same-color bars at a local extreme; high/low is a necessary qualifier.
    if n>=20:
        prior20=good[-20:-5]
        five=good[-5:]
        avg_rng=sum(rng(x) for x in prior20)/len(prior20)
        if avg_rng>0:
            if all(up(x) and body(x)<.68*avg_rng for x in five):
                if min(x['low'] for x in five)<=min(x['low'] for x in prior20)*1.04:
                    flag('陽線五本（下位）','安値圏参考','安値付近で小さめの陽線5本。上抜けまでは底打ち未確定')
            if all(down(x) and body(x)<.68*avg_rng for x in five):
                if max(x['high'] for x in five)>=max(x['high'] for x in prior20)*.96:
                    flag('陰線五本（上位）','高値圏警戒','高値付近で小さめの陰線5本。下抜けまでは反転未確定')
    # Pullback-continuation patterns verified on the LAST candle.
    if n>=5:
        first=good[-5]
        mid=good[-4:-1]
        if up(first) and body(first)>=.56*rng(first) and up(d) and d['close']>first['high']:
            if all(down(x) and first['low']<=x['low'] and x['high']<=first['high'] for x in mid):
                flag('押さえ込み線','上昇継続参考','大陽線後の3本の調整を終値の高値上抜けで解消した簡易形')
    if n>=3:
        first,middle,last=good[-3:]
        if up(first) and down(middle) and up(last):
            if middle['open']>first['close'] and middle['close']>first['open'] and middle['close']<first['close'] and last['close']>middle['high']:
                flag('カブセの上抜け','上昇継続参考','陽線→上放れ陰線→前日高値を超える陽線の近似形')
        if up(first) and up(middle) and down(last):
            if last['close']<middle['open'] and last['open']>=middle['close']*.985 and ishigh:
                flag('ツタイの打ち返し','高値圏警戒','上昇後に直前陽線の始値を下抜く陰線。戻り売り警戒（暫定判定）')


# v7.6: 13 more formations from user's reference diagrams (IMG_9886--9895).
# These are reproducible, conservative proxies. They are NOT authoritative Sakata
# definitions and are never scored as an established predictive advantage.
# Compute ONLY from daily OHLCV ending on the selection date.
EXTRA_PATTERN_NAMES_V76 = (
    '上げ三法','下げ三方（下げ三法）','上放れの並び赤','下放れの並び赤',
    '小幅上放れ黒線','下放れ二本黒','上値遊び','下値遊び',
    '上放れタスキ','下放れタスキ','毛抜き底','毛抜き天井','波高い線',
)


def extra_patterns_v76(good, flag, position):
    """Provisional reference-day checks; confirmation is never drawn from future bars."""
    n=len(good)
    if n < 2:
        return
    last=good[-1]; prev=good[-2]
    def body(x):return abs(x['close']-x['open'])
    def height(x):return max(x['high']-x['low'],1e-9)
    def bull(x):return x['close']>x['open']
    def bear(x):return x['close']<x['open']
    def close_to(x,y,tolerance=.008):return abs(x-y)/max(x,y,1e-9)<=tolerance
    highzone=position=='20日レンジ高値付近'
    lowzone=position=='20日レンジ安値付近'

    # Retracement stays within the first wide-range candle; fifth candle
    # confirms continuation, not merely a suggestive mid-formation pattern.
    if n>=5:
        first=good[-5]; middle=good[-4:-1]
        contained=all(first['low']<=x['low'] and x['high']<=first['high'] for x in middle)
        modest=all(body(x)<=.62*body(first) for x in middle)
        if contained and modest and body(first)>=.58*height(first) and body(last)>=.52*height(last):
            if bull(first) and bull(last) and last['close']>first['high']*1.002 and middle[-1]['close']<=middle[0]['close']:
                flag('上げ三法','上昇継続参考','先頭の大陽線内で3本の小幅調整後、終値が先頭高値を上抜け（暫定定義）')
            if bear(first) and bear(last) and last['close']<first['low']*.998 and middle[-1]['close']>=middle[0]['close']:
                flag('下げ三方（下げ三法）','下落警戒','先頭の大陰線内で3本の小幅戻り後、終値が先頭安値を下抜け（暫定定義）')

    if n>=3:
        anchor,first,second=good[-3:]
        # Full price gaps (between prior day's full wick and the next day's wick).
        gap_up=first['low']>anchor['high']*1.001
        gap_down=first['high']<anchor['low']*.999
        if gap_up and bull(first) and bull(second) and close_to(first['open'],second['open'],.018) and second['close']>=first['close']*.99:
            flag('上放れの並び赤','上昇継続参考','上方向に窓を空け、陽線2本の始値が近い形（上昇継続は未確定）')
        if gap_down and bull(first) and bull(second) and close_to(first['open'],second['open'],.018):
            flag('下放れの並び赤','要監視','下方向に窓を空けた後に陽線2本。売りと買い戻しのせめぎ合い')
        if gap_up and bear(first) and body(first)<=.6*height(first) and second['close']>=anchor['high']:
            flag('小幅上放れ黒線','高値圏警戒','上放れ後に小幅陰線を挟み、窓を維持。高値追いは警戒')
        if gap_down and bear(first) and bear(second) and second['close']<first['close']:
            flag('下放れ二本黒','下落警戒','下方向の窓の後に陰線2本が連続し、終値を切り下げ')
        if gap_up and bull(first) and bear(second):
            if second['close']>anchor['high'] and second['close']<first['close']:
                flag('上放れタスキ','上昇継続参考','上方窓の後に陰線で調整するが窓を完全には埋めていない')
        if gap_down and bear(first) and bull(second):
            if second['close']<anchor['low'] and second['close']>first['close']:
                flag('下放れタスキ','下落警戒','下方窓の後に陽線で戻すが窓を完全には埋めていない')

    # Two candles meeting at (roughly) the same low/high; a breakout is not
    # assumed. Constrain to local position and opposite real-body colors.
    if lowzone and bear(prev) and bull(last) and close_to(prev['low'],last['low'],.006):
        flag('毛抜き底','安値圏参考','前日陰線と当日陽線の安値が0.6%以内で一致（暫定）。反転確認が必要')
    if highzone and bull(prev) and bear(last) and close_to(prev['high'],last['high'],.006):
        flag('毛抜き天井','高値圏警戒','前日陽線と当日陰線の高値が0.6%以内で一致（暫定）。天井は未確定')
    # High-wave, long double shadows with indecisive small real body.
    if highzone and body(last)<=.22*height(last):
        up=last['high']-max(last['open'],last['close'])
        down=min(last['open'],last['close'])-last['low']
        if up>=.30*height(last) and down>=.30*height(last):
            flag('波高い線','高値圏警戒','高値圏の小実体＋上下の長いヒゲ。乱高下と迷いを示唆する形')

    # '遊び' consists of an impulse, a multi-day consolidation near its end,
    # followed by an actual closing breakout in the direction of the impulse.
    if n>=9:
        impulse=good[-9]
        rest=good[-8:-1]
        hi=max(x['high'] for x in rest); lo=min(x['low'] for x in rest)
        base_width=(hi-lo)/max(rest[-1]['close'],1e-9)
        impulse_size=body(impulse)/max(impulse['close'],1e-9)
        if base_width<=.085 and impulse_size>=.035 and body(last)>=.48*height(last):
            if bull(impulse) and bull(last) and last['close']>hi*1.003 and lo>=impulse['open']:
                flag('上値遊び','上昇継続参考','上昇大陽線後7本の狭い高値もみ合いから終値上抜け（暫定）')
            if bear(impulse) and bear(last) and last['close']<lo*.997 and hi<=impulse['open']:
                flag('下値遊び','下落警戒','下落大陰線後7本の狭い安値もみ合いから終値下抜け（暫定）')

# v7.3: Screenshot-defined Japanese candlestick patterns (extended in v7.4). These are mechanical
# OHLCV shape flags, not evidence of a reversal, order-flow identity, or edge.
# In user's reference charts: red = close > open, green = close < open.
# Detect by prices rather than chart color, which varies by platform.
def candle_shapes(bars, asof):
    """Analyze exact reference-date daily OHLCV; do not look ahead."""
    good=[]
    for b in bars:
        try:
            d=str(b['date'])[:10]
            o,h,l,c,v=(float(b[k]) for k in ('open','high','low','close','volume'))
            if not (l>0 and l<=min(o,c)<=max(o,c)<=h and v>=0 and all(math.isfinite(x) for x in (o,h,l,c,v))):
                continue
            if d<=asof:good.append({'date':d,'open':o,'high':h,'low':l,'close':c,'volume':v})
        except (KeyError,ValueError,TypeError):
            continue
    good.sort(key=lambda x:x['date'])
    if not good or good[-1]['date']!=asof or len(good)<2:
        return {'status':'unavailable','asOf':asof,'patterns':[],
                'note':'選定基準日の連続した日足OHLCVを確認できないため未判定'}
    # Reserve reproducibility: keep only historical candles ending at selection date.
    last=good[-1];prev=good[-2]
    o,h,l,c=last['open'],last['high'],last['low'],last['close']
    rng=h-l;body=abs(c-o);upper=h-max(o,c);lower=min(o,c)-l
    pats=[]
    def flag(name,kind,evidence):
        pats.append({'name':name,'category':kind,'evidence':evidence})
    if rng>0:
        if c<o and lower>=max(1.8*body, .40*rng) and upper<=.32*rng:
            flag('下影陰線','警戒','陰線で長い下ヒゲ。反発はあっても終値が始値を下回る形')
        if c<o and upper>=max(1.8*body, .40*rng) and lower<=.32*rng:
            flag('上影陰線','警戒','陰線で長い上ヒゲ。高値圏では売り圧力を点検')
        if c>o and upper>=max(1.8*body, .45*rng) and lower<=.25*rng:
            flag('陽のトンカチ','警戒','陽線で上ヒゲが長い。高値圏では上値の重さを点検')
        if c>o and body<=.25*rng and upper>=.22*rng and lower>=.22*rng:
            flag('陽のコマ','中立','小さな陽線実体に上下ヒゲ。買いと売りの拮抗候補')
    po,pc,ph,pl=prev['open'],prev['close'],prev['high'],prev['low']
    prevbody=abs(pc-po);prevrange=ph-pl
    inner=min(o,c)>=min(po,pc) and max(o,c)<=max(po,pc)
    small_enough=prevbody>0 and body<=.65*prevbody and prevrange>0 and prevbody/prevrange>=.42
    if inner and small_enough:
        if pc>po:
            kind='陽のはらみ寄せ線' if rng>0 and body/rng<=.12 else '陽の陰はらみ' if c<o else '陽の陽はらみ' if c>o else '陽のはらみ寄せ線'
            flag('陽のはらみ線','高値圏警戒',f'前日陽線の実体内に当日実体が収まる形（{kind}）。出現位置と翌日を要確認')
        elif pc<po:
            kind='陰のはらみ寄せ線' if rng>0 and body/rng<=.12 else '陰の陽はらみ' if c>o else '陰の陰はらみ' if c<o else '陰のはらみ寄せ線'
            flag('陰のはらみ線','安値圏参考',f'前日陰線の実体内に当日実体が収まる形（{kind}）。安値圏の反転は未確定')
    # 7-8. 抱き線: the current real body fully engulfs yesterday's body.
    # Do not mistake a mere overlapping shadow for real-body engulfing.
    if prevbody>0 and prevrange>0 and body>=prevbody*1.05:
        prev_low,prev_high=sorted((po,pc))
        now_low,now_high=sorted((o,c))
        if now_low<=prev_low and now_high>=prev_high and (now_low<prev_low or now_high>prev_high):
            if pc<po and c>o:
                flag('陽の抱き線','安値圏参考','前日陰線の実体を当日陽線の実体が包む。反転には翌日以降の確認が必要')
            elif pc>po and c<o:
                flag('陰の抱き線','高値圏警戒','前日陽線の実体を当日陰線の実体が包む。高値圏では売り圧力に注意')
    # 13-15. 三兵 and 三手大黒線: consecutive rising/falling candle bodies.
    if len(good)>=3:
        a,b,d=good[-3:]
        if all(x['close']>x['open'] for x in (a,b,d)) and a['close']<b['close']<d['close']:
            if all(min(x['open'],x['close'])<=y['open']<=max(x['open'],x['close']) for x,y in ((a,b),(b,d))):
                flag('赤三兵','上昇継続参考','3本続けて陽線・終値切り上げ。高値圏では過熱にも注意')
        if all(x['close']<x['open'] for x in (a,b,d)) and a['close']>b['close']>d['close']:
            if all(min(x['open'],x['close'])<=y['open']<=max(x['open'],x['close']) for x,y in ((a,b),(b,d))):
                flag('黒三兵（三羽ガラス）','下落警戒','3本続けて陰線・終値切り下げ。戻り売りや弱い需給を確認')
        if all(x['close']<x['open'] and (x['open']-x['close'])>=.58*max(x['high']-x['low'],1e-9) for x in (a,b,d)) and a['close']>b['close']>d['close']:
            flag('三手大黒線','安値圏警戒','長い陰線3本で下落加速。売られ過ぎでも反発はまだ未確認')
    # 16-17. 三空: exactly three successive full-range gaps need four bars.
    if len(good)>=4:
        last4=good[-4:]
        if all(last4[i]['low']>last4[i-1]['high'] for i in range(1,4)):
            flag('三空踏み上げ','高値圏警戒','3回連続の上方窓（高値安値で判定）。過熱候補、売り指示ではない')
        if all(last4[i]['high']<last4[i-1]['low'] for i in range(1,4)):
            flag('三空叩き込み','安値圏参考','3回連続の下方窓（高値安値で判定）。反発は未確認')
    # 18-19. 捨て子線: gap-isolated middle doji plus third-day reversal candle.
    if len(good)>=3:
        a,b,d=good[-3:]
        br=max(b['high']-b['low'],1e-9)
        doji=abs(b['close']-b['open'])<=.10*br
        if doji and a['close']<a['open'] and d['close']>d['open'] and b['high']<min(a['low'],d['low']):
            flag('強気の捨て子線','安値圏参考','中央の小さな寄せ線が上下の足と下方で窓を形成し、次の陽線が出た形')
        if doji and a['close']>a['open'] and d['close']<d['open'] and b['low']>max(a['high'],d['high']):
            flag('弱気の捨て子線','高値圏警戒','中央の小さな寄せ線が上下の足と上方で窓を形成し、次の陰線が出た形')
    # 9-12. 3-peak / 3-trough structures require a confirmed neckline break.
    structure_shapes(good,flag)
    range20=good[-20:]
    last20hi=max(x['high'] for x in range20)
    last20lo=min(x['low'] for x in range20)
    # High/low relative position is context, not proof of actual reversal.
    if last20hi<=last20lo:position='レンジ不明'
    else:
        pos=(c-last20lo)/(last20hi-last20lo)
        position='20日レンジ高値付近' if pos>=.78 else '20日レンジ安値付近' if pos<=.22 else '20日レンジ中間'
    extra_patterns_v75(good, flag, position)
    extra_patterns_v76(good, flag, position)
    pastvol=[x['volume'] for x in good[-21:-1] if x['volume']>0]
    vr=round(last['volume']/(sum(pastvol)/len(pastvol)),2) if len(pastvol)>=10 and sum(pastvol)>0 else None
    return {'status':'observed','asOf':asof,'source':'Yahoo Finance via yfinance（日足・非公式）',
            'patterns':pats,'candidatePatternCount':57,'context':position,'volumeRatio20':vr,'patternVersion':'v7.6','availableBars':len(good),
            'latestCandle':last,'previousCandle':prev,
            # For the user's annotated real-chart examples: preserve ONLY pretrade
            # OHLCV. The browser draws an evidence chart; no future bars are read.
            # 65 bars cover ~35 plotted bars and 25MA lookback; compact for iPhone.
            'recentBars':[
                {'date':x['date'],'open':round(x['open'],2),'high':round(x['high'],2),
                 'low':round(x['low'],2),'close':round(x['close'],2),'volume':int(x['volume'])}
                for x in good[-65:]
            ],
            'chartEvidenceVersion':'v7.7',
            'note':'ローソク足の機械的な形状一致。トレンド転換・翌週の上昇を証明しない'}


def fetch_candle_shapes(code, asof):
    """Snapshot only up to the known trading-day close, no future data leakage."""
    try:
        import yfinance as yf
        dt=date.fromisoformat(asof)
        df=yf.Ticker(f'{code}.T').history(
            start=(dt-timedelta(days=175)).isoformat(),
            end=(dt+timedelta(days=1)).isoformat(), interval='1d',
            auto_adjust=False, actions=False, timeout=12)
        if df is None or df.empty:raise ValueError('株価データなし')
        dates=(df.index.tz_convert(JST).strftime('%Y-%m-%d')
               if getattr(df.index,'tz',None) is not None else df.index.strftime('%Y-%m-%d'))
        bars=[{'date':str(d),'open':row['Open'],'high':row['High'],
               'low':row['Low'],'close':row['Close'],'volume':row['Volume']}
              for d,(_,row) in zip(dates,df.iterrows())]
        return candle_shapes(bars,asof)
    except Exception as ex:
        return {'status':'unavailable','asOf':asof,'patterns':[],
                'note':'選定時の連続日足を取得できず未判定（'+str(ex)[:65]+'）'}


def selection_rationale(s):
    """Only Friday market facts and explicit Sunday weekend hypotheses."""
    c=val(s,'close')
    facts=[]
    risks=[]
    unknown=[]
    g=val(s,'gainPct')
    v=val(s,'volumeDay')
    t=val(s,'turnoverThousand')
    if g is not None:facts.append(f'週間騰落率 {g:+.2f}%（ランキング基準日）')
    if v is not None:facts.append(f'金曜出来高 {v:,.0f}株')
    if t is not None:facts.append(f'売買代金 {t/100000:.2f}億円')
    m5,m25,m200,m300=(val(s,k) for k in ('ma5','ma25','ma200','ma300'))
    if c is not None and m5 is not None and m25 is not None:
        facts.append(f'終値と短期MA：株価{c:,.1f}円／5MA{m5:,.1f}円／25MA{m25:,.1f}円')
        dev=(c/m25-1)*100 if m25>0 else None
        if dev is not None:
            facts.append(f'25MA乖離率 {dev:+.1f}%')
            if dev>8:risks.append('25MAからの上方乖離が大きい（8%は仮の警戒ライン）')
    else:unknown.append('短期MAまたは終値が不足')
    for value,name in ((m200,'200MA'),(m300,'300MA')):
        if c is not None and value is not None:facts.append(f'{name} {value:,.1f}円：終値は{"上" if c>=value else "下"}')
        else:unknown.append(f'{name}未取得')
    unknown.append('MA200/300の傾き・押し安値切り上げ・N字の確定には別途時系列分析が必要')
    candles=s.get('candleAnalysis') or {}
    if candles.get('status')=='observed':
        facts.append(f"日足基準日 {candles['asOf']}：{candles.get('context','不明')}。足型検出 {len(candles.get('patterns',[]))}件")
        if candles.get('volumeRatio20') is not None:facts.append(f"足型当日の出来高倍率 {candles['volumeRatio20']:.2f}倍（直前20日平均・有効10日以上）")
        for pat in candles.get('patterns',[]):
            facts.append(f"足型一致：{pat['name']}（{pat['category']}）／{pat['evidence']}")
            if '警戒' in pat['category']:risks.append(f"{pat['name']}を検出。高値圏・出来高・翌日の値動きを再確認")
        if candles.get('recentBars'):
            bars=candles['recentBars']
            facts.append(f"選定時点の実チャート証拠：{len(bars)}営業日分のOHLCVを保存（{bars[0]['date']}〜{bars[-1]['date']}）。終値は予測ではなく過去の実測参照。")
        if not candles.get('patterns'):facts.append('今回の57分類に該当する足型なし（強気・弱気を断定しない）')
    else:unknown.append('酒田57分類の連続日足データが未取得：判定不能')
    rsi=val(s,'rsi')
    if rsi is not None:
        facts.append(f'RSI {rsi:.1f}')
        if rsi>70:risks.append('RSI70超：過熱に注意')
    else:unknown.append('RSI未確認')
    vr=val(s,'volumeRatio')
    if vr is not None:facts.append(f'出来高倍率 {vr:.2f}倍（20日平均対比）')
    else:unknown.append('出来高倍率未確認')
    buy,sell,delta=(val(s,k) for k in ('creditBuy','creditSell','creditDelta'))
    if buy is not None and sell is not None:facts.append(f'信用買残{buy:,.0f}／売残{sell:,.0f}株')
    else:unknown.append('信用買残・売残とその基準日未確認')
    if delta is not None:facts.append(f'信用買残増減{delta:+,.0f}株')
    else:unknown.append('信用買残の推移未確認')
    unknown.extend(['機関空売りの実際の建値・意図は不明', '企業材料・決算予定・板の厚さ・業種対比は別途一次情報で確認'])
    p=s.get('volumeProfile') or {}
    support=val(p,'support'); resistance=val(p,'resistance')
    rr=None
    if c is not None and support is not None and resistance is not None and 0<support<c<resistance:
        rr=round((resistance-c)/(c-support),2)
        facts.append(f'参考支持帯{support:,.1f}円／抵抗帯{resistance:,.1f}円・仮RR {rr}:1（日足均等配分推計）')
        if rr<2:risks.append('参考RRは2：1未満。ただし支持抵抗そのものも推計')
    else:unknown.append('買い指値・損切り・利確の合理的な価格帯不足：RR算出不可')
    risks.append('金曜の先行上昇、翌週のギャップ、戻り売り、決算・公表遅延は要確認')
    return {'facts':facts,'riskHypotheses':risks,'unverified':unknown,'estimatedRR':rr,
            'note':'事前固定時点の観察。予測・実売買推奨ではなく、未確認事項は良好扱いしない。'}


def snapshot_stock(s, learn):
    out={k:s.get(k) for k in ('code','name','market','close','gainPct','volumeDay','turnoverThousand','ma5','ma25','ma75','ma100','ma200','ma250','ma300','rsi','volumeRatio','creditBuy','creditSell','creditDelta','shortDelta','earnings','sector','priceAsOf') if k in s}
    out['baseScore']=base_score(s)
    out['decisionScore']=adjusted_score(s,learn)
    out['features']=features(s)
    if isinstance(s.get('volumeProfile'),dict):
        out['volumeProfile']={k:s['volumeProfile'].get(k) for k in ('method','support','resistance') if k in s['volumeProfile']}
    if isinstance(s.get('candleAnalysis'),dict):out['candleAnalysis']=s['candleAnalysis']
    out['selectionRationale']=selection_rationale(s)
    return out


def freeze(data, today=None, legacy=False, clock=None):
    """One Sunday-locked snapshot per Friday ranking date; never rewrite."""
    clock=clock or datetime.now(JST)
    today=today or clock.date().isoformat()
    asof=data['asOf']
    mon,fri=week_dates(asof)
    # Keep already-published Friday picks immutable; Sunday decisions live in a separate file.
    store_path = PICKS if legacy else SUNDAY_PICKS
    store=read_json(store_path, {'version':1,'weeks':[]})
    existing=next((x for x in store['weeks'] if x['asOf']==asof),None)
    if existing:return existing
    if not legacy:
        sunday=date.fromisoformat(today)
        friday=(sunday-timedelta(days=2)).isoformat()
        if sunday.weekday()!=6 or asof!=friday:
            raise ValueError('日曜選定には直近金曜の確定ランキングが必要（古い週を固定しない）')
        if clock.date().isoformat()==today and clock.time()<time(20,40):
            raise ValueError('日曜20:40 JSTより前は候補を確定させない')
    if not legacy and data.get('sourceStatus')=='historical-reference':
        raise ValueError('Historical sample may only be saved as retrospective')
    learn=read_json(LEARNING, {'status':'insufficient_data','weights':{}})
    universe=[snapshot_stock(s,learn) for s in data['candidates'] if eligible(s)]
    # Known earnings in the holding week are excluded; unknown is NOT verified as safe.
    universe=[s for s in universe if not (s.get('earnings') and mon<=s['earnings']<=fri)]
    universe.sort(key=lambda s:(-s['decisionScore'],-(s.get('turnoverThousand') or 0),s['code']))
    weekend=read_json(WEEKEND,{}) if not legacy else {}
    if not legacy and (weekend.get('rankFriday')!=asof or weekend.get('selectionSunday')!=today):
        weekend={'status':'unavailable','source':'未取得','companies':{},
                 'limitations':['日曜の週末ニュース収集が未実行または失敗。週末材料は未確認。']}
    alerts=[]
    if not legacy:
        for stock in universe:
            n=(weekend.get('companies') or {}).get(stock['code']) or {}
            stock['weekendRisk']={'retrieval':n.get('retrieval','unavailable'),
                'needsReview':bool(n.get('needsReview')),
                'headlines':n.get('riskHeadlines',[])[:3],
                'note':n.get('note','企業IRを含む最新情報を確認できていない')}
            if n.get('needsReview'):alerts.append(stock['code'])
            stock['selectionRationale']=selection_rationale(stock)
            stock['selectionRationale']['unverified'].append('土日〜月曜朝の材料を公式IRで最終確認するまで売買は未確認')
            if n.get('needsReview'):
                stock['selectionRationale']['riskHypotheses'].append('週末ニュースの見出しに懸念語を検出。一次IR照合まで候補の自動選定を保留')
        # A worrying *headline* creates a conservative hold, not a conclusion
        # about corporate facts or evidence of a real change in fundamentals.
        ranked_safe=[s for s in universe if s['code'] not in alerts]
        picked=[s['code'] for s in ranked_safe[:5]]
        for stock in ranked_safe[:5]:
            stock['candleAnalysis']=fetch_candle_shapes(stock['code'],asof)
            stock['selectionRationale']=selection_rationale(stock)
            stock['selectionRationale']['unverified'].append('日曜時点でも週末ニュースの網羅確認・最新IR確認は未完了')
            stock['weekendRisk']['note']='週末見出しの自動照合は限定的。一次情報の確認が必要。'
        # Always store the top selections even if blocked candidates were high ranked.
        universe=ranked_safe[:5]+[s for s in universe if s['code'] not in picked]
        universe=universe[:25]
    if legacy:
        # The initial v6.x app shipped with these five default *research* picks.
        default_codes=['7360','6262','3399','7384','7309']
        picked=[s['code'] for code in default_codes for s in universe if s['code']==code]
        mode='retrospective_initial_reference'
    else:
        mode='pretrade_locked'
    out={'asOf':asof,'tradeMonday':mon,'tradeFriday':fri,'selectionMode':mode,
         'snapshotDate':today,'selectionTiming':'Sunday-before-trading',
         'selectionTimeJST':clock.strftime('%Y-%m-%d %H:%M JST') if not legacy else '',
         'weekendScreen':{'status':weekend.get('status','unavailable'),
                          'source':weekend.get('source','未取得'),
                          'collectedAt':weekend.get('collectedAt'),
                          'riskHeldCodes':alerts},
         'dataSource':data.get('source',''),
         'learningVersion':learn.get('version',0),'pickedCodes':picked,
         'universe':universe[:25],
         'limitations':'日曜夜に固定した調査候補。週末見出しは網羅的・一次IR確認済みではない。決算/信用残は未確認なら買い推奨ではない。100株紙上比較。'}
    store['weeks'].append(out)
    store['weeks'].sort(key=lambda x:x['asOf'])
    write_json(store_path,store)
    return out


def yahoo_prices(code, mon, fri):
    """Use exact Monday open, exact Friday close. No holiday or missing-day substitution."""
    import yfinance as yf
    df=yf.Ticker(f'{code}.T').history(start=mon,end=(date.fromisoformat(fri)+timedelta(days=1)).isoformat(),interval='1d',auto_adjust=False,actions=False)
    if df is None or df.empty:return None,None,'データ取得失敗'
    d=(df.index.tz_convert(JST).strftime('%Y-%m-%d') if getattr(df.index,'tz',None) is not None else df.index.strftime('%Y-%m-%d'))
    rows={str(day):row for day,(_,row) in zip(d,df.iterrows())}
    a=rows.get(mon); b=rows.get(fri)
    if a is None or b is None:return None,None,'月曜始値または金曜終値がない（休場・取引停止等）'
    opening=float(a['Open']); closing=float(b['Close'])
    if not all(math.isfinite(x) and x>0 for x in (opening,closing)):
        return None,None,'無効な価格データ'
    return round(opening,2),round(closing,2),'Yahoo Finance via yfinance / unofficial'


def review_entry(s, picked, prices):
    code=s['code']; a,b,source=prices
    valid=a is not None and b is not None and a>0 and b>0
    p=round((b-a)*100) if valid else None
    ret=round((b/a-1)*100,2) if valid else None
    f=s.get('features') or features(s)
    notes=[]
    if f.get('rsi_hot') is True:notes.append('選定時RSI70超：急騰後の反落リスクを点検')
    if f.get('volume_expansion') is False:notes.append('選定時出来高倍率1.5未満：勢いの継続を検証')
    if f.get('ma_uptrend') is False:notes.append('5日・25日線の上昇配列を満たさない／位置関係に注意')
    if not all(k in s for k in ('creditBuy','creditSell','creditDelta')):
        notes.append('信用需給未確認：買い残・売り残の確認が必要')
    if not s.get('earnings'):notes.append('決算日未確認：発表の有無を再確認')
    if s.get('selectionRationale') and s['selectionRationale'].get('estimatedRR') is not None:
        notes.append('事前の仮RRと週次実績を比較。価格帯別出来高は推計であり約定値を意味しない')
    if valid and ret<0:notes.append('損失：要因を事後断定せず、チャート・需給と照合する')
    if valid and ret>0:notes.append('利益：偶然の可能性もあり同一条件の再現を確認')
    return {'code':code,'name':s.get('name',''),'picked':picked,
            'mondayOpen':a,'fridayClose':b,'grossProfit100':p,'returnPct':ret,
            'status':('勝ち' if p>0 else '負け' if p<0 else '引分') if valid else '未確定',
            'source':source,'selectionBaseScore':s.get('baseScore'),
            'selectionDecisionScore':s.get('decisionScore'),'features':f,
            'selectionRsi':s.get('rsi'),'selectionVolumeRatio':s.get('volumeRatio'),
            'reviewPoints':notes,'selectionRationale':s.get('selectionRationale'),
             'candleAnalysis':s.get('candleAnalysis'),
             'weekendRisk':s.get('weekendRisk')}


def build_report(snapshot, price_fn=yahoo_prices, generated=None):
    selected=set(snapshot['pickedCodes']); entries=[]
    for s in snapshot['universe']:
        try:prices=price_fn(s['code'],snapshot['tradeMonday'],snapshot['tradeFriday'])
        except Exception as e:prices=(None,None,f'価格取得エラー: {str(e)[:90]}')
        entries.append(review_entry(s,s['code'] in selected,prices))
    picks=[x for x in entries if x['picked']]
    known=[x for x in picks if x['grossProfit100'] is not None]
    losses=[x for x in known if x['grossProfit100'] < 0]
    winners=[x for x in known if x['grossProfit100'] > 0]
    observations=[]
    if known:
        observations.append(f'評価可能{len(known)}件中、利益{len(winners)}件・損失{len(losses)}件（仮想100株）。')
    if losses:
        hot=sum(x.get('features',{}).get('rsi_hot') is True for x in losses)
        observations.append(f'損失{len(losses)}件のうち選定時RSI70超は{hot}件。過熱の影響は別週でも検証する。')
    if known:
        lowv=sum(x.get('features',{}).get('volume_expansion') is False for x in known)
        observations.append(f'評価可能{len(known)}件中、選定時の出来高倍率1.5未満は{lowv}件。利益との関係を複数週で比較。')
    missing=len(picks)-len(known)
    if missing:observations.append(f'{missing}件は正確な月曜始値または金曜終値が取得できず、損益集計から除外。')
    observations.append('次回はMA配列・信用買残増減・決算予定・業種強弱を選定時点で記録し、損益との関連を検証する。')
    observations.append('観測結果は原因の証明ではなく、条件変更は複数週の時系列検証が必要。')
    wins=sum(x['grossProfit100']>0 for x in known)
    principal=sum(x['mondayOpen']*100 for x in known)
    total=sum(x['grossProfit100'] for x in known)
    r={'version':1,'asOf':snapshot['asOf'],'tradeMonday':snapshot['tradeMonday'],
       'tradeFriday':snapshot['tradeFriday'],'generatedAt':generated or datetime.now(JST).strftime('%Y-%m-%d %H:%M JST'),
       'selectionMode':snapshot['selectionMode'],
       'learningEligible':snapshot['selectionMode']=='pretrade_locked',
       'lockedCodes':snapshot['pickedCodes'],'expectedCount':len(picks),
       'completeCount':len(known),'complete':len(known)==len(picks) and len(picks)>0,
       'winners':wins,'losers':sum(x['grossProfit100']<0 for x in known),
       'winRatePct':round(wins/len(known)*100,1) if known else None,
       'grossProfit100Total':total if known else None,
       'capitalEvaluated':round(principal) if known else None,
       'portfolioReturnPct':round(total/principal*100,2) if principal else None,
       'picks':picks,'universeResults':entries,'lessons':observations,
       'notes':['手数料・税金・スリッページ・指値不成立は反映していません。',
                '月曜日が休場なら火曜始値で代用せず、当該銘柄は未確定とします。',
                '信用残・決算など未確認情報を「問題なし」とは判定しません。',
                '結果から過去の選定条件を上書きしません。結果と原因の関係は仮説です。']}
    return r


def md_report(report):
    r=report
    f=lambda x: '--' if x is None else f'{x:+,.0f}'
    lines=[f'# WEEKLY SWING LAB 週次振り返り {r["tradeFriday"]}',
           f'- 選定基準：{r["asOf"]} / 売買期間：{r["tradeMonday"]}〜{r["tradeFriday"]}',
           f'- 種類：{"事前固定候補" if r["learningEligible"] else "過去の初期参考候補（学習対象外）"}',
           f'- 評価可能：{r["completeCount"]}/{r["expectedCount"]}銘柄',
           f'- 勝率：{r["winRatePct"] if r["winRatePct"] is not None else "未確定"}%（評価可能分のみ）',
           f'- 100株ずつの合計損益（税・手数料前）：{f(r["grossProfit100Total"])}円',
           f'- 購入額に対する騰落率：{r["portfolioReturnPct"] if r["portfolioReturnPct"] is not None else "未確定"}%',
           '', '| 銘柄 | 月曜始値 | 金曜終値 | 100株損益 | 騰落率 | 状態 |',
           '|---|---:|---:|---:|---:|---|']
    for p in r['picks']:
        lines.append(f'| {p["code"]} {p["name"]} | {p["mondayOpen"] if p["mondayOpen"] is not None else "未取得"} | {p["fridayClose"] if p["fridayClose"] is not None else "未取得"} | {f(p["grossProfit100"])} | {p["returnPct"] if p["returnPct"] is not None else "—"}% | {p["status"]} |')
    lines.extend(['','## 銘柄ごとの振り返り（因果関係の断定ではありません）'])
    for p in r['picks']:
        lines.append(f'### {p["name"]}（{p["code"]}）')
        lines.extend('- '+item for item in p['reviewPoints'])
    lines.extend(['','## 今週の観測と次回の確認事項']+['- '+x for x in r.get('lessons',[])]+['','## 今後の検証','- 週間上昇率5〜12％・金曜出来高1万株以上・決算跨ぎ回避は固定。',
                  '- 上昇銘柄と下落銘柄の選定時RSI・出来高倍率・MA配列を比較。',
                  '- 取引条件を変える前に複数週で再現性を検証。','']+['- '+n for n in r['notes']])
    return '\n'.join(lines)+'\n'


def finalize(today=None, price_fn=yahoo_prices):
    today=today or datetime.now(JST).date().isoformat()
    history=read_json(PICKS,{'version':1,'weeks':[]})
    data=read_json(ROOT/'data.json',{})
    if data.get('sourceStatus')=='historical-reference' and data.get('asOf') and not any(w['asOf']==data['asOf'] for w in history['weeks']):
        if week_dates(data['asOf'])[1]<=today:
            freeze(data,today=today,legacy=True)
            history=read_json(PICKS,{'version':1,'weeks':[]})
    # Sunday-fixed candidates replace Friday's preliminary picks for the SAME
    # trading week in reporting only. The Friday JSON remains unchanged/auditable.
    sunday=read_json(SUNDAY_PICKS,{'version':1,'weeks':[]})
    by_week={w['asOf']:w for w in history['weeks']}
    for w in sunday.get('weeks',[]):
        if w.get('selectionTiming')=='Sunday-before-trading':
            by_week[w['asOf']]=w
    report_weeks=sorted(by_week.values(),key=lambda x:x['asOf'])
    REPORT_DIR.mkdir(exist_ok=True)
    rows=[]
    for wk in report_weeks:
        if wk['tradeFriday']>today:continue
        # Never file an end-of-week report while Friday trading is ongoing.
        clock=datetime.now(JST)
        if wk['tradeFriday']==today and clock.date().isoformat()==today and clock.time()<time(15,30):
            continue
        report_path=REPORT_DIR/(wk['asOf']+'.json')
        saved=read_json(report_path, {})
        r=saved if saved.get('complete') else build_report(wk,price_fn)
        write_json(report_path,r)
        md_path=REPORT_DIR/(wk['asOf']+'.md')
        md=md_report(r)
        if not md_path.exists() or md_path.read_text(encoding='utf-8')!=md:
            md_path.write_text(md,encoding='utf-8')
        rows.append({'asOf':r['asOf'],'tradeFriday':r['tradeFriday'],
                     'complete':r['complete'],'winRatePct':r['winRatePct'],
                     'grossProfit100Total':r['grossProfit100Total'],'portfolioReturnPct':r['portfolioReturnPct'],
                     'selectionMode':r['selectionMode'],'expectedCount':r['expectedCount'],
                     'completeCount':r['completeCount'],
                     'reportPath':'reports/'+wk['asOf']+'.json'})
    rows.sort(key=lambda r:r['asOf'],reverse=True)
    write_json(INDEX,{'version':1,'reports':rows})
    return rows


def calculate_learning(reports, min_weeks=16):
    """Walk-forward check against frozen weekly universes.

    Adapt only additive score (max ±6). Hard filters never change.
    Trials include only snapshots frozen before trading and complete weekly prices.
    """
    completed=[r for r in reports if r.get('learningEligible') and r.get('complete')]
    completed.sort(key=lambda r:r['asOf'])
    result={'version':1,'status':'insufficient_data','totalCompleteWeeks':len(completed),
            'minimumWeeks':min_weeks,'weights':{},'appliedChanges':[],
            'reason':'紙上評価が不足。選定基準は変更せず観察する。',
            'safeguards':['上昇率5〜12%、出来高1万株以上、決算跨ぎ回避を自動変更しない',
                          '初期参考週や未確定価格は学習除外','4週間の時間順ホールドアウトが基準を上回る場合に限り小幅に補正']}
    if len(completed)<min_weeks:return result
    train,holdout=completed[:-4],completed[-4:]
    observations=[]
    for r in train:
        for s in r['universeResults']:
            if s.get('returnPct') is not None:
                observations.append(s)
    if len(observations)<80:
        result['reason']='学習可能な事前固定母集団が80件未満'
        return result
    weights={}
    details={}
    for key in FEATURES:
        one=[s['returnPct'] for s in observations if s.get('features',{}).get(key) is True]
        zero=[s['returnPct'] for s in observations if s.get('features',{}).get(key) is False]
        if len(one)<15 or len(zero)<15:continue
        diff=sum(one)/len(one)-sum(zero)/len(zero)
        # Avoid extremes from a single lucky stock; cap adjustment per feature.
        w=3 if diff>=1.5 else 2 if diff>=0.7 else -3 if diff<=-1.5 else -2 if diff<=-0.7 else 0
        if w:
            weights[key]=w
            details[key]={'positiveSamples':len(one),'negativeSamples':len(zero),'avgReturnDifferencePctPoint':round(diff,2),'proposedBonus':w}
    if not weights:
        result['reason']='十分なサンプルはあるが有意な参考差を見出せない'
        result['featureObservations']=details
        return result
    # Holdout comparisons must use all same-week candidate results with complete prices.
    gains=[]
    for r in holdout:
        entries=[s for s in r['universeResults'] if s.get('returnPct') is not None]
        if len(entries)<5:continue
        base=sorted(entries,key=lambda s:(-float(s.get('selectionBaseScore') or 0),s['code']))[:5]
        adapt=sorted(entries,key=lambda s:(-max(0,min(100,float(s.get('selectionBaseScore') or 0)+max(-6,min(6,sum(weights.get(k,0) for k,v in (s.get('features') or {}).items() if v is True))))),s['code']))[:5]
        gains.append(sum(s['returnPct'] for s in adapt)/5-sum(s['returnPct'] for s in base)/5)
    result['featureObservations']=details
    result['holdoutWeeks']=len(gains)
    result['holdoutAverageAdvantagePt']=round(sum(gains)/len(gains),3) if gains else None
    if len(gains)<4 or sum(gains)/len(gains)<0.3:
        result['reason']='直近4週の時間外検証で十分な改善が確認できないため自動適用しない'
        return result
    result.update(status='active',weights=weights,
                  appliedChanges=[{'feature':k,'scoreDelta':v} for k,v in weights.items()],
                  reason='時間順4週の検証に合格した補助スコアのみ自動適用（成績保証ではない）')
    return result


def learn():
    idx=read_json(INDEX,{'reports':[]})
    reports=[]
    for x in idx['reports']:
        f=ROOT/x['reportPath']
        if f.exists():reports.append(read_json(f,{}))
    latest=calculate_learning(reports)
    previous=read_json(LEARNING,{})
    # Preserve prior active weights unless a new validated config supersedes it.
    # Never silently discard live weights due to temporary data incompleteness.
    if previous.get('status')=='active' and latest.get('status')!='active':
        latest['status']='active'
        latest['weights']=previous['weights']
        latest['appliedChanges']=previous.get('appliedChanges',[])
        latest['reason']='直近の新規学習が未合格のため、以前検証済みの補正を据え置き'
    if latest.get('status')=='active':
        latest['effectiveFrom']=previous.get('effectiveFrom') if latest.get('reason','').startswith('直近の新規学習が未合格') else datetime.now(JST).date().isoformat()
    latest['generatedAt']=datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')
    write_json(LEARNING,latest)
    return latest


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','finalize','learn','all'],required=True)
    a=p.parse_args()
    if a.mode=='freeze':
        d=read_json(ROOT/'data.json',{})
        if not d:raise RuntimeError('data.json not found')
        s=freeze(d);print('[OK] Frozen',s['asOf'],len(s['pickedCodes']),'research candidates')
    elif a.mode=='finalize':
        rows=finalize();print('[OK] Reviewed',len(rows),'weekly snapshots')
    elif a.mode=='learn':
        r=learn();print('[INFO] Learning',r['status'],r['reason'])
    else:
        rows=finalize();r=learn();print('[OK] Review rows',len(rows),'Learning:',r['status'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[ERROR] Review and learning not published:',e,file=sys.stderr)
        sys.exit(1)
