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
REPORT_DIR = ROOT / 'reports'
INDEX = REPORT_DIR / 'index.json'
LEARNING = ROOT / 'learning.json'
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


def snapshot_stock(s, learn):
    out={k:s.get(k) for k in ('code','name','market','close','gainPct','volumeDay','turnoverThousand','ma5','ma25','ma75','ma100','ma200','ma250','ma300','rsi','volumeRatio','creditBuy','creditSell','creditDelta','shortDelta','earnings','sector','priceAsOf') if k in s}
    out['baseScore']=base_score(s)
    out['decisionScore']=adjusted_score(s,learn)
    out['features']=features(s)
    return out


def freeze(data, today=None, legacy=False):
    """One snapshot per asOf; never rewrite after a future result is known."""
    today=today or datetime.now(JST).date().isoformat()
    asof=data['asOf']
    mon,fri=week_dates(asof)
    store=read_json(PICKS, {'version':1,'weeks':[]})
    existing=next((x for x in store['weeks'] if x['asOf']==asof),None)
    if existing:return existing
    if not legacy and asof!=today:
        raise ValueError('Cannot call this a pretrade selection unless ranking date is today')
    if not legacy and datetime.now(JST).date().isoformat()==today and datetime.now(JST).time()<time(15,30):
        raise ValueError('Do not lock picks before Friday close')
    if not legacy and data.get('sourceStatus')=='historical-reference':
        raise ValueError('Historical sample may only be saved as retrospective')
    learn=read_json(LEARNING, {'status':'insufficient_data','weights':{}})
    universe=[snapshot_stock(s,learn) for s in data['candidates'] if eligible(s)]
    # Known earnings in the holding week are excluded; unknown is NOT verified as safe.
    universe=[s for s in universe if not (s.get('earnings') and mon<=s['earnings']<=fri)]
    universe.sort(key=lambda s:(-s['decisionScore'],-(s.get('turnoverThousand') or 0),s['code']))
    if legacy:
        # The initial v6.x app shipped with these five default *research* picks.
        default_codes=['7360','6262','3399','7384','7309']
        picked=[s['code'] for code in default_codes for s in universe if s['code']==code]
        mode='retrospective_initial_reference'
    else:
        picked=[s['code'] for s in universe[:5]]
        mode='pretrade_locked'
    out={'asOf':asof,'tradeMonday':mon,'tradeFriday':fri,'selectionMode':mode,
         'snapshotDate':today,'dataSource':data.get('source',''),
         'learningVersion':learn.get('version',0),'pickedCodes':picked,
         'universe':universe[:25],
         'limitations':'決算/信用残が未確認なら調査候補であり買い推奨ではない。現物100株の紙上比較。'}
    store['weeks'].append(out)
    store['weeks'].sort(key=lambda x:x['asOf'])
    write_json(PICKS,store)
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
    if valid and ret<0:notes.append('損失：要因を事後断定せず、チャート・需給と照合する')
    if valid and ret>0:notes.append('利益：偶然の可能性もあり同一条件の再現を確認')
    return {'code':code,'name':s.get('name',''),'picked':picked,
            'mondayOpen':a,'fridayClose':b,'grossProfit100':p,'returnPct':ret,
            'status':('勝ち' if p>0 else '負け' if p<0 else '引分') if valid else '未確定',
            'source':source,'selectionBaseScore':s.get('baseScore'),
            'selectionDecisionScore':s.get('decisionScore'),'features':f,
            'selectionRsi':s.get('rsi'),'selectionVolumeRatio':s.get('volumeRatio'),
            'reviewPoints':notes}


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
    REPORT_DIR.mkdir(exist_ok=True)
    rows=[]
    for wk in history['weeks']:
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
