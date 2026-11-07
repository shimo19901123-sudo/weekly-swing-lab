"""Non-authoritative WEEKLY SWING LAB public weekend headline monitor.

Runs on Sunday after Friday ranking has been saved. Candidate headlines are
only precautionary alerts. Sources are syndicated Google News RSS; a missing
headline or successful request is NEVER equivalent to confirmation of safety.
Never edit market prices, Friday ranking, or prior frozen snapshots.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime, timedelta, time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
JST=ZoneInfo('Asia/Tokyo')
DATA=ROOT/'data.json'
NEWS=ROOT/'weekend_news.json'
SOURCE='Google ニュース RSS（新聞記事の見出し検索・一次IR照合なし）'
RISK_TERMS=('下方修正','上場廃止','不正','業績悪化','会計問題','破綻','民事再生','取引停止','決算延期',
            '増資','売出し','希薄化','赤字転落','業績予想引き下げ','サイバー攻撃','行政処分','株主訴訟',
            '操業停止','重大事故','格下げ','粉飾','監理銘柄','整理銘柄')
MARKET_QUERIES=('日本株 米国株 週末 市場 金利 為替','日本企業 上場株式 決算 下方修正 週末',
                '日経平均 外国為替 米国雇用統計 FRB 日銀')

def eligible(s):
    try:
        return (5<=float(s['gainPct'])<=12 and float(s['volumeDay'])>=10000
                and float(s['close'])>=250 and float(s['turnoverThousand'])>=50000)
    except (KeyError,ValueError,TypeError):return False

def fetch_rss(query,fetcher=None):
    url='https://news.google.com/rss/search?q='+quote(query+' when:3d')+'&hl=ja&gl=JP&ceid=JP:ja'
    if fetcher is None:
        def fetcher(url):
            req=Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; SwingLabResearch/1.0)'})
            with urlopen(req,timeout=8) as r:return r.read(450000)
    root=ET.fromstring(fetcher(url))
    out=[]
    for e in root.findall('.//item')[:35]:
        title=' '.join((e.findtext('title') or '').split())
        link=(e.findtext('link') or '').strip()
        source=e.findtext('source') or ''
        published=e.findtext('pubDate') or ''
        try:
            parsed=parsedate_to_datetime(published).astimezone(JST)
            dt=parsed.isoformat(timespec='minutes')
        except Exception:dt=''
        if not title or not link or not link.startswith('https://') or not dt:continue
        out.append({'title':title[:180],'url':link[:1200], 'publisher':source[:90], 'publishedAt':dt})
    return out

def collect(data,now=None,fetcher=None,max_codes=15):
    now=now or datetime.now(JST)
    if now.tzinfo is None:raise ValueError('now must be timezone aware')
    if now.weekday()!=6:raise ValueError('週末ニュース確認は日本時間の日曜日に実行してください')
    asof=data.get('asOf','')
    expected=(now.date()-timedelta(days=2)).isoformat()
    if asof!=expected or data.get('sourceStatus')=='historical-reference':
        raise ValueError('金曜日の新しいランキングが未取得。古い候補で週末選定してはいけません')
    cutoff=datetime.combine(now.date()-timedelta(days=2),time(15,30),JST)
    candidates=[s for s in data.get('candidates',[]) if eligible(s)]
    # Match the Sunday snapshot's base-score ordering as closely as possible.
    from weekly_review import base_score
    candidates=sorted(candidates,key=lambda s:(-base_score(s),-float(s.get('turnoverThousand') or 0),s['code']))[:max_codes]
    global_articles=[]
    perstock={}
    failed=[]; success=0
    def relevant(rows):return [a for a in rows if cutoff<=datetime.fromisoformat(a['publishedAt'])<=now]
    for query in MARKET_QUERIES:
        try:
            global_articles.extend(relevant(fetch_rss(query,fetcher)))
            success+=1
        except Exception as ex: failed.append('市場トピック: '+str(ex)[:110])
    for s in candidates:
        code=str(s['code']);name=str(s.get('name') or '').strip()
        if not name:continue
        try:
            rows=relevant(fetch_rss('"'+name+'" 株式',fetcher))
            # Require name in the headline, not just a result of a broad query.
            rows=[a for a in rows if name.lower() in a['title'].lower()]
            unique={a['title']:a for a in rows}
            articles=list(unique.values())[:8]
            hits=[a for a in articles if any(w in a['title'] for w in RISK_TERMS)]
            perstock[code]={'name':name,'retrieval':'headlines_only',
                            'headlines':articles[:6],'riskHeadlines':hits[:4],
                            'needsReview':bool(hits),
                            'note':'見出しヒットは公式IR未確認。ヒット0件でも材料なしと証明できない。'}
            success+=1
        except Exception as ex:
            perstock[code]={'name':name,'retrieval':'unavailable','headlines':[],
                            'riskHeadlines':[],'needsReview':False,
                            'note':'取得失敗。週末ニュースの有無は未確認。'}
            failed.append(f'{code}: {str(ex)[:100]}')
    unique={a['title']:a for a in global_articles}
    global_articles=sorted(unique.values(),key=lambda x:x['publishedAt'],reverse=True)[:18]
    complete=not failed and len(perstock)==len(candidates)
    return {'version':1,'rankFriday':asof,'selectionSunday':now.date().isoformat(),
            'collectedAt':now.isoformat(timespec='minutes'),
            'status':'headlines_only' if complete else 'partial' if success else 'unavailable',
            'source':SOURCE,'globalHeadlines':global_articles,
            'companies':perstock,'failures':failed[:20],
            'coverage':{'requestedCompanies':len(candidates),'companiesProcessed':len(perstock),
                        'successQueries':success,'failedQueries':len(failed)},
            'limitations':['週末のすべての材料・公式IRを検証したものではない。',
                           '見出しの重大語は誤検出・情報遅延の可能性があるため売買を自動確定しない。',
                           '見出しの有無のみでは安全性を確認できない。日曜夜と月曜寄付前に一次情報を要確認。']}

def write_json(path,obj):
    body=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.write_text(body,encoding='utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['sunday'],default='sunday')
    args=p.parse_args()
    d=json.loads(DATA.read_text(encoding='utf-8'))
    result=collect(d)
    write_json(NEWS,result)
    print('[INFO] Weekend scan',result['rankFriday'],result['status'],
          'company coverage',result['coverage']['companiesProcessed'],
          'risk headline companies',sum(x['needsReview'] for x in result['companies'].values()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[ERROR] Weekend scan could not be published:',e,file=sys.stderr)
        sys.exit(1)
