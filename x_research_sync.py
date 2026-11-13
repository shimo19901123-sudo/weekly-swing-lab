#!/usr/bin/env python3
"""Read-only X API v2 research feed for WEEKLY SWING LAB.

X_BEARER_TOKEN must be a GitHub Actions secret; NEVER put it in index.html.
Stores post IDs, URLs and *keyword-derived research checklists only*, not
complete post text. No scoring, auto-trading, or editing of existing stock data.

A locked account is skipped and its previously displayed entries are removed.
This script never scrapes the X website or bypasses account access controls.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FEED = ROOT / 'x_research_feed.json'
ACCOUNTS = [
    ('keith_market', 'Keith'),
    ('harukun1717', 'はるくん'),
    ('cSWN3RHEZM17282', '自分用'),
    ('toushi_kenshou', 'ぽこたん'),
    ('Michael_MIRAI_', '未来手法参考'),
]
# Keywords identify discussion topics, not sentiments or recommendations.
TOPICS = (
    ('出来高・需給', r'出来高|売買代金|信用|買残|売残|貸借|空売り|需給|ショートカバー|short.?cover|short.?squeeze|volume|liquidity',
     '選定基準日時点の出来高・信用残・終値を揃え、翌週の結果と比較する。'),
    ('押し目・酒田', r'押し目|下ヒゲ|上ヒゲ|上髭|下髭|上げ三法|赤三兵|捨て子|ローソク|窓開|ギャップ|三尊|ネックライン|candlestick|pullback|breakout',
     '確定済み日足だけで形状と安値維持を照合し、騙しと翌週の損益を記録する。'),
    ('移動平均・テクニカル', r'移動平均|MA\s?\d|RSI|MACD|ボリンジャ|乖離率|トレンド|テクニカル|moving average|bollinger|momentum',
     '移動平均・RSI・値幅・過熱感を事前の確定足で計算し、後から補正しない。'),
    ('金利・景気・地合い', r'利上げ|利下げ|日銀|FRB|FOMC|CPI|PPI|金利|為替|ドル円|VIX|日経VI|SQ|先物|景気|inflation|fed|treasury|rates',
     '公表日時と市場予想・金利・指数の変化を区別し、週末リスクを記録する。'),
    ('半導体・テーマ', r'半導体|AI|データセンター|SOX|SOXX|NVDA|AMD|半導体製造|電力|インフラ|semiconductor|NVIDIA|memory|chip|data.?center',
     '業種と個別株の値動きを分け、期待先行と業績・空売り買戻しを区別する。'),
    ('業績・財務・配当', r'配当|利回り|PER|PBR|CAGR|増配|減配|時価総額|業績|決算|上方修正|優待|earnings|dividend|valuation',
     '企業IR・決算予定・配当予想を確認し、週次売買の利益率と長期配当を混同しない。'),
    ('資金管理・売買計画', r'損切|利確|利益確定|目標価格|買値|買い場|エントリー|リスクリワード|JC|ダービー|stop.?loss|take.?profit|risk.?reward',
     '利幅・損切り・月曜寄付ギャップを事前に固定し、金曜引けの結果と比較する。'),
)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def read_feed():
    try:
        old = json.loads(FEED.read_text(encoding='utf-8'))
        return old if isinstance(old, dict) else {}
    except (OSError, ValueError):
        return {}


def write_feed(feed):
    out = json.dumps(feed, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if not FEED.exists() or FEED.read_text(encoding='utf-8') != out:
        FEED.write_text(out, encoding='utf-8')


def request_json(path, token, params=None, opener=urlopen):
    url = 'https://api.x.com/2/' + path.lstrip('/')
    if params:
        url += '?' + urlencode(params)
    req = Request(url, headers={
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/json',
        'User-Agent': 'WEEKLY-SWING-LAB-ResearchFeed/1.0',
    })
    try:
        with opener(req, timeout=15) as resp:
            return json.loads(resp.read(400000).decode('utf-8'))
    except HTTPError as e:
        # Avoid leaking any response-body contents or authentication information.
        raise RuntimeError('X_API_HTTP_' + str(e.code)) from None
    except URLError:
        raise RuntimeError('X_API_NETWORK_ERROR') from None


def classify(text):
    tags, plans = [], []
    for name, pattern, plan in TOPICS:
        if re.search(pattern, text or '', flags=re.I):
            tags.append(name)
            plans.append(plan)
    return tags[:4], plans[:2]


def make_post(post, handle, label):
    pid = str(post.get('id', ''))
    if not pid.isdigit() or len(pid) > 30:
        return None
    tags, plans = classify(str(post.get('text', '')))
    return {
        'id': pid,
        'handle': handle,
        'account': label,
        'publishedAt': str(post.get('created_at') or ''),
        'url': 'https://x.com/' + handle + '/status/' + pid,
        'topics': tags or ['未分類・原文確認'],
        'checkpoints': plans or ['原文を読んでから投資の話題か確認する。'],
        'reviewStatus': 'キーワードによる仮分類／手法の有効性は未検証',
    }


def run(token, requester=request_json):
    old = read_feed()
    saved = [p for p in old.get('posts', []) if isinstance(p, dict) and p.get('id')]
    old_highwater = old.get('latestIds', {}) if isinstance(old.get('latestIds'), dict) else {}
    now = timestamp()
    feed = {
        'version': 1,
        'status': 'not_configured' if not token else 'running',
        'lastSuccessAt': old.get('lastSuccessAt'),
        'checkedAt': None,
        'accounts': [],
        'latestIds': dict(old_highwater),
        'posts': saved[:150],
        'disclaimer': 'X公式APIで確認できた公開投稿のID・リンクと、キーワードで推定した研究テーマだけを保存。投稿の賛否・意図・投資成績は判定できません。銘柄選定スコアは変更しません。',
    }
    if not token:
        for handle, label in ACCOUNTS:
            feed['accounts'].append({'handle': handle, 'label': label, 'status': 'api_not_configured', 'newCount': 0})
        write_feed(feed)
        print('[INFO] X_BEARER_TOKEN not configured; safe inactive feed generated.')
        return feed

    try:
        lookup = requester('users/by', token, {'usernames': ','.join(h for h, _ in ACCOUNTS), 'user.fields': 'protected'})
    except Exception as exc:
        feed['status'] = 'api_error'
        feed['checkedAt'] = now
        feed['accounts'] = [{'handle': h, 'label': label, 'status': 'lookup_failed', 'newCount': 0} for h, label in ACCOUNTS]
        write_feed(feed)
        print('[WARN] X profile lookup failed: ' + type(exc).__name__, file=sys.stderr)
        return feed

    found = {str(u.get('username', '')).lower(): u for u in lookup.get('data', []) if isinstance(u, dict)}
    posts_by_id = {str(p['id']): p for p in saved}
    ok = 0
    for handle, label in ACCOUNTS:
        user = found.get(handle.lower())
        info = {'handle': handle, 'label': label, 'status': 'not_found_or_unavailable', 'newCount': 0}
        if not user:
            feed['accounts'].append(info)
            continue
        if user.get('protected'):
            info['status'] = 'protected_skipped'
            # Do not retain previously saved posts from accounts which later lock.
            posts_by_id = {k: v for k, v in posts_by_id.items() if v.get('handle', '').lower() != handle.lower()}
            feed['latestIds'].pop(handle, None)
            feed['accounts'].append(info)
            continue
        uid = str(user.get('id', ''))
        if not uid.isdigit():
            feed['accounts'].append(info)
            continue
        params = {'max_results': 5, 'tweet.fields': 'created_at,lang', 'exclude': 'retweets,replies'}
        if str(old_highwater.get(handle, '')).isdigit():
            params['since_id'] = str(old_highwater[handle])
        try:
            result = requester('users/' + uid + '/tweets', token, params)
        except Exception as exc:
            info['status'] = 'timeline_failed'
            info['errorType'] = type(exc).__name__
            feed['accounts'].append(info)
            continue
        data = result.get('data') or []
        high = int(old_highwater.get(handle, '0')) if str(old_highwater.get(handle, '0')).isdigit() else 0
        count = 0
        for post in data:
            p = make_post(post, handle, label)
            if not p:
                continue
            high = max(high, int(p['id']))
            if p['id'] not in posts_by_id:
                count += 1
            posts_by_id[p['id']] = p
        if high:
            feed['latestIds'][handle] = str(high)
        info.update(status='public_checked', newCount=count)
        if result.get('meta', {}).get('next_token'):
            info['status'] = 'public_checked_more_unfetched'
        feed['accounts'].append(info)
        ok += 1
    feed['posts'] = sorted(posts_by_id.values(), key=lambda p: (p.get('publishedAt', ''), p.get('id', '')), reverse=True)[:150]
    feed['checkedAt'] = now
    if ok:
        feed['lastSuccessAt'] = now
    feed['status'] = 'ok' if ok == len(ACCOUNTS) else 'partial' if ok else 'api_error'
    write_feed(feed)
    print(f'[INFO] X public accounts checked {ok}/{len(ACCOUNTS)}, records retained {len(feed["posts"])}')
    return feed


def main():
    return run(os.environ.get('X_BEARER_TOKEN', '').strip())


if __name__ == '__main__':
    main()
