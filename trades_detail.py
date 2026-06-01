"""每笔交易详细记录"""
import sys, sqlite3, pandas as pd, numpy as np
from datetime import datetime

SL, TP = 0.015, 0.015
FEE, SLIP = 0.0005, 0.0005
COINS = ['BTC/USDT:USDT','ETH/USDT:USDT','SOL/USDT:USDT','DOGE/USDT:USDT',
         'XRP/USDT:USDT','ADA/USDT:USDT','AVAX/USDT:USDT','DOT/USDT:USDT',
         'LINK/USDT:USDT','UNI/USDT:USDT','ARB/USDT:USDT','OP/USDT:USDT',
         'SUI/USDT:USDT','APT/USDT:USDT','NEAR/USDT:USDT']

conn = sqlite3.connect('data/trading.db')
df_all = pd.read_sql_query('SELECT * FROM klines WHERE timeframe="15m" ORDER BY timestamp ASC', conn)
conn.close()
df_all = df_all[df_all['symbol'].isin(COINS)]

def resample_4h(df):
    t = df[['timestamp','open','high','low','close','volume']].copy()
    t.index = pd.to_datetime(t['timestamp'], unit='ms')
    a = t.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    a['timestamp'] = (a['timestamp'].astype('int64')//1_000_000).astype(int)
    return a

def trend_map(df4h):
    c = df4h['close']; e = c.ewm(span=55,adjust=False).mean()
    m = {}
    for i in range(len(df4h)):
        p, em = c.iloc[i], e.iloc[i]
        m[df4h['timestamp'].iloc[i]] = 'bullish' if p>em*1.01 else ('bearish' if p<em*0.99 else 'neutral')
    return m

def get_trend(m, ts):
    for k in sorted(m.keys(), reverse=True):
        if k <= ts:
            return m[k]
    return 'neutral'

start = pd.Timestamp('2026-05-24')
end = pd.Timestamp('2026-05-31')
start_ts = int(start.timestamp()*1000)
end_ts = int(end.timestamp()*1000)
early = start - pd.Timedelta(days=15)
early_ts = int(early.timestamp()*1000)

all_trades = []

for symbol in COINS:
    cdf = df_all[df_all['symbol']==symbol].copy()
    if cdf.empty:
        continue
    fd = cdf[cdf['timestamp']>=early_ts][['timestamp','open','high','low','close','volume']].copy()
    fd = fd.sort_values('timestamp').reset_index(drop=True)
    if len(fd)<100:
        continue
    d4h = resample_4h(fd)
    if len(d4h)<55:
        continue
    tm = trend_map(d4h)
    td = fd[(fd['timestamp']>=start_ts)&(fd['timestamp']<=end_ts)].copy()
    td = td.reset_index(drop=True)
    cl = td['close']; delta = cl.diff()
    g = delta.where(delta>0,0).rolling(14).mean()
    l = (-delta.where(delta<0,0)).rolling(14).mean()
    td['rsi'] = 100-(100/(1+g/l))
    td['vr'] = td['volume']/td['volume'].rolling(20).mean()
    td = td.dropna(subset=['rsi','vr']).reset_index(drop=True)
    cap = 10.0; pos = None
    name = symbol.split('/')[0]
    for i in range(len(td)):
        r = td.iloc[i]
        ts = r['timestamp']; px = float(r['close']); rsi = float(r['rsi']); vr = float(r['vr'])
        trend = get_trend(tm, ts)
        dt = datetime.utcfromtimestamp(ts/1000).strftime('%m-%d %H:%M')
        if pos:
            ep = pos['ep']; d = pos['d']
            pp = (px-ep)/ep if d=='L' else (ep-px)/ep
            if pp <= -SL:
                pnl = pos['sz']*(pp-FEE-SLIP)
                cap += pos['sz']+pnl
                all_trades.append({'s':name,'dir':d,'et':pos['et'],'ep':ep,'xt':dt,'xp':px,'pnl':pnl,'pp':pp-FEE-SLIP,'res':'SL','sz':pos['sz']})
                pos = None
            elif pp >= TP:
                pnl = pos['sz']*(pp-FEE-SLIP)
                cap += pos['sz']+pnl
                all_trades.append({'s':name,'dir':d,'et':pos['et'],'ep':ep,'xt':dt,'xp':px,'pnl':pnl,'pp':pp-FEE-SLIP,'res':'TP','sz':pos['sz']})
                pos = None
        if pos is None and cap > 0.1:
            sig = None
            if trend in ('bullish','neutral') and rsi<40 and vr>1.0:
                sig = 'L'
            elif trend in ('bearish','neutral') and rsi>60 and vr>1.0:
                sig = 'S'
            if sig:
                sz = min(cap*0.95, cap)
                if sz > 0.1:
                    pos = {'d':sig,'ep':px,'sz':sz,'et':dt}
                    cap -= sz

hdr = "%4s %4s %12s %3s %10s %12s %10s %3s %8s %8s %8s %9s" % ('#','币种','入场时间','方向','入场价','出场时间','出场价','结果','仓位','盈亏%','盈亏U','累计U')
print(hdr)
print('-'*105)

cum = 0
for i, t in enumerate(all_trades, 1):
    cum += t['pnl']
    d = '多' if t['dir']=='L' else '空'
    res = '止盈' if t['res']=='TP' else '止损'
    s1 = '+' if t['pnl']>0 else ''
    s2 = '+' if cum>0 else ''
    line = "%4d %4s %12s %3s %10.4f %12s %10.4f %3s %8.4f %s%7.2f%% %s%7.4f %s%8.4f" % (
        i, t['s'], t['et'], d, t['ep'], t['xt'], t['xp'], res, t['sz'],
        s1, t['pp']*100, s1, t['pnl'], s2, cum)
    print(line)

print('-'*105)
w = sum(1 for t in all_trades if t['pnl']>0)
l = len(all_trades)-w
wp = sum(t['pnl'] for t in all_trades if t['pnl']>0)
lp = sum(t['pnl'] for t in all_trades if t['pnl']<=0)
print("")
print("交易统计:")
print("  总交易: %d笔" % len(all_trades))
print("  盈利: %d笔 (+%.4fU)" % (w, wp))
print("  亏损: %d笔 (%.4fU)" % (l, lp))
print("  净盈亏: %+.4fU" % cum)
print("  胜率: %.1f%%" % (w/len(all_trades)*100))
print("  盈亏比: %.2f" % (abs(wp/lp) if lp != 0 else 0))
