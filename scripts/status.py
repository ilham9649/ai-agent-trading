#!/usr/bin/env python3
"""Print the paper-trader status + P&L. Run on the VPS: .venv/bin/python scripts/status.py"""
import json
import sqlite3

st = json.load(open("data/portfolio_state.json"))
con = sqlite3.connect("data/portfolio_paper.sqlite")


def latest(kind):
    row = con.execute("select payload from events where kind=? order by id desc limit 1", (kind,)).fetchone()
    return json.loads(row[0]) if row else None


obs = latest("observation") or {}
ks = latest("killswitch")
hb = latest("heartbeat")
eq = obs.get("equity", st["cash"])
pnl = (eq - 500) / 500 * 100
print(f"invested={st['invested']}  cash=${st['cash']:.0f}  equity=${eq:.2f}  P&L={pnl:+.2f}%")
print(f"basket_dd={obs.get('basket_dd')}  trades_last_cycle={obs.get('n_trades')}")
if ks:
    print(f"last went-flat: {ks.get('reason')}")
if hb:
    print(f"last re-entry: {hb.get('note')}")
