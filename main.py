"""TightTrail — BTC race bot for make.fun btc-race-24h-20261117-0603 (account 3).

Long only, one open position, full cash per trade. Two setups, both longs:

  Setup B — capitulation bounce. A deep washout (close down >= 550 bps from
  the 720-bar high) plus a long-lower-wick bar in the last 20 bars, a
  reversal bar closing up >= 60 bps and above the previous bar's high, RSI
  turning up, and at least 2:1 reward-to-risk back to the 720-bar high.
  The stop sits below the washout low, clamped to 50-100 bps under entry.

  Setup D — sharp-drop dip-buy with a TIGHT exit. This is the account-3
  edge: the dip must be a fast fall (>= 120 bps off the 120-bar high AND
  >= 150 bps over the last 60 bars), RSI under 50, current bar closing up,
  and the drop from the 720-bar high under 550 bps (a real capitulation
  belongs to Setup B). The exit is tight on purpose: a 60 bps hard stop
  and a trailing stop that arms at +80 bps and trails the peak by 80 bps,
  so a dip that keeps sliding is cut quickly and a winner is banked early
  instead of being ridden back down. At most one dip trade per window.

Guards: at most 3 entries per window (1 dip + 2 bounce); stop after 2
consecutive BOUNCE losses only (a dip loss never counts, a dip win never
resets); stop entering if equity falls under 98% of start; no entries
after bar 1320; 60-bar cooldown after an exit; max slippage 200 bps;
at least 200 closes of history before any entry. No setup, no trade.

Deterministic: no clock, no network, no files. Every number comes from
the bars the harness sends. One process per run, state starts fresh.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
BPS = 10_000

# --- history and indicators ---
MIN_HISTORY_BARS = 200      # closes needed before any entry
RSI_PERIOD = 14

# --- Setup B (capitulation bounce) ---
BOUNCE_DROP60_BPS = 150     # or >= 250 bps over 120 bars (prev bar)
BOUNCE_DROP120_BPS = 250
BOUNCE_HIGH_LOOKBACK = 720  # 12h high for the capitulation measure
BOUNCE_HIGH_BPS = 550
BOUNCE_CONFIRM_BPS = 60     # reversal bar closes up >= 60 bps
WICK_MIN_BPS = 20           # wick bar range >= 20 bps
WICK_TOP_PCT = 40           # close inside the top 40% of the bar range
WICK_LOOKBACK = 20
STOP_MIN_BPS = 50           # stop clamped to 50-100 bps below entry
STOP_MAX_BPS = 100
STOP_BELOW_LOW_BPS = 20     # stop sits 20 bps below the washout low
BOUNCE_ARM_BPS = 80         # trailing arms at +80 bps
BOUNCE_TRAIL_BPS = 80       # then trails the peak by 80 bps
BOUNCE_MAX_HOLD = 300
MAX_BOUNCE_TRADES = 2

# --- Setup D (sharp-drop dip-buy, tight exit) ---
DIP_HIGH_LOOKBACK = 120
DIP_DROP_BPS = 120          # >= 120 bps off the 120-bar high (prev bar)
DIP_RSI_MAX = 50
DIP_DROP60_BPS = 150        # AND >= 150 bps over the last 60 bars: fast fall
DIP_STOP_BPS = 60           # hard stop 60 bps below entry (tight)
DIP_ARM_BPS = 80            # trailing arms at +80 bps
DIP_TRAIL_BPS = 80          # then trails the peak by 80 bps
DIP_MAX_HOLD = 300
DIP_MAX_TRADES = 1

# --- guards ---
MAX_TRADES = 3
MAX_CONSEC_LOSSES = 2       # bounce losses only
MAX_DAILY_LOSS_BPS = 200    # stop entering under 98% of start equity
LAST_ENTRY_BAR = 1320
COOLDOWN_BARS = 60
MAX_SLIPPAGE_BPS = 200
WIN_COST_BPS = 26           # round-trip cost, for win/loss calls

# --- per-run state (a fresh process per run, so plain module state is fine) ---
closes = []                 # raw close ints, in bar order
highs = []                  # raw high ints
lows = []                   # raw low ints
rsi_vals = []               # RSI(14) per close, from the 15th close on
_avg_gain = None            # Wilder smoothing state for RSI
_avg_loss = None
entry = None                # dict once a buy is decided; None when flat
confirmed = False           # True once the position shows in state.positions
trades_done = 0
bounce_done = 0
dip_done = 0
consec_losses = 0
cooldown_until = -1


def _rsi_from_avgs():
    """RSI from the current Wilder averages (floats, deterministic)."""
    if _avg_loss == 0:
        return 100.0
    if _avg_gain == 0:
        return 0.0
    rs = _avg_gain / _avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _rsi_update():
    """Fold the newest close into Wilder RSI(14); append one value per bar
    once 14 changes exist (so rsi_vals[-1] always matches closes[-1])."""
    global _avg_gain, _avg_loss
    n = len(closes)
    if n < 2:
        return
    if _avg_gain is None:
        if n - 1 < RSI_PERIOD:
            return
        gains = 0
        losses = 0
        for i in range(1, RSI_PERIOD + 1):
            d = closes[i] - closes[i - 1]
            if d > 0:
                gains += d
            else:
                losses += -d
        _avg_gain = gains / RSI_PERIOD
        _avg_loss = losses / RSI_PERIOD
        rsi_vals.append(_rsi_from_avgs())
        return
    chg = closes[-1] - closes[-2]
    gain = chg if chg > 0 else 0
    loss = -chg if chg < 0 else 0
    _avg_gain = (_avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
    _avg_loss = (_avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD
    rsi_vals.append(_rsi_from_avgs())


def _drop_bps(ref_close, past_close):
    """How far past_close fell to reach ref_close, in bps (int, truncated)."""
    if past_close <= 0:
        return 0
    return (past_close - ref_close) * BPS // past_close


def _high_before(lookback):
    """Highest high of the `lookback` bars ending at the previous bar.
    Uses whatever history exists when the window is shorter."""
    end = len(highs) - 2
    if end < 0:
        return None
    start = max(0, end - lookback + 1)
    return max(highs[start:end + 1])


def _low_before(lookback):
    """Lowest low of the `lookback` bars ending at the previous bar."""
    end = len(lows) - 2
    if end < 0:
        return None
    start = max(0, end - lookback + 1)
    return min(lows[start:end + 1])


def _has_wick():
    """True if a long-lower-wick bar appeared in the last WICK_LOOKBACK bars:
    range >= 20 bps and the close inside the top 40% of the range."""
    n = len(closes)
    start = max(0, n - WICK_LOOKBACK)
    for i in range(start, n):
        rng = highs[i] - lows[i]
        if rng <= 0:
            continue
        if rng * BPS // closes[i] < WICK_MIN_BPS:
            continue
        if (highs[i] - closes[i]) * 100 // rng <= WICK_TOP_PCT:
            return True
    return False


def _bounce_setup():
    """Setup B: capitulation bounce. Returns the stop price if it fires
    on the current bar, else None. Drops are measured on the previous bar
    (idx -2); the reversal itself is the current bar."""
    if len(closes) < MIN_HISTORY_BARS:
        return None
    prev = closes[-2]
    cur = closes[-1]
    # (1) a real fall first: >= 150 bps over 60 bars or >= 250 over 120
    d60 = _drop_bps(prev, closes[-62])
    d120 = _drop_bps(prev, closes[-122])
    if d60 < BOUNCE_DROP60_BPS and d120 < BOUNCE_DROP120_BPS:
        return None
    # (2) capitulation: >= 550 bps under the 720-bar high
    hi720 = _high_before(BOUNCE_HIGH_LOOKBACK)
    if hi720 is None or _drop_bps(prev, hi720) < BOUNCE_HIGH_BPS:
        return None
    # (3) a long-lower-wick bar in the last 20 bars
    if not _has_wick():
        return None
    # (4) reversal: current bar up >= 60 bps and above the previous high
    if (cur - prev) * BPS // prev < BOUNCE_CONFIRM_BPS:
        return None
    if cur <= highs[-2]:
        return None
    # (5) RSI(14) turning up
    if len(rsi_vals) < 2 or rsi_vals[-1] <= rsi_vals[-2]:
        return None
    # (6) reward-to-risk: the way back to the 720-bar high is >= 2x the stop
    cap_low = _low_before(BOUNCE_HIGH_LOOKBACK)
    if cap_low is None:
        return None
    raw = (cur - cap_low) + cur * STOP_BELOW_LOW_BPS // BPS
    lo = cur * STOP_MIN_BPS // BPS
    hi = cur * STOP_MAX_BPS // BPS
    dist = max(lo, min(hi, raw))
    if hi720 - cur < 2 * dist:
        return None
    return cur - dist


def _dip_setup():
    """Setup D: sharp-drop dip-buy. True only when every filter passes on
    the current bar. The sharp-drop filter is the point: a fast fall is
    oversold and bounces; a slow drift down is a downtrend and loses."""
    if len(closes) < MIN_HISTORY_BARS:
        return False
    prev = closes[-2]
    cur = closes[-1]
    # (1) >= 120 bps off the 120-bar high, measured on the previous bar
    hi120 = _high_before(DIP_HIGH_LOOKBACK)
    if hi120 is None or _drop_bps(prev, hi120) < DIP_DROP_BPS:
        return False
    # (2) not a capitulation: under 550 bps off the 720-bar high
    hi720 = _high_before(BOUNCE_HIGH_LOOKBACK)
    if hi720 is None or _drop_bps(prev, hi720) >= BOUNCE_HIGH_BPS:
        return False
    # (3) RSI(14) under 50
    if not rsi_vals or rsi_vals[-1] >= DIP_RSI_MAX:
        return False
    # (4) the current bar closes up (any amount)
    if cur <= prev:
        return False
    # (5) sharp-drop filter: fell >= 150 bps over the last 60 bars
    if _drop_bps(prev, closes[-62]) < DIP_DROP60_BPS:
        return False
    return True


def on_batch(state, batch):
    """One reply per batch. Returns a list of orders (possibly empty)."""
    global entry, confirmed, trades_done, bounce_done, dip_done
    global consec_losses, cooldown_until
    if state.warming_up:
        return []
    bar = state.latest.get(SYMBOL)
    if bar is None:
        return []
    c = int(bar["c"])
    h = int(bar["h"])
    l = int(bar["l"])
    closes.append(c)
    highs.append(h)
    lows.append(l)
    _rsi_update()
    seq = state.seq
    holding = state.positions.get(SYMBOL)

    # --- confirm or reject a buy decided on the previous batch ---
    if entry is not None and not confirmed:
        if holding is not None:
            confirmed = True
            entry["bar"] = seq      # hold time counts from the fill bar
            entry["peak"] = c
            trades_done += 1
            if entry["kind"] == "bounce":
                bounce_done += 1
            else:
                dip_done += 1
        elif seq > entry["decided_bar"] + 1:
            entry = None            # the buy never filled; drop it

    # --- manage an open position: tight stop, trailing stop, max hold ---
    if holding is not None and entry is not None and confirmed:
        entry["peak"] = max(entry["peak"], c)
        if entry["kind"] == "dip":
            stop = entry["px"] - entry["px"] * DIP_STOP_BPS // BPS
            if entry["peak"] - entry["px"] >= entry["px"] * DIP_ARM_BPS // BPS:
                stop = max(stop, entry["peak"] - entry["px"] * DIP_TRAIL_BPS // BPS)
            max_hold = DIP_MAX_HOLD
        else:
            stop = entry["stop_px"]
            if entry["peak"] - entry["px"] >= entry["px"] * BOUNCE_ARM_BPS // BPS:
                stop = max(stop, entry["peak"] - entry["px"] * BOUNCE_TRAIL_BPS // BPS)
            max_hold = BOUNCE_MAX_HOLD
        if c <= stop or seq - entry["bar"] >= max_hold:
            ret_bps = (c - entry["px"]) * BPS // entry["px"]
            if entry["kind"] == "bounce":
                # only bounce trades move the consecutive-loss counter
                if ret_bps < WIN_COST_BPS:
                    consec_losses += 1
                else:
                    consec_losses = 0
            cooldown_until = seq + COOLDOWN_BARS
            qty = holding.qty
            entry = None
            confirmed = False
            return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS}]
        return []

    # --- flat: drop any stale entry record ---
    if holding is None and entry is not None and confirmed:
        entry = None
        confirmed = False
    if holding is not None and entry is None:
        # safety net: a position with no recorded entry; manage it as a
        # bounce with a plain 50 bps stop until the next batch takes over
        entry = {"kind": "bounce", "px": c, "peak": c,
                 "stop_px": c - c * STOP_MIN_BPS // BPS,
                 "bar": seq, "decided_bar": seq}
        confirmed = True
        return []

    # --- entries: flat, nothing pending, guards green ---
    if entry is not None:
        return []
    if seq > LAST_ENTRY_BAR:
        return []
    if seq < cooldown_until:
        return []
    if consec_losses >= MAX_CONSEC_LOSSES:
        return []
    if trades_done >= MAX_TRADES:
        return []
    start_eq = int(state.hello["capital"]["quote_units"])
    if state.equity * BPS < start_eq * (BPS - MAX_DAILY_LOSS_BPS):
        return []
    if len(closes) < MIN_HISTORY_BARS:
        return []

    # Setup B first: it owns capitulations. Then Setup D.
    if bounce_done < MAX_BOUNCE_TRADES:
        stop_px = _bounce_setup()
        if stop_px is not None:
            entry = {"kind": "bounce", "px": c, "peak": c, "stop_px": stop_px,
                     "bar": seq, "decided_bar": seq}
            confirmed = False
            return [{"side": "buy", "symbol": SYMBOL,
                     "notional_quote_units": state.cash,
                     "max_slippage_bps": MAX_SLIPPAGE_BPS}]
    if dip_done < DIP_MAX_TRADES and _dip_setup():
        entry = {"kind": "dip", "px": c, "peak": c, "stop_px": None,
                 "bar": seq, "decided_bar": seq}
        confirmed = False
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": state.cash,
                 "max_slippage_bps": MAX_SLIPPAGE_BPS}]
    return []


if __name__ == "__main__":
    makefun_bars.run(on_batch)
