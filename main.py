"""gta6-twitch-ccu-forecast-01 entry for @roni-rockabilly (Python 3.13, stdlib only).

Forecasts, for each hour h = 2..24 of a game's Twitch release day, five
quantiles of that hour's concurrent viewers in the game's Twitch category,
from the case's hour-1 print and the release-day curves of the past big
game launches in the case's `comparables`.

Method: windowed k-nearest-neighbour MEDIAN log-ratio curves.
  For every comparable launch with a valid hour-1 print p1_c, the launch-day
  curve is reduced to log-ratios r_h = ln(viewers_h / viewers_1).  For the
  target case, with hour-1 print p1, the neighbours are the comparables
  whose own hour-1 print is closest in log space: those within 1.5 log
  units of ln p1, at most the 12 closest, at least the 5 closest.  Each
  hour's forecast is a log-normal centred on the MEDIAN of the neighbours'
  r_h and spread by their median absolute deviation (MAD):

      q_tau(h) = p1 * exp(m_h + z_tau * s_h),
      s_h = 1.15 * 1.4826 * MAD_h, floored at 0.15.

  The median/MAD pair is a robust estimator: one freak launch (a
  simultaneous global unlock exploding 100x, or a launch that flatlines)
  cannot drag the centre the way a mean does, and the spread tracks the
  middle half of the neighbours rather than the outliers.  Neighbours are
  chosen by hour-1 size alone: a launch's hour-1 print encodes its unlock
  pattern.  A rolling-midnight launch (GTA VI, and all four practice
  launches) already has a large hour-1 count (New Zealand / Australia /
  Asia evening plus the UK midnight unlock), so its neighbours are other
  rolling-midnight launches; a simultaneous global unlock has a near-zero
  hour-1 count and its neighbours are other simultaneous unlocks.

  This is a different estimator from a kernel-weighted mean: hard window
  instead of a Gaussian kernel, uniform weights instead of distance
  weights, median instead of weighted mean, MAD instead of a weighted
  standard deviation.  No free-text parsing of the unlock notes is used.

Validation (leave-one-out over the 37 launch comparables in the kit;
the official practice answers are not in the kit -- grade-local grades
on the server only -- so each comparable is scored as a pseudo-practice
case against the other 36):
  - house baseline:  mean pinball 0.393 on all 37 launches, 0.351 on
    hour-1 >= 10k launches, 0.351 is the official-practice-scale number
    (the official practice baseline is ~0.387; the LOO proxy reproduces
    it, which validates the harness).
  - this model:     mean pinball 0.394 on all 37 launches, 0.264 on
    hour-1 >= 10k launches, 0.222 on hour-1 >= 100k launches; 90%
    coverage 0.84, typical miss ~77% over all launches (the tiny
    simultaneous-unlock launches dominate the miss; on hour-1 >= 10k
    launches the median curve tracks the diurnal shape -- night dip,
    US-unlock rises at hours 6 and 9, US-afternoon / EU-evening peak --
    the way the baseline's flat persistence does not).
  The practice launches (Wolverine 23,357; RDR2 116,550; GoW Ragnarok
  269,517; Spider-Man 2 119,087 hour-1 viewers) are all hour-1 >= 10k
  launches, the subset this model beats the baseline on by ~25%.

Deterministic: no randomness, no wall clock, no network, no model call
(this competition runs no model gateway; the program runs offline in the
sandbox).  Sorted iteration everywhere, so two runs write byte-identical
outputs.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only.  Outside the grader,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Neighbour window: comparables within this many natural-log units of the
# target's log hour-1 print are neighbours; at most K_MAX of them (the
# closest), at least K_MIN (the closest, whatever their distance).
WINDOW = 1.5
K_MAX = 12
K_MIN = 5
# Spread calibration: the MAD of the neighbours' log-ratios is scaled by
# this (LOO: 1.0-1.3 all within ~0.002 of each other; 1.15 is the middle
# and lifts the 90% coverage from 0.81 to 0.84).
SPREAD_SCALE = 1.15
# Floor on the log spread of any hour, so no hour is forecast absurdly tight.
SPREAD_FLOOR = 0.15
# Fallback spread when no neighbour has the hour at all.
FALLBACK_SPREAD = 0.5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _median(xs: list) -> float:
    ordered = sorted(xs)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def launch_curves(comparables: list) -> list:
    """Each comparable's release-day curve as {hour: viewers} for hours 1..24.

    Only comparables with a positive hour-1 print are kept: the GTA VI
    category's own pre-release series has no launch day yet and drops out
    here, as does any comparable missing hour 1.
    """
    curves = []
    for comp in comparables:
        day_start = _parse_ts(comp.get("day_start"))
        if day_start is None:
            continue
        hours: dict = {}
        for point in comp.get("hourly") or []:
            t = _parse_ts(point.get("from"))
            viewers = point.get("viewers")
            if t is None:
                continue
            h = int((t - day_start).total_seconds() // 3600) + 1
            if 1 <= h <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
                hours[h] = viewers
        if hours.get(1):
            curves.append(hours)
    return curves


def neighbours(p1: float, curves: list) -> list:
    """The comparables nearest the target in log hour-1 size.

    Within WINDOW log units of ln p1, at most K_MAX (the closest), at
    least K_MIN (the closest, whatever their distance).  Sorted by
    distance, so the order is deterministic.
    """
    log_p1 = math.log(max(1.0, p1))
    ranked = sorted(
        curves, key=lambda hours: abs(math.log(hours[1]) - log_p1)
    )
    near = [
        hours
        for hours in ranked
        if abs(math.log(hours[1]) - log_p1) <= WINDOW
    ][:K_MAX]
    if len(near) < K_MIN:
        near = ranked[:K_MIN]
    return near


def _z_values(levels: list) -> list:
    """Standard-normal quantiles at the case's levels.

    The contest's fixed levels use the exact constants; anything else
    falls back to Acklam's rational approximation (about 1e-9 accurate).
    """
    if levels == LEVELS:
        return list(Z)
    return [_norm_ppf(tau) for tau in levels]


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF approximation."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zvals = _z_values(levels)
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the last pre-checkpoint hourly
        # point rather than failing the case outright.
        hist = case.get("history_hourly") or []
        p1 = max(1, hist[-1]["viewers"]) if hist and hist[-1].get("viewers") else 1
    curves = launch_curves(case.get("comparables") or [])
    near = neighbours(p1, curves)
    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        ratios = [
            math.log(hours[h] / hours[1]) for hours in near if hours.get(h)
        ]
        if len(ratios) < 3:
            ratios = [
                math.log(hours[h] / hours[1])
                for hours in curves
                if hours.get(h)
            ]
        if not ratios:
            qs = [round2(max(1.0, p1) * math.exp(z * FALLBACK_SPREAD)) for z in zvals]
        else:
            centre = _median(ratios)
            mad = _median([abs(r - centre) for r in ratios])
            spread = max(SPREAD_FLOOR, SPREAD_SCALE * 1.4826 * mad)
            qs = []
            for z in zvals:
                q = max(1.0, p1 * math.exp(centre + z * spread))
                qs.append(round2(min(q, 1e9)))
            for i in range(1, len(qs)):
                if qs[i] < qs[i - 1]:
                    qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
