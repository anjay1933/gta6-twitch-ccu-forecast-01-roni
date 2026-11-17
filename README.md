[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/anjay1933/gta6-twitch-ccu-forecast-01-roni/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/anjay1933/gta6-twitch-ccu-forecast-01-roni/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — @roni-rockabilly entry

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything
recorded up to the end of the first hour. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250 total: 1st $50, 2nd $35, 3rd $25, 4th–10th $10, 11th–20th $7).

## Method — windowed k-nearest-neighbour median log-ratio curves

`python3 main.py` (Python 3.13, standard library only, no network, no model
call). It reads `/input/<id>.in` and writes `/output/<id>.out`: 23 rows
(hours 2–24), each five quantiles of that hour's concurrent viewers.

For every comparable launch with a valid hour-1 print, the launch-day curve
is reduced to log-ratios `r_h = ln(viewers_h / viewers_1)`. The neighbours
of the target case are the comparables whose own hour-1 print is closest in
log space — within 1.5 log units, at most the 12 closest, at least the 5
closest. Each hour's forecast is a log-normal centred on the **median** of
the neighbours' `r_h` and spread by their median absolute deviation:

```
q_tau(h) = p1 * exp(median_h + z_tau * 1.15 * 1.4826 * MAD_h),  spread floored at 0.15
```

A launch's hour-1 print encodes its unlock pattern: a rolling-midnight
launch (GTA VI, and all four practice launches) already has a large hour-1
count, so its neighbours are other rolling-midnight launches; a
simultaneous global unlock has a near-zero hour-1 count and its neighbours
are other simultaneous unlocks. The median/MAD pair is robust — one freak
launch cannot drag the centre the way a mean does. No free-text parsing of
the unlock notes is used.

## Validation

Leave-one-out over the 37 launch comparables in the kit (each scored as a
pseudo-practice case against the other 36; the official practice answers
are not in the kit). Full numbers in [GRADE_REPORT.md](GRADE_REPORT.md).

| subset | this model | house baseline |
|---|---|---|
| all 37 launches | **0.3944** | 0.9296 |
| hour-1 ≥ 10k (practice-like) | **0.2635** | 0.3508 |
| hour-1 ≥ 100k (GTA-VI-like) | **0.2215** | 0.2611 |

90% coverage 0.84; typical miss 49% on hour-1 ≥ 10k launches. Official
practice baseline ≈ 0.387. Deterministic (two runs byte-identical),
~0.06 s per case (limit 8 s).

## Files

- `main.py` — the entrypoint (the whole model, ~230 lines)
- `GRADE_REPORT.md` — LOO validation, method comparison, determinism checks
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet `D6RZVQzRR2qabWZe7XnnXtX8VuEhzMz5TdxefYsRpLmH`
