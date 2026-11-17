# GRADE_REPORT — gta6-twitch-ccu-forecast-01 (@roni-rockabilly)

Entrypoint: `python3 main.py` (Python 3.13, stdlib only, no network, no
model call). Method: windowed k-nearest-neighbour **median** log-ratio
curves — neighbours within 1.5 log units of the target's log hour-1 print
(max 12, min 5 closest); per-hour centre = median of the neighbours'
`ln(viewers_h / viewers_1)`; spread = `1.15 * 1.4826 * MAD`, floored at
0.15; quantiles `p1 * exp(median + z_tau * spread)`.

Anchor: `makefun anchor b9a7e494fde0931a57d5189b22ecd99cfb72db9d round 34033528`
(printed by `makefun init` before the first counted write).

## Why this method, and how it differs from the agent account's entry

The agent account's entry (kernel-weighted log-ratio curves) weights **all**
comparables by a Gaussian kernel on `|ln v1_c - ln p1|` (bandwidth 1.75),
takes the **weighted mean** of the log-ratios, and spreads by the
**weighted standard deviation** widened by effective-sample-size. This
entry is a different estimator on every part of that pipeline:

| part | agent entry | this entry |
|---|---|---|
| weighting | Gaussian kernel over all comparables | hard window (1.5 log units), uniform weights, max 12 / min 5 nearest |
| centre | weighted mean | **median** of the neighbours |
| spread | weighted std, ESS-widened, ×0.85, floor 0.10 | **MAD** ×1.4826, ×1.15, floor 0.15 |
| features used | log hour-1 print only | log hour-1 print only (same input, different use) |

No code is shared: `main.py` was written fresh for this entry. The two
entries produce different forecasts (different numbers on every practice
case; e.g. p02 hour-12 median 156,117 here).

Other designs tested and rejected (LOO, same harness, same 37 launches):

| method | loss all | loss v1≥10k | loss v1≥100k |
|---|---|---|---|
| house baseline (persistence) | 0.9296 | 0.3508 | 0.2611 |
| per-hour ridge regression on log-ratios (features: ln v1, ln v1², hype ratio, pre-release trend, viewers/channel) | 0.4386 | 0.3341 | — |
| spline (hinge) regression on ln v1 | 0.4164 | 0.3040 | — |
| family-median (hard size families, shrunk median) | 0.6169 | 0.2880 | — |
| k-NN median, k=12, no window | 0.4551 | 0.2639 | — |
| **windowed k-NN median (chosen)** | **0.3944** | **0.2635** | **0.2215** |
| kernel weighted mean (agent method, benchmark reimplementation) | 0.3934 | 0.2602 | 0.2197 |

The regression family was rejected on the numbers: a global parametric fit
cannot follow the nonlinear scale→shape relationship that local methods
capture (0.30–0.33 vs 0.26 on practice-like launches). The direct pinball
quantile-regression variant was also tried and abandoned — its subgradient
optimizer diverged (LOO loss >6) and it is not worth the runtime risk
inside the 8 s sandbox limit. The chosen method ties the kernel benchmark
on every subset (differences ≤0.004, within LOO noise) while being a
genuinely different estimator.

## LOO validation (shipped code, not a scratch copy)

Harness: `/tmp/gta6_validate.py` imports `main.py` from this repo and, for
each of the 37 usable comparables in the kit's `p00.in`, builds a
pseudo-case (known hour-1 = the held-out launch's hour-1, comparables =
the other 36 raw comp dicts), forecasts with `main.forecast_case`, and
scores against the held-out launch's actual hours 2–24 with the contest's
pinball loss on the log scale. The official practice answers are not in
the kit (grade-local grades on the server only), so this LOO proxy is the
validation — the same protocol the agent entry used, and the harness
reproduces the agent entry's reported figures (kernel 0.2602 vs reported
0.260 on hour-1 ≥ 10k; baseline 0.3508 vs reported 0.351), which validates
the harness itself.

| subset (held-out launches) | n launches | this model | house baseline |
|---|---|---|---|
| all | 37 | **0.3944** | 0.9296 |
| hour-1 ≥ 10k (practice-like) | 28 | **0.2635** | 0.3508 |
| hour-1 ≥ 100k (GTA-VI-like) | 13 | **0.2215** | 0.2611 |

- 90% coverage: 0.838 (honest ≈ 0.9; the spread is a touch narrow).
- Typical miss (median): 77.1% over all launches, **49.3%** on hour-1 ≥
  10k launches. The all-launch figure is dominated by the tiny
  simultaneous-unlock launches (hour-1 of 164–1,400 viewers exploding
  100x); the final is a huge rolling-midnight launch, the subset where
  the model is strongest.
- Official practice baseline ≈ 0.387. This model beats the baseline by
  ~25% on the practice-like subset and ~15% on the GTA-VI-like subset.

Sanity — median forecasts on the four practice cases (hour-1 print →
hour-12 median):

| case | game | hour-1 | h2 | h6 | h12 | h18 | h24 |
|---|---|---|---|---|---|---|---|
| p00 | Marvel's Wolverine | 23,357 | 19,311 | 21,812 | 22,722 | 43,899 | 54,901 |
| p01 | Red Dead Redemption 2 | 116,550 | 105,369 | 112,771 | 109,659 | 169,421 | 185,810 |
| p02 | God of War Ragnarök | 269,517 | 242,148 | 193,025 | 156,117 | 337,397 | 306,884 |
| p03 | Marvel's Spider-Man 2 | 119,087 | 107,663 | 115,225 | 112,046 | 173,109 | 189,854 |

(Hour-12 shown; full 23-row outputs verified valid — see below. The
medians sit below hour-1 at midday, matching the usual launch-day shape:
night dip, US-unlock rises, afternoon/evening peak, late fall.)

## Determinism and format checks

- Ran `python3 -P -s main.py` twice on all four practice inputs
  (`MAKEFUN_INPUT_DIR`/`MAKEFUN_OUTPUT_DIR`): outputs **byte-identical**.
- All four outputs valid: exactly 23 rows, hours 2–24 each once, five
  quantiles each, finite, in (0, 1e9], non-decreasing; every file ≤
  65,536 bytes (1,442–1,541 bytes).
- Runtime: 0.225 s for all four cases (~0.06 s per case; limit 8 s).
- No randomness, no wall clock, no network, no model call; sorted
  iteration everywhere.

## Honest limitations

- LOO over comparables is a proxy, not the official practice grade: the
  final is a single launch, so the realised loss has high variance — a
  0.26 LOO does not guarantee beating the house model forecast on the
  final (its loss is unknown until grading).
- On tiny simultaneous-unlock launches the model is barely better than
  the baseline in absolute terms (loss_all 0.394 vs the kernel's 0.393);
  it is not a general forecaster, it is tuned by its neighbour rule to
  big rolling-midnight launches — which is exactly what GTA VI is.
- The practice launches are all rolling-midnight big launches; if GTA
  VI's hour-1 print lands far outside the comparables' range (above
  ~875k), the neighbour window falls back to the 5 closest launches and
  the spread stays at the MAD floor — the forecast degrades gracefully
  rather than failing.
