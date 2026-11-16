# Realized WP shadow forward research

Current version realized_wp_shrunk200_drift_v4: this game's no-vig WP + (cell win rate − cell mean no-vig WP) × n/(n+200) over the latest 200 cell results + pooled drift (half-life 100 picks, 200 pseudo-games, latest 600), clipped to 0–100%.
realized_wp_rolling200_bandadj_v3 (the unshrunk cell adjustment over the latest 200, no drift), realized_wp_rolling50_bandadj_v2 (the same over the latest 50) and realized_wp_rolling50_v1 (raw cell win rate) are closed; their captures stay graded under their own versions, never pooled.

First valid saved forecast within five hours of tip. No backfill. Each source forecast must also be within that window.
Flat 1u only at positive expected return; either side; at most one position per estimator per game.
All probability scores use the same official graded games, including abstentions. Saved prices are paper quotes.
Build-time saves do not prove publication, viewing or executable prices. Public records and picks are unchanged.

Saved distinct pre-tip captures: 388.

## realized_wp_rolling200_bandadj_v3

Official games: 5; graded: 5; pending: 0.

### pooled

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.2598 | 0.7101 | 0.4000 | 5 | 3.5973 | 0.7195 |
| market | 0.2937 | 0.7916 | 0.6000 | 0 | 0.0000 | — |
| realized | 0.3148 | 0.8319 | 0.4000 | 2 | -2.0000 | -1.0000 |

Same-game market favorite: {"bets":5,"net_units":-0.38881075167995005,"roi":-0.07776215033599002}

### opener

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### early

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### established

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.2598 | 0.7101 | 0.4000 | 5 | 3.5973 | 0.7195 |
| market | 0.2937 | 0.7916 | 0.6000 | 0 | 0.0000 | — |
| realized | 0.3148 | 0.8319 | 0.4000 | 2 | -2.0000 | -1.0000 |

Same-game market favorite: {"bets":5,"net_units":-0.38881075167995005,"roi":-0.07776215033599002}

## realized_wp_rolling50_bandadj_v2

Official games: 4; graded: 4; pending: 0.

### pooled

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.0926 | 0.3583 | 1.0000 | 4 | -2.5633 | -0.6408 |
| market | 0.0901 | 0.3536 | 1.0000 | 0 | 0.0000 | — |
| realized | 0.0655 | 0.2851 | 1.0000 | 3 | -0.1812 | -0.0604 |

Same-game market favorite: {"bets":4,"net_units":1.4837925886344416,"roi":0.3709481471586104}

### opener

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### early

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### established

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.0926 | 0.3583 | 1.0000 | 4 | -2.5633 | -0.6408 |
| market | 0.0901 | 0.3536 | 1.0000 | 0 | 0.0000 | — |
| realized | 0.0655 | 0.2851 | 1.0000 | 3 | -0.1812 | -0.0604 |

Same-game market favorite: {"bets":4,"net_units":1.4837925886344416,"roi":0.3709481471586104}

## realized_wp_rolling50_v1

Official games: 4; graded: 4; pending: 0.

### pooled

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.0926 | 0.3583 | 1.0000 | 4 | -2.5633 | -0.6408 |
| market | 0.0901 | 0.3536 | 1.0000 | 0 | 0.0000 | — |
| realized | 0.0672 | 0.2873 | 1.0000 | 3 | -0.1812 | -0.0604 |

Same-game market favorite: {"bets":4,"net_units":1.4837925886344416,"roi":0.3709481471586104}

### opener

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### early

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### established

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.0926 | 0.3583 | 1.0000 | 4 | -2.5633 | -0.6408 |
| market | 0.0901 | 0.3536 | 1.0000 | 0 | 0.0000 | — |
| realized | 0.0672 | 0.2873 | 1.0000 | 3 | -0.1812 | -0.0604 |

Same-game market favorite: {"bets":4,"net_units":1.4837925886344416,"roi":0.3709481471586104}

## realized_wp_shrunk200_drift_v4

Official games: 23; graded: 23; pending: 0.

### pooled

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.1958 | 0.5838 | 0.6957 | 23 | 4.8950 | 0.2128 |
| market | 0.2055 | 0.5994 | 0.6957 | 0 | 0.0000 | — |
| realized | 0.2126 | 0.6245 | 0.6957 | 20 | -2.0164 | -0.1008 |

Same-game market favorite: {"bets":23,"net_units":-1.41499299457391,"roi":-0.06152143454669173}

### opener

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### early

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | — | — | — | 0 | 0.0000 | — |
| market | — | — | — | 0 | 0.0000 | — |
| realized | — | — | — | 0 | 0.0000 | — |

Same-game market favorite: {"bets":0,"net_units":0,"roi":null}

### established

| Estimator | Brier | Log loss | Winner accuracy | Bets | Net units | ROI |
| --- | --- | --- | --- | --- | --- | --- |
| model | 0.1958 | 0.5838 | 0.6957 | 23 | 4.8950 | 0.2128 |
| market | 0.2055 | 0.5994 | 0.6957 | 0 | 0.0000 | — |
| realized | 0.2126 | 0.6245 | 0.6957 | 20 | -2.0164 | -0.1008 |

Same-game market favorite: {"bets":23,"net_units":-1.41499299457391,"roi":-0.06152143454669173}

