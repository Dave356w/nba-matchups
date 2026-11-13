# Realized WP shadow forward research

Current version realized_wp_rolling50_bandadj_v2: this game's no-vig WP + (cell win rate − cell mean no-vig WP), clipped to 0–100%.
realized_wp_rolling50_v1 (raw cell win rate) is closed; its captures stay graded under their own version, never pooled.

First valid saved forecast within five hours of tip. No backfill. Each source forecast must also be within that window.
Flat 1u only at positive expected return; either side; at most one position per estimator per game.
All probability scores use the same official graded games, including abstentions. Saved prices are paper quotes.
Build-time saves do not prove publication, viewing or executable prices. Public records and picks are unchanged.

Saved distinct pre-tip captures: 19.

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

