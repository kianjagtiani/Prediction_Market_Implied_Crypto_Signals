# Focused Research Report: Prediction Markets as a BTC Signal

Run date: 2026-05-05 15:07 PDT
Sample: 2025-01-01 through 2026-04-01 UTC, primary BTC Kalshi universe only.

## Executive Conclusion
The focused run passes the statistical lead-lag screen (7 raw-significant pairs, 5 Bonferroni-significant), but it does not clear the direction-adjusted event-study gate: no definition shows statistically significant peak net CAR above 0.3% after a 10 bp cost assumption. The best OOS backtest is D1 H=240 with Sharpe=1.914, and a no-forward-fill sensitivity still has D1 OOS Sharpe=1.623, but this is not enough to declare a tradeable edge because the event-study evidence points the other way under both treatments.

## Data Coverage

| ticker                      | observed_bars | coverage_pct       | first_bar                 | last_bar                  |
| --------------------------- | ------------- | ------------------ | ------------------------- | ------------------------- |
| KXBTCMAXY-26DEC31-109999.99 | 50959         | 7.777625152625153  | 2026-01-02 23:01:00+00:00 | 2026-04-01 00:00:00+00:00 |
| KXBTCMAX100-26-SEP          | 22913         | 3.4971001221001217 | 2026-02-17 21:01:00+00:00 | 2026-03-31 23:57:00+00:00 |
| KXBTCMAXY-26DEC31-99999.99  | 55184         | 8.422466422466423  | 2026-01-02 23:01:00+00:00 | 2026-03-31 23:59:00+00:00 |
| KXBTCMAX100-26-JUNE         | 59516         | 9.083638583638583  | 2026-01-05 20:06:00+00:00 | 2026-03-31 23:59:00+00:00 |
| KXBTCMAX100-26-MAY          | 54344         | 8.294261294261295  | 2026-01-05 20:06:00+00:00 | 2026-03-31 23:53:00+00:00 |

## Signal Counts

| market                      | D1  | D2 | D3 | D4 |
| --------------------------- | --- | -- | -- | -- |
| KXBTCMAX100-26-JUNE         | 127 | 16 | 9  | 63 |
| KXBTCMAX100-26-MAY          | 79  | 18 | 4  | 31 |
| KXBTCMAX100-26-SEP          | 24  | 5  | 5  | 11 |
| KXBTCMAXY-26DEC31-109999.99 | 45  | 3  | 1  | 35 |
| KXBTCMAXY-26DEC31-99999.99  | 38  | 5  | 1  | 35 |

## Lead-Lag Gate

| market                      | definition | n_events | min_p_raw              | best_lag | min_p_bonferroni       | significant_raw | significant_bonferroni |
| --------------------------- | ---------- | -------- | ---------------------- | -------- | ---------------------- | --------------- | ---------------------- |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 45       | 2.7325805624779556e-10 | 27       | 2.1860644499823645e-09 | True            | True                   |
| KXBTCMAX100-26-MAY          | D4         | 31       | 0.0010189839769715713  | 23       | 0.00815187181577257    | True            | True                   |
| KXBTCMAXY-26DEC31-109999.99 | D4         | 35       | 0.0016199134215469355  | 24       | 0.012959307372375484   | True            | True                   |
| KXBTCMAXY-26DEC31-99999.99  | D1         | 38       | 0.0029222015586466282  | 30       | 0.023377612469173026   | True            | True                   |
| KXBTCMAXY-26DEC31-99999.99  | D4         | 35       | 0.0036905812735095016  | 30       | 0.029524650188076013   | True            | True                   |
| KXBTCMAX100-26-JUNE         | D1         | 127      | 0.01501798901693867    | 10       | 0.12014391213550936    | True            | False                  |
| KXBTCMAX100-26-JUNE         | D4         | 63       | 0.02408071498273657    | 30       | 0.19264571986189255    | True            | False                  |
| KXBTCMAX100-26-MAY          | D1         | 79       | 0.09459337923390665    | 5        | 0.7567470338712532     | False           | False                  |

## Best-Pair Lead-Lag Regression

| market                      | definition | horizon_min | beta                  | t_stat             | p_value            | r2                 | n_obs |
| --------------------------- | ---------- | ----------- | --------------------- | ------------------ | ------------------ | ------------------ | ----- |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 1           | -0.0001886588617086   | -0.61780500700464  | 0.540039542180524  | 0.1131517785317642 | 45    |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 10          | 0.0008110115013096    | 1.5094701452261208 | 0.1386656584158697 | 0.2052450140675236 | 45    |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 30          | 0.0003067743521446    | 0.2989638651515797 | 0.7664419571340708 | 0.3124357602472869 | 45    |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 60          | 2.543958766510932e-05 | 0.0180585955930034 | 0.9856776254960044 | 0.4369858483114843 | 45    |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 240         | 0.0013410890529783    | 0.5426564375663954 | 0.5902345571029814 | 0.1478419750970796 | 45    |
| KXBTCMAXY-26DEC31-109999.99 | D1         | 1440        | 0.0017664437210997    | 0.2961497075451328 | 0.7686106733592533 | 0.113645266249667  | 44    |

## Cross-Asset Placebo

| lag_min | corr_target         | ci_lower            | ci_upper           | corr_other          | corr_diff           |
| ------- | ------------------- | ------------------- | ------------------ | ------------------- | ------------------- |
| -60     | 0.0042408356466424  | -0.0055046842201608 | 0.0055046842201608 | 0.0024671940234067  | 0.0017736416232356  |
| -30     | 0.001965862598253   | -0.0055046842201608 | 0.0055046842201608 | -0.0005093676429912 | 0.0024752302412442  |
| -10     | -0.0021420256224254 | -0.0055046842201608 | 0.0055046842201608 | -0.0011391055757725 | -0.0010029200466528 |
| -5      | 0.0037982702126042  | -0.0055046842201608 | 0.0055046842201608 | 0.0045737618038139  | -0.0007754915912097 |
| -1      | 0.0051674279813011  | -0.0055046842201608 | 0.0055046842201608 | 0.0052887564457051  | -0.000121328464404  |
| 0       | 0.0015785989446198  | -0.0055046842201608 | 0.0055046842201608 | 0.005951299397428   | -0.0043727004528082 |
| 1       | -0.0006005351201383 | -0.0055046842201608 | 0.0055046842201608 | -0.0014748249855589 | 0.0008742898654206  |
| 5       | 0.0026494630336716  | -0.0055046842201608 | 0.0055046842201608 | 0.0029342007569505  | -0.0002847377232789 |
| 10      | -0.0002783559520205 | -0.0055046842201608 | 0.0055046842201608 | -0.0023318757533385 | 0.0020535198013179  |
| 30      | -0.0015611748463116 | -0.0055046842201608 | 0.0055046842201608 | 0.0003537672012651  | -0.0019149420475768 |
| 60      | 0.0013003474529955  | -0.0055046842201608 | 0.0055046842201608 | 0.0040311549889151  | -0.0027308075359195 |

## Event Study Gate

| definition | n_events | peak_t | peak_direction_adjusted_car | peak_p_value        | net_peak_car_after_10bp | passes_tradeability_gate |
| ---------- | -------- | ------ | --------------------------- | ------------------- | ----------------------- | ------------------------ |
| D1         | 304      | 1      | -0.018%                     | 0.1689889711044861  | -0.118%                 | False                    |
| D2         | 44       | 232    | 0.105%                      | 0.5401489833059001  | 0.005%                  | False                    |
| D3         | 18       | 20     | 0.356%                      | 0.18518707350349836 | 0.256%                  | False                    |
| D4         | 170      | 12     | -0.029%                     | 0.34780439012744996 | -0.129%                 | False                    |

## Backtest Gate

| definition | holding_period | sharpe | sortino | max_drawdown_pct | win_rate_pct | profit_factor | n_trades | signal_events |
| ---------- | -------------- | ------ | ------- | ---------------- | ------------ | ------------- | -------- | ------------- |
| D1         | 240            | 1.914  | 0.269   | 2.2              | 21.8         | 2.17          | 124      | 304           |
| D4         | 240            | 0.816  | 0.062   | 2.35             | 17.6         | 1.51          | 68       | 170           |
| D2         | 240            | 0.03   | 0.0     | 1.22             | 25.0         | 1.03          | 20       | 44            |
| D3         | 10             | -2.431 | -0.026  | 0.94             | 10.0         | 0.04          | 10       | 18            |

## No-Forward-Fill Sensitivity

This reruns signal construction on observed Kalshi bars only, without carrying the last probability across inactive minutes.

| definition | events | best_h | test_sharpe | event_peak_net_car | event_p  | granger_min_p |
| ---------- | ------ | ------ | ------------ | ------------------ | -------- | ------------- |
| D1         | 258    | 240    | 1.623        | -0.115%            | 0.515    | 2.04e-07      |
| D2         | 37     | 240    | 0.350        | 0.057%             | 0.427    | 9.93e-06      |
| D3         | 16     | 10     | -1.080       | 0.305%             | 0.181    | n/a           |
| D4         | 147    | 240    | 0.493        | -0.122%            | 0.529    | 9.69e-03      |

## Remaining Work

- Repeat the run on the full 53-market universe if disk/time permits.
- Run a 15-minute aggregation check.
- Treat CPI/FED markets as regime filters only after the primary BTC signal is validated.
- Use the historical DVOL files for conditioning once the spot signal passes the event-study and OOS backtest gates.
