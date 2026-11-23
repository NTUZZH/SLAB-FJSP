# Out-of-distribution evaluation: SLAB policy against SPT and LLF

- Paired test sets: every shifted order has the generator seed of its nominal twin and differs only in the shifted parameter (make_ood_testsets.py).
- SLAB: greedy decoding of seeds 301, 302, 303 on the CPU with one torch thread; rules with eval_ppvc tie seeds (50 + order index). Every schedule is validated against the waits of its set.
- Gain: 100 x (rule - SLAB seed mean) / rule on mean makespans, positive = SLAB shorter; rule = the better of SPT and LLF on that set. W/T/L: orders on which the SLAB seed mean is shorter / equal / longer. p: two-sided Wilcoxon signed-rank (zeros dropped); p (Holm) over all rows. r: matched-pairs rank-biserial, r < 0 favours SLAB.
- sigma: mean over orders of Q / (Q + W) on the SPT schedule (share of module idle time between operations spent queuing).

| condition | modules | orders | SPT | LLF | SLAB s301 | s302 | s303 | SLAB mean | rule | gain % | gain % per seed | W/T/L | p | p (Holm) | r | sigma SPT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal (waits x1, 25 stations, spread +-20%) | 10 | 100 | 216.0 | 215.7 | 210.9 | 211.0 | 211.8 | 211.2 | LLF | 2.08 | 1.83 to 2.22 | 76/5/19 | 4.0e-09 | 3.6e-08 | -0.70 | 0.127 |
| nominal (waits x1, 25 stations, spread +-20%) | 50 | 50 | 449.0 | 450.8 | 412.5 | 415.1 | 417.9 | 415.2 | SPT | 7.53 | 6.91 to 8.13 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.630 |
| waits x0.5 | 10 | 100 | 171.2 | 170.8 | 165.6 | 165.2 | 165.9 | 165.6 | LLF | 3.05 | 2.85 to 3.27 | 90/2/8 | 1.6e-15 | 2.9e-14 | -0.93 | 0.285 |
| waits x0.5 | 50 | 50 | 407.6 | 410.2 | 376.2 | 379.3 | 379.6 | 378.4 | SPT | 7.16 | 6.87 to 7.68 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.769 |
| waits x1.5 | 10 | 100 | 269.8 | 269.7 | 265.9 | 266.3 | 266.5 | 266.2 | LLF | 1.26 | 1.16 to 1.41 | 66/4/30 | 7.7e-06 | 2.3e-05 | -0.53 | 0.083 |
| waits x1.5 | 50 | 50 | 485.1 | 486.0 | 448.0 | 449.1 | 451.2 | 449.4 | SPT | 7.35 | 6.98 to 7.64 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.526 |
| waits x2 | 10 | 100 | 326.1 | 326.2 | 323.4 | 323.5 | 323.7 | 323.5 | SPT | 0.80 | 0.76 to 0.83 | 60/6/34 | 5.1e-04 | 0.001 | -0.41 | 0.063 |
| waits x2 | 50 | 50 | 517.1 | 517.2 | 486.3 | 482.8 | 486.9 | 485.3 | SPT | 6.14 | 5.84 to 6.62 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.437 |
| waits x3 | 10 | 100 | 441.3 | 441.3 | 440.0 | 439.6 | 439.6 | 439.7 | SPT | 0.35 | 0.29 to 0.39 | 51/2/47 | 0.090 | 0.090 | -0.20 | 0.042 |
| waits x3 | 50 | 50 | 588.9 | 589.1 | 573.6 | 558.7 | 564.7 | 565.7 | SPT | 3.94 | 2.59 to 5.12 | 43/0/7 | 4.7e-08 | 2.8e-07 | -0.89 | 0.289 |
| 20 stations | 10 | 100 | 225.8 | 225.6 | 221.2 | 221.6 | 222.3 | 221.7 | LLF | 1.74 | 1.47 to 1.95 | 77/0/23 | 4.6e-09 | 3.7e-08 | -0.67 | 0.179 |
| 20 stations | 50 | 50 | 562.7 | 566.5 | 519.9 | 533.7 | 528.3 | 527.3 | SPT | 6.29 | 5.16 to 7.60 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.712 |
| 30 stations | 10 | 100 | 210.0 | 209.9 | 206.0 | 206.2 | 206.6 | 206.3 | LLF | 1.72 | 1.54 to 1.83 | 69/5/26 | 4.0e-08 | 2.8e-07 | -0.65 | 0.116 |
| 30 stations | 50 | 50 | 385.6 | 386.1 | 366.2 | 362.5 | 367.0 | 365.2 | SPT | 5.28 | 4.82 to 6.00 | 46/0/4 | 2.6e-09 | 2.6e-08 | -0.97 | 0.560 |
| 35 stations | 10 | 100 | 205.1 | 205.0 | 203.0 | 202.7 | 203.0 | 202.9 | LLF | 1.04 | 0.99 to 1.12 | 65/6/29 | 3.4e-06 | 1.4e-05 | -0.55 | 0.064 |
| 35 stations | 50 | 50 | 356.8 | 357.5 | 330.8 | 334.4 | 333.7 | 333.0 | SPT | 6.69 | 6.30 to 7.29 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.520 |
| duration spread +-40% | 10 | 100 | 207.1 | 207.1 | 204.1 | 203.5 | 203.9 | 203.8 | LLF | 1.57 | 1.45 to 1.72 | 66/5/29 | 1.9e-06 | 9.4e-06 | -0.56 | 0.121 |
| duration spread +-40% | 50 | 50 | 417.2 | 421.8 | 384.9 | 380.0 | 384.5 | 383.1 | SPT | 8.16 | 7.73 to 8.91 | 50/0/0 | 7.5e-10 | 1.3e-08 | -1.00 | 0.609 |

## Mean makespan relative to the nominal twins (%)

| condition | modules | SPT | LLF | SLAB seed mean |
|---|---|---|---|---|
| nominal (waits x1, 25 stations, spread +-20%) | 10 | +0.0 | +0.0 | +0.0 |
| nominal (waits x1, 25 stations, spread +-20%) | 50 | +0.0 | +0.0 | +0.0 |
| waits x0.5 | 10 | -20.7 | -20.8 | -21.6 |
| waits x0.5 | 50 | -9.2 | -9.0 | -8.9 |
| waits x1.5 | 10 | +24.9 | +25.0 | +26.1 |
| waits x1.5 | 50 | +8.0 | +7.8 | +8.3 |
| waits x2 | 10 | +51.0 | +51.2 | +53.2 |
| waits x2 | 50 | +15.2 | +14.7 | +16.9 |
| waits x3 | 10 | +104.3 | +104.6 | +108.2 |
| waits x3 | 50 | +31.2 | +30.7 | +36.3 |
| 20 stations | 10 | +4.5 | +4.6 | +5.0 |
| 20 stations | 50 | +25.3 | +25.7 | +27.0 |
| 30 stations | 10 | -2.8 | -2.7 | -2.3 |
| 30 stations | 50 | -14.1 | -14.4 | -12.0 |
| 35 stations | 10 | -5.0 | -4.9 | -3.9 |
| 35 stations | 50 | -20.5 | -20.7 | -19.8 |
| duration spread +-40% | 10 | -4.1 | -4.0 | -3.5 |
| duration spread +-40% | 50 | -7.1 | -6.4 | -7.7 |
