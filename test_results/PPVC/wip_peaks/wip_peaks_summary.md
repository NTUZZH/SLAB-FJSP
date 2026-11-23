# Work in progress and lag-buffer peaks of the makespan schedules

- Nominal test sets (all orders). SLAB: greedy decoding of seeds 301, 302, 303 on the CPU with one torch thread; rules with eval_ppvc tie seeds (50 + order index). Schedules validated against the true waits.
- WIP(t): modules between the start of their first and the completion of their last operation. Peak = max over [0, makespan]; P95 = 95th percentile over time; mean = time average (= sum of flow times / makespan). Buffer: modules in a post-operation wait at the same instant.
- Values are means over orders; "max" columns are the largest value over the orders.

| modules | method | orders | makespan | mean flow | WIP mean | WIP P95 | WIP peak | WIP peak (max) | buffer mean | buffer peak | buffer peak (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | SLAB-s301 | 100 | 210.9 | 169.3 | 8.03 | 10.00 | 10.00 | 10 | 3.74 | 8.76 | 10 |
| 10 | SLAB-s302 | 100 | 211.0 | 168.9 | 8.01 | 10.00 | 10.00 | 10 | 3.74 | 8.88 | 10 |
| 10 | SLAB-s303 | 100 | 211.8 | 168.8 | 7.98 | 10.00 | 10.00 | 10 | 3.73 | 8.86 | 10 |
| 10 | SPT | 100 | 216.0 | 177.6 | 8.23 | 10.00 | 10.00 | 10 | 3.66 | 8.90 | 10 |
| 10 | LLF | 100 | 215.7 | 177.7 | 8.24 | 10.00 | 10.00 | 10 | 3.66 | 8.87 | 10 |
| 10 | SLAB seed mean | 100 | 211.2 | 169.0 | 8.01 | 10.00 | 10.00 | 10 | 3.74 | 8.83 | 10 |
| 20 | SLAB-s301 | 50 | 254.5 | 188.6 | 14.82 | 20.00 | 20.00 | 20 | 6.41 | 13.28 | 15 |
| 20 | SLAB-s302 | 50 | 254.0 | 191.3 | 15.07 | 20.00 | 20.00 | 20 | 6.42 | 13.52 | 16 |
| 20 | SLAB-s303 | 50 | 255.6 | 191.3 | 14.98 | 20.00 | 20.00 | 20 | 6.39 | 13.54 | 15 |
| 20 | SPT | 50 | 265.1 | 207.3 | 15.64 | 20.00 | 20.00 | 20 | 6.16 | 13.52 | 16 |
| 20 | LLF | 50 | 265.0 | 207.8 | 15.69 | 20.00 | 20.00 | 20 | 6.16 | 13.40 | 16 |
| 20 | SLAB seed mean | 50 | 254.7 | 190.4 | 14.95 | 20.00 | 20.00 | 20 | 6.41 | 13.45 | 15 |
| 30 | SLAB-s301 | 20 | 310.4 | 202.6 | 19.59 | 29.75 | 29.85 | 30 | 7.69 | 15.30 | 18 |
| 30 | SLAB-s302 | 20 | 305.9 | 210.7 | 20.67 | 29.75 | 29.95 | 30 | 7.80 | 15.20 | 17 |
| 30 | SLAB-s303 | 20 | 308.8 | 208.2 | 20.24 | 29.55 | 29.90 | 30 | 7.73 | 15.60 | 17 |
| 30 | SPT | 20 | 324.9 | 242.5 | 22.40 | 30.00 | 30.00 | 30 | 7.35 | 15.20 | 17 |
| 30 | LLF | 20 | 326.0 | 243.7 | 22.42 | 30.00 | 30.00 | 30 | 7.32 | 14.85 | 17 |
| 30 | SLAB seed mean | 20 | 308.4 | 207.2 | 20.16 | 29.68 | 29.90 | 30 | 7.74 | 15.37 | 17 |
| 50 | SLAB-s301 | 50 | 412.5 | 237.0 | 28.72 | 45.40 | 46.34 | 50 | 9.91 | 20.48 | 23 |
| 50 | SLAB-s302 | 50 | 415.1 | 258.9 | 31.18 | 47.20 | 48.02 | 50 | 9.84 | 22.34 | 28 |
| 50 | SLAB-s303 | 50 | 417.9 | 246.3 | 29.48 | 45.94 | 47.10 | 50 | 9.78 | 19.90 | 24 |
| 50 | SPT | 50 | 449.0 | 309.2 | 34.43 | 49.98 | 50.00 | 50 | 9.10 | 18.08 | 24 |
| 50 | LLF | 50 | 450.8 | 311.0 | 34.48 | 49.96 | 50.00 | 50 | 9.06 | 18.14 | 23 |
| 50 | SLAB seed mean | 50 | 415.2 | 247.4 | 29.79 | 46.18 | 47.15 | 50 | 9.84 | 20.91 | 24 |
| 80 | SLAB-s301 | 50 | 566.8 | 278.3 | 39.28 | 61.86 | 63.30 | 70 | 11.51 | 23.88 | 27 |
| 80 | SLAB-s302 | 50 | 575.4 | 318.3 | 44.25 | 67.58 | 69.30 | 73 | 11.34 | 26.20 | 30 |
| 80 | SLAB-s303 | 50 | 571.2 | 287.9 | 40.32 | 63.36 | 65.18 | 72 | 11.42 | 23.32 | 28 |
| 80 | SPT | 50 | 620.3 | 390.9 | 50.41 | 77.12 | 78.40 | 80 | 10.52 | 22.70 | 27 |
| 80 | LLF | 50 | 623.6 | 393.6 | 50.49 | 76.98 | 78.44 | 80 | 10.46 | 22.50 | 26 |
| 80 | SLAB seed mean | 50 | 571.1 | 294.9 | 41.28 | 64.27 | 65.93 | 71 | 11.42 | 24.47 | 27 |

## SLAB seed mean against each rule (per-order differences, SLAB minus rule)

| modules | rule | measure | mean diff | lower/equal/higher | p | r |
|---|---|---|---|---|---|---|
| 10 | SPT | wip_peak | +0.00 | 0/100/0 | 1.000 | 0.00 |
| 10 | SPT | wip_p95 | +0.00 | 0/100/0 | 1.000 | 0.00 |
| 10 | SPT | buffer_peak | -0.07 | 35/31/34 | 0.734 | -0.05 |
| 10 | SPT | wip_mean | -0.22 | 77/0/23 | 5.3e-09 | -0.67 |
| 10 | SPT | flow_mean | -8.62 | 97/0/3 | 1.6e-17 | -0.98 |
| 10 | LLF | wip_peak | +0.00 | 0/100/0 | 1.000 | 0.00 |
| 10 | LLF | wip_p95 | +0.00 | 0/100/0 | 1.000 | 0.00 |
| 10 | LLF | buffer_peak | -0.04 | 33/33/34 | 0.980 | 0.00 |
| 10 | LLF | wip_mean | -0.23 | 81/0/19 | 1.1e-09 | -0.70 |
| 10 | LLF | flow_mean | -8.66 | 96/0/4 | 1.4e-17 | -0.98 |
| 20 | SPT | wip_peak | +0.00 | 0/50/0 | 1.000 | 0.00 |
| 20 | SPT | wip_p95 | +0.00 | 0/50/0 | 1.000 | 0.00 |
| 20 | SPT | buffer_peak | -0.07 | 24/7/19 | 0.558 | -0.10 |
| 20 | SPT | wip_mean | -0.68 | 43/0/7 | 1.1e-10 | -0.91 |
| 20 | SPT | flow_mean | -16.87 | 50/0/0 | 1.8e-15 | -1.00 |
| 20 | LLF | wip_peak | +0.00 | 0/50/0 | 1.000 | 0.00 |
| 20 | LLF | wip_p95 | +0.00 | 0/50/0 | 1.000 | 0.00 |
| 20 | LLF | buffer_peak | +0.05 | 19/6/25 | 0.614 | 0.09 |
| 20 | LLF | wip_mean | -0.73 | 43/0/7 | 6.5e-10 | -0.89 |
| 20 | LLF | flow_mean | -17.39 | 48/0/2 | 8.9e-15 | -1.00 |
| 30 | SPT | wip_peak | -0.10 | 3/17/0 | 0.083 | -1.00 |
| 30 | SPT | wip_p95 | -0.32 | 7/13/0 | 0.016 | -1.00 |
| 30 | SPT | buffer_peak | +0.17 | 10/1/9 | 0.492 | 0.18 |
| 30 | SPT | wip_mean | -2.23 | 20/0/0 | 1.9e-06 | -1.00 |
| 30 | SPT | flow_mean | -35.32 | 20/0/0 | 1.9e-06 | -1.00 |
| 30 | LLF | wip_peak | -0.10 | 3/17/0 | 0.083 | -1.00 |
| 30 | LLF | wip_p95 | -0.32 | 7/13/0 | 0.016 | -1.00 |
| 30 | LLF | buffer_peak | +0.52 | 4/3/13 | 0.041 | 0.56 |
| 30 | LLF | wip_mean | -2.26 | 20/0/0 | 1.9e-06 | -1.00 |
| 30 | LLF | flow_mean | -36.49 | 20/0/0 | 1.9e-06 | -1.00 |
| 50 | SPT | wip_peak | -2.85 | 48/2/0 | 1.6e-09 | -1.00 |
| 50 | SPT | wip_p95 | -3.80 | 50/0/0 | 7.4e-10 | -1.00 |
| 50 | SPT | buffer_peak | +2.83 | 2/1/47 | 3.0e-09 | 0.97 |
| 50 | SPT | wip_mean | -4.63 | 50/0/0 | 1.8e-15 | -1.00 |
| 50 | SPT | flow_mean | -61.81 | 50/0/0 | 1.8e-15 | -1.00 |
| 50 | LLF | wip_peak | -2.85 | 48/2/0 | 1.6e-09 | -1.00 |
| 50 | LLF | wip_p95 | -3.78 | 50/0/0 | 7.4e-10 | -1.00 |
| 50 | LLF | buffer_peak | +2.77 | 0/1/49 | 1.1e-09 | 1.00 |
| 50 | LLF | wip_mean | -4.69 | 50/0/0 | 1.8e-15 | -1.00 |
| 50 | LLF | flow_mean | -63.62 | 50/0/0 | 1.8e-15 | -1.00 |
| 80 | SPT | wip_peak | -12.47 | 50/0/0 | 7.5e-10 | -1.00 |
| 80 | SPT | wip_p95 | -12.85 | 50/0/0 | 7.5e-10 | -1.00 |
| 80 | SPT | buffer_peak | +1.77 | 8/1/41 | 6.7e-07 | 0.81 |
| 80 | SPT | wip_mean | -9.12 | 50/0/0 | 1.8e-15 | -1.00 |
| 80 | SPT | flow_mean | -96.01 | 50/0/0 | 1.8e-15 | -1.00 |
| 80 | LLF | wip_peak | -12.51 | 50/0/0 | 7.5e-10 | -1.00 |
| 80 | LLF | wip_p95 | -12.71 | 50/0/0 | 7.5e-10 | -1.00 |
| 80 | LLF | buffer_peak | +1.97 | 5/2/43 | 1.8e-08 | 0.93 |
| 80 | LLF | wip_mean | -9.21 | 50/0/0 | 1.8e-15 | -1.00 |
| 80 | LLF | flow_mean | -98.77 | 50/0/0 | 1.8e-15 | -1.00 |

## Checks against the stored files

| set | method | by-product file: makespan mismatches, max abs diff WIP mean, flow mean | buffer file: makespan / peak / per-type mismatches |
|---|---|---|---|
| 10x25+ppvc-mixed | SLAB-s301 | 0/100, 0.0e+00, 0.0e+00 | no stored file |
| 10x25+ppvc-mixed | SLAB-s302 | no stored file | no stored file |
| 10x25+ppvc-mixed | SLAB-s303 | no stored file | no stored file |
| 10x25+ppvc-mixed | SPT | 0/100, 0.0e+00, 0.0e+00 | 0 / 0 / 0 of 100 |
| 10x25+ppvc-mixed | LLF | 0/100, 0.0e+00, 0.0e+00 | no stored file |
| 20x25+ppvc-mixed | SLAB-s301 | 0/50, 0.0e+00, 0.0e+00 | no stored file |
| 20x25+ppvc-mixed | SLAB-s302 | no stored file | no stored file |
| 20x25+ppvc-mixed | SLAB-s303 | no stored file | no stored file |
| 20x25+ppvc-mixed | SPT | 0/50, 0.0e+00, 0.0e+00 | 0 / 0 / 0 of 50 |
| 20x25+ppvc-mixed | LLF | 0/50, 0.0e+00, 0.0e+00 | no stored file |
| 30x25+ppvc-mixed | SLAB-s301 | 0/20, 0.0e+00, 0.0e+00 | no stored file |
| 30x25+ppvc-mixed | SLAB-s302 | no stored file | no stored file |
| 30x25+ppvc-mixed | SLAB-s303 | no stored file | no stored file |
| 30x25+ppvc-mixed | SPT | 0/20, 0.0e+00, 0.0e+00 | 0 / 0 / 0 of 20 |
| 30x25+ppvc-mixed | LLF | 0/20, 0.0e+00, 0.0e+00 | no stored file |
| 50x25+ppvc-mixed | SLAB-s301 | 0/50, 0.0e+00, 0.0e+00 | no stored file |
| 50x25+ppvc-mixed | SLAB-s302 | no stored file | no stored file |
| 50x25+ppvc-mixed | SLAB-s303 | no stored file | no stored file |
| 50x25+ppvc-mixed | SPT | 0/50, 0.0e+00, 0.0e+00 | 0 / 0 / 0 of 50 |
| 50x25+ppvc-mixed | LLF | 0/50, 0.0e+00, 0.0e+00 | no stored file |
| 80x25+ppvc-mixed | SLAB-s301 | 0/50, 0.0e+00, 0.0e+00 | no stored file |
| 80x25+ppvc-mixed | SLAB-s302 | no stored file | no stored file |
| 80x25+ppvc-mixed | SLAB-s303 | no stored file | no stored file |
| 80x25+ppvc-mixed | SPT | 0/50, 0.0e+00, 0.0e+00 | 0 / 0 / 0 of 50 |
| 80x25+ppvc-mixed | LLF | 0/50, 0.0e+00, 0.0e+00 | no stored file |
