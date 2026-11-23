# CP-SAT incumbent at fixed solver times and best proven bounds

Mean incumbent makespan (h) over the orders of each run; n/N = orders with a solution at that time when not all have one. SLAB = per-order mean of three seeds on the same orders.

## 10x25+ppvc-mixed (SLAB 211.3 h; best CP-SAT bound 202.0 h; best known 203.2 h; SLAB/bound 1.047; SLAB/best known 1.040)

SLAB time on one CPU core: 0.73 s

| run | n | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s | 600 s | 1800 s | at SLAB time | final | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 100 | 216.0 (SLAB 211.3) | 204.4 (SLAB 211.3) | 203.8 (SLAB 211.3) | 203.5 (SLAB 211.3) | 203.4 (SLAB 211.3) | 203.2 (SLAB 211.3) | -- | -- | 228.2 (SLAB 211.3) | 203.2 | 83 |
| warm-spt | 100 | 206.5 (SLAB 211.3) | 204.0 (SLAB 211.3) | 203.6 (SLAB 211.3) | 203.4 (SLAB 211.3) | 203.3 (SLAB 211.3) | 203.2 (SLAB 211.3) | -- | -- | 207.5 (SLAB 211.3) | 203.2 | 85 |

## 20x25+ppvc-mixed (SLAB 255.0 h; best CP-SAT bound 210.7 h; best known 250.7 h; SLAB/bound 1.211; SLAB/best known 1.017)

SLAB time on one CPU core: 1.72 s

| run | n | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s | 600 s | 1800 s | at SLAB time | final | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 50 | 371.6 (SLAB 255.0) | 300.1 (SLAB 255.0) | 278.7 (SLAB 255.0) | 265.6 (SLAB 255.0) | 261.2 (SLAB 255.0) | 253.7 (SLAB 255.0) | -- | -- | 355.8 (SLAB 255.0) | 253.7 | 0 |
| warm-spt | 50 | 263.1 (SLAB 255.0) | 261.5 (SLAB 255.0) | 260.7 (SLAB 255.0) | 259.0 (SLAB 255.0) | 257.5 (SLAB 255.0) | 253.4 (SLAB 255.0) | -- | -- | 263.0 (SLAB 255.0) | 253.4 | 0 |

## 30x25+ppvc-mixed (SLAB 308.7 h; best CP-SAT bound 210.8 h; best known 305.0 h; SLAB/bound 1.465; SLAB/best known 1.012)

SLAB time on one CPU core: 2.96 s

| run | n | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s | 600 s | 1800 s | at SLAB time | final | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 20 | 515.2 (SLAB 308.7) | 462.9 (SLAB 308.7) | 402.4 (SLAB 308.7) | 347.2 (SLAB 308.7) | 337.2 (SLAB 308.7) | 318.9 (SLAB 308.7) | -- | -- | 490.4 (SLAB 308.7) | 318.9 | 0 |
| warm-spt | 20 | 322.8 (SLAB 308.7) | 322.4 (SLAB 308.7) | 322.1 (SLAB 308.7) | 321.1 (SLAB 308.7) | 320.5 (SLAB 308.7) | 318.5 (SLAB 308.7) | -- | -- | 322.4 (SLAB 308.7) | 318.5 | 0 |

## 50x25+ppvc-mixed (SLAB 414.9 h; best CP-SAT bound 247.4 h; best known 408.6 h; SLAB/bound 1.678; SLAB/best known 1.015)

SLAB time on one CPU core: 6.22 s

| run | n | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s | 600 s | 1800 s | at SLAB time | final | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 50 | 792.7 (SLAB 414.9) | 753.5 (SLAB 414.9) | 717.4 (SLAB 414.9) | 555.6 (SLAB 414.9) | 524.8 (SLAB 414.9) | 480.0 (SLAB 414.9) | 464.2 (SLAB 414.9) | -- | 747.4 (SLAB 414.9) | 464.2 | 0 |
| warm-spt | 50 | 448.9 (SLAB 414.9) | 447.5 (SLAB 414.9) | 447.4 (SLAB 414.9) | 446.8 (SLAB 414.9) | 446.1 (SLAB 414.9) | 444.7 (SLAB 414.9) | 444.1 (SLAB 414.9) | -- | 447.5 (SLAB 414.9) | 444.1 | 0 |
| warm-policy | 50 | 412.4 (SLAB 414.9) | 411.4 (SLAB 414.9) | 411.4 (SLAB 414.9) | 410.7 (SLAB 414.9) | 410.4 (SLAB 414.9) | 409.7 (SLAB 414.9) | 409.4 (SLAB 414.9) | -- | 411.4 (SLAB 414.9) | 409.4 | 0 |
| cold-1800 | 20 | 787.3 (SLAB 414.5) | 745.5 (SLAB 414.5) | 709.4 (SLAB 414.5) | 554.7 (SLAB 414.5) | 524.9 (SLAB 414.5) | 480.1 (SLAB 414.5) | 463.6 (SLAB 414.5) | 444.6 (SLAB 414.5) | 740.5 (SLAB 414.5) | 444.6 | 0 |
| warm-spt-1800 | 20 | 448.4 (SLAB 414.5) | 446.7 (SLAB 414.5) | 446.6 (SLAB 414.5) | 445.6 (SLAB 414.5) | 444.8 (SLAB 414.5) | 442.6 (SLAB 414.5) | 441.9 (SLAB 414.5) | 439.6 (SLAB 414.5) | 446.7 (SLAB 414.5) | 439.6 | 0 |

## 80x25+ppvc-mixed (SLAB 570.3 h; best CP-SAT bound 313.0 h; best known 562.0 h; SLAB/bound 1.823; SLAB/best known 1.015)

SLAB time on one CPU core: 12.98 s

| run | n | 1 s | 5 s | 10 s | 30 s | 60 s | 300 s | 600 s | 1800 s | at SLAB time | final | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 50 | 0/50 solved | 1146.5 (SLAB 570.3) | 1139.1 (SLAB 570.3) | 917.0 (SLAB 570.3) | 823.6 (SLAB 570.3) | 749.8 (SLAB 570.3) | 724.5 (SLAB 570.3) | -- | 1132.8 (SLAB 570.3) | 724.5 | 0 |
| warm-spt | 50 | 620.3 (SLAB 570.3) | 619.1 (SLAB 570.3) | 619.1 (SLAB 570.3) | 619.0 (SLAB 570.3) | 618.6 (SLAB 570.3) | 617.8 (SLAB 570.3) | 617.1 (SLAB 570.3) | -- | 619.1 (SLAB 570.3) | 617.1 | 0 |
| warm-policy | 50 | 566.8 (SLAB 570.3) | 565.6 (SLAB 570.3) | 565.4 (SLAB 570.3) | 565.1 (SLAB 570.3) | 564.7 (SLAB 570.3) | 564.2 (SLAB 570.3) | 563.9 (SLAB 570.3) | -- | 565.4 (SLAB 570.3) | 563.9 | 0 |
| cold-1800 | 20 | 0/20 solved | 1147.8 (SLAB 568.6) | 1140.0 (SLAB 568.6) | 912.0 (SLAB 568.6) | 830.0 (SLAB 568.6) | 746.0 (SLAB 568.6) | 719.0 (SLAB 568.6) | 672.0 (SLAB 568.6) | 1131.2 (SLAB 568.6) | 672.0 | 0 |
| warm-spt-1800 | 20 | 620.2 (SLAB 568.6) | 618.8 (SLAB 568.6) | 618.8 (SLAB 568.6) | 618.7 (SLAB 568.6) | 618.5 (SLAB 568.6) | 617.6 (SLAB 568.6) | 617.0 (SLAB 568.6) | 616.0 (SLAB 568.6) | 618.8 (SLAB 568.6) | 616.0 | 0 |
| warm-spt-w16 | 20 | 620.2 (SLAB 568.6) | 618.4 (SLAB 568.6) | 618.4 (SLAB 568.6) | 618.0 (SLAB 568.6) | 617.5 (SLAB 568.6) | 616.3 (SLAB 568.6) | 615.6 (SLAB 568.6) | -- | 618.4 (SLAB 568.6) | 615.6 | 0 |

