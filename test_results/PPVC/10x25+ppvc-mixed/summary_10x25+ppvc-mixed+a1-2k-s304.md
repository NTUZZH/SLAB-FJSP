# PPVC evaluation summary — 10x25+ppvc-mixed+a1-2k-s304

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1-2k-s304.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 212.0 | 9.5 | 1.8702 | 100/100 | 4.38% (100) |
