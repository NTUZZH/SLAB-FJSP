# PPVC evaluation summary — 10x25+ppvc-mixed+a1lfrew-2k-s305

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lfrew-2k-s305.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9, bound_lag_free_reward=True (training reward only)
- CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 212.4 | 9.4 | 1.8419 | 100/100 | 4.55% (100) |
