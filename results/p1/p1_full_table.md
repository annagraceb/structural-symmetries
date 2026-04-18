# P1 Full Results Table
_n_models = 33, k_values = [4, 8, 12, 16], n_random_trials = 100, d_model = 64_
## Per-site, per-k mean ablation drop (across 33 models)

### layer1_result_0
- d = 64, K_pca = 32, PCA variance retained = 0.998, denom condition number = 1.60e+03, ridge = 7.06e-09
- Top-5 ratio eigvals (shared): ['0.8991', '0.8382', '0.7800', '0.6940', '0.6664']
- Bottom-5 ratio eigvals (anti-shared): ['0.1167', '0.1106', '0.0950', '0.0788', '0.0655']

| k | shared | complement | anti_ortho | anti_raw | random | shared trace | comp trace |
|---|---|---|---|---|---|---|---|
| 4 | 0.1975 | 0.4409 | 0.0006 | 0.0000 | 0.0030 | 8.44 | 11.56 |
| 8 | 0.3761 | 0.2507 | 0.0533 | 0.0000 | 0.0118 | 11.15 | 10.73 |
| 12 | 0.5512 | 0.1799 | 0.0668 | 0.0001 | 0.0281 | 13.01 | 9.36 |
| 16 | 0.6705 | 0.1056 | 0.1056 | 0.0004 | 0.0525 | 15.27 | 7.31 |

#### Per-variance hurt (drop / projection_trace) at primary k = 8:

- **shared**: drop = 0.3761, trace = 11.1500, drop/trace = 0.0337
- **complement_top_k**: drop = 0.2507, trace = 10.7284, drop/trace = 0.0234
- **anti_shared_ortho**: drop = 0.0533, trace = 4.4161, drop/trace = 0.0121
- **anti_shared_raw**: drop = 0.0000, trace = 0.2015, drop/trace = 0.0002

### layer2_equals
- d = 64, K_pca = 32, PCA variance retained = 0.997, denom condition number = 1.89e+03, ridge = 9.85e-09
- Top-5 ratio eigvals (shared): ['0.8577', '0.7047', '0.6802', '0.6248', '0.6065']
- Bottom-5 ratio eigvals (anti-shared): ['0.1260', '0.1116', '0.1102', '0.1007', '0.0943']

| k | shared | complement | anti_ortho | anti_raw | random | shared trace | comp trace |
|---|---|---|---|---|---|---|---|
| 4 | 0.0071 | 0.0150 | 0.0000 | -0.0000 | 0.0001 | 11.97 | 15.73 |
| 8 | 0.0195 | 0.0280 | 0.0002 | 0.0000 | 0.0008 | 16.42 | 13.79 |
| 12 | 0.0416 | 0.0242 | 0.0069 | 0.0004 | 0.0020 | 19.50 | 11.62 |
| 16 | 0.1912 | 0.0223 | 0.0223 | 0.0039 | 0.0037 | 23.78 | 7.75 |

#### Per-variance hurt (drop / projection_trace) at primary k = 8:

- **complement_top_k**: drop = 0.0280, trace = 13.7921, drop/trace = 0.0020
- **shared**: drop = 0.0195, trace = 16.4152, drop/trace = 0.0012
- **anti_shared_ortho**: drop = 0.0002, trace = 3.1988, drop/trace = 0.0001
- **anti_shared_raw**: drop = 0.0000, trace = 0.4161, drop/trace = 0.0000

### layer2_result_0
- d = 64, K_pca = 32, PCA variance retained = 0.996, denom condition number = 8.63e+02, ridge = 1.10e-08
- Top-5 ratio eigvals (shared): ['0.8983', '0.8880', '0.7845', '0.7742', '0.7345']
- Bottom-5 ratio eigvals (anti-shared): ['0.1824', '0.1646', '0.1458', '0.1300', '0.0941']

| k | shared | complement | anti_ortho | anti_raw | random | shared trace | comp trace |
|---|---|---|---|---|---|---|---|
| 4 | 0.2647 | 0.1533 | 0.0001 | 0.0000 | 0.0008 | 16.89 | 14.30 |
| 8 | 0.3120 | 0.1001 | 0.0004 | 0.0000 | 0.0026 | 20.11 | 13.97 |
| 12 | 0.4149 | 0.0514 | 0.0012 | 0.0000 | 0.0063 | 23.95 | 10.87 |
| 16 | 0.5228 | 0.0174 | 0.0174 | 0.0001 | 0.0122 | 26.58 | 8.53 |

#### Per-variance hurt (drop / projection_trace) at primary k = 8:

- **shared**: drop = 0.3120, trace = 20.1144, drop/trace = 0.0155
- **complement_top_k**: drop = 0.1001, trace = 13.9700, drop/trace = 0.0072
- **anti_shared_ortho**: drop = 0.0004, trace = 2.2804, drop/trace = 0.0002
- **anti_shared_raw**: drop = 0.0000, trace = 0.3270, drop/trace = 0.0000

### layer3_plus *(CONTROL)*
- d = 64, K_pca = 32, PCA variance retained = 0.990, denom condition number = 2.58e+02, ridge = 1.43e-08
- Top-5 ratio eigvals (shared): ['0.3609', '0.2571', '0.2379', '0.2129', '0.2112']
- Bottom-5 ratio eigvals (anti-shared): ['0.0431', '0.0405', '0.0397', '0.0354', '0.0339']

| k | shared | complement | anti_ortho | anti_raw | random | shared trace | comp trace |
|---|---|---|---|---|---|---|---|
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.29 | 29.48 |
| 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.65 | 37.48 |
| 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.03 | 39.73 |
| 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.92 | 39.70 |

#### Per-variance hurt (drop / projection_trace) at primary k = 8:

- **shared**: drop = 0.0000, trace = 2.6524, drop/trace = 0.0000
- **complement_top_k**: drop = 0.0000, trace = 37.4839, drop/trace = 0.0000
- **anti_shared_ortho**: drop = 0.0000, trace = 3.2844, drop/trace = 0.0000
- **anti_shared_raw**: drop = 0.0000, trace = 1.7884, drop/trace = 0.0000

## Tikhonov ridge stability check

Single site (layer1_result_0), single k = 10 (eps-sweep config). Tests whether the eigenproblem solution is stable under different ridge regularization scales.

| ε scale | shared drop | anti_shared_raw drop | comp drop | denom cond # | ridge |
|---|---|---|---|---|---|
| 1e-05 | 0.4481 | 0.0001 | 0.2196 | 1.60e+03 | 7.06e-06 |
| 0.0001 | 0.4532 | 0.0001 | 0.2170 | 1.58e+03 | 7.06e-05 |
| 0.001 | 0.4977 | 0.0001 | 0.1923 | 1.43e+03 | 7.06e-04 |
