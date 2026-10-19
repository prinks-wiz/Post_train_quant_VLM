## All variants (full 10,000-image CIFAR-10 test set)

| variant             | backend   | precision   | selection   |   N |   top1_acc |   agree_fp32 |   cos_fp32_mean |   cos_fp32_p01 |   lat_p50_ms |   lat_p99_ms |   peak_rss_mb | model_file_mb   |
|:--------------------|:----------|:------------|:------------|----:|-----------:|-------------:|----------------:|---------------:|-------------:|-------------:|--------------:|:----------------|
| fp32_ort_cpu        | ort_cpu   | fp32        | -           |   0 |      88.3  |       100    |          1      |         1      |        30.89 |        31.42 |           811 | 335             |
| fp32_torch_mps      | torch_mps | fp32        | -           |   0 |      88.3  |       100    |          1      |         1      |         9.97 |        10.66 |           773 | -               |
| fp16_torch_mps      | torch_mps | fp16        | -           |   0 |      88.3  |        99.92 |          1      |         0.9996 |         9.02 |         9.72 |          1030 | -               |
| int8_kcenter_n32    | ort_cpu   | int8        | kcenter     |  32 |      85.73 |        86.95 |          0.8178 |         0.711  |        13.07 |        13.54 |           634 | 85              |
| int8_kcenter_n128   | ort_cpu   | int8        | kcenter     | 128 |      85.56 |        86.51 |          0.8134 |         0.7052 |        13.09 |        13.5  |           630 | 85              |
| int8_kcenter_n512   | ort_cpu   | int8        | kcenter     | 512 |      84.26 |        85.3  |          0.8122 |         0.7011 |        13.03 |        13.33 |           646 | 85              |
| int8_random_s0_n32  | ort_cpu   | int8        | random_s0   |  32 |      86.05 |        86.86 |          0.8164 |         0.71   |        13.05 |        13.34 |           638 | 85              |
| int8_random_s0_n128 | ort_cpu   | int8        | random_s0   | 128 |      86.08 |        86.82 |          0.8173 |         0.7102 |        13.02 |        13.25 |           632 | 85              |
| int8_random_s0_n512 | ort_cpu   | int8        | random_s0   | 512 |      85.64 |        86.22 |          0.814  |         0.7094 |        13.05 |        13.53 |           641 | 85              |
| int8_random_s1_n32  | ort_cpu   | int8        | random_s1   |  32 |      85.75 |        86.41 |          0.8136 |         0.7049 |        13.02 |        13.34 |           655 | 85              |
| int8_random_s1_n128 | ort_cpu   | int8        | random_s1   | 128 |      85.89 |        86.79 |          0.8142 |         0.7074 |        13.03 |        13.31 |           647 | 85              |
| int8_random_s1_n512 | ort_cpu   | int8        | random_s1   | 512 |      85.81 |        86.57 |          0.8182 |         0.7136 |        13.07 |        13.58 |           633 | 85              |
| int8_random_s2_n32  | ort_cpu   | int8        | random_s2   |  32 |      85.76 |        86.78 |          0.8175 |         0.7151 |        13.09 |        13.59 |           650 | 85              |
| int8_random_s2_n128 | ort_cpu   | int8        | random_s2   | 128 |      85.47 |        86.25 |          0.8179 |         0.7136 |        13.07 |        13.43 |           651 | 85              |
| int8_random_s2_n512 | ort_cpu   | int8        | random_s2   | 512 |      85.58 |        86.27 |          0.8127 |         0.7073 |        13.09 |        13.46 |           646 | 85              |

## INT8: k-center vs random by N

|      N |   kcenter_acc |   random_acc_mean |   random_acc_min |   random_acc_max |   kcenter_agree |   random_agree_mean |   kcenter_cos |   random_cos_mean |
|-------:|--------------:|------------------:|-----------------:|-----------------:|----------------:|--------------------:|--------------:|------------------:|
|  32.00 |         85.73 |             85.85 |            85.75 |            86.05 |           86.95 |               86.68 |          0.82 |              0.82 |
| 128.00 |         85.56 |             85.81 |            85.47 |            86.08 |           86.51 |               86.62 |          0.81 |              0.82 |
| 512.00 |         84.26 |             85.68 |            85.58 |            85.81 |           85.30 |               86.35 |          0.81 |              0.81 |

## Calibration-set coverage (pool of 5,000 train images)

| selection   |   N |   coverage_radius |   n_classes |
|:------------|----:|------------------:|------------:|
| kcenter     |  32 |             0.701 |           8 |
| kcenter     | 128 |             0.621 |          10 |
| kcenter     | 512 |             0.537 |          10 |
| random_s0   |  32 |             0.761 |          10 |
| random_s0   | 128 |             0.733 |          10 |
| random_s0   | 512 |             0.639 |          10 |
| random_s1   |  32 |             0.726 |          10 |
| random_s1   | 128 |             0.685 |          10 |
| random_s1   | 512 |             0.657 |          10 |
| random_s2   |  32 |             0.720 |          10 |
| random_s2   | 128 |             0.698 |          10 |
| random_s2   | 512 |             0.664 |          10 |
