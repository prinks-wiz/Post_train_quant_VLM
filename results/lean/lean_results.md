## Lean CPU path: every variant (batch-1, 3 repeats, median; full-test accuracy)

| spec                                 | precision   | head_kind   |   acc |   frac_stage_b |   lat_p50_ms |   lat_p99_ms |   lat_mean_ms |   model_ram_mb |   peak_rss_mb |   rss_base_mb |   model_file_mb |
|:-------------------------------------|:------------|:------------|------:|---------------:|-------------:|-------------:|--------------:|---------------:|--------------:|--------------:|----------------:|
| fp32_d12_aligned_r128                | fp32        | aligned     |  84.6 |            0.0 |         10.8 |         12.1 |          10.9 |          364.1 |         449.1 |          74.5 |           333.6 |
| fp32_d12_aligned_r128-224_f10        | fp32        | aligned     |  87.3 |            9.7 |         10.8 |         34.8 |          13.1 |          365.9 |         454.8 |          74.4 |           333.6 |
| fp32_d12_aligned_r160                | fp32        | aligned     |  87.4 |            0.0 |         14.0 |         15.6 |          14.1 |          364.5 |         451.5 |          74.0 |           333.6 |
| fp32_d12_aligned_r128-224_f20        | fp32        | aligned     |  88.4 |           22.7 |         10.8 |         34.9 |          16.0 |          366.4 |         455.3 |          74.3 |           333.6 |
| fp32_d12_aligned_r160-224_f10        | fp32        | aligned     |  88.7 |           12.0 |         14.1 |         38.8 |          17.0 |          366.1 |         455.5 |          74.1 |           333.6 |
| fp32_d12_aligned_r128-224_f30        | fp32        | aligned     |  88.8 |           34.0 |         10.9 |         35.7 |          18.7 |          366.3 |         455.3 |          74.5 |           333.6 |
| fp32_d12_aligned_r160-224_f20        | fp32        | aligned     |  89.1 |           21.7 |         14.0 |         38.3 |          19.0 |          365.9 |         454.7 |          74.4 |           333.6 |
| fp32_d12_aligned_r160-224_f30        | fp32        | aligned     |  88.9 |           31.7 |         14.0 |         38.5 |          21.3 |          366.3 |         454.7 |          74.3 |           333.6 |
| fp32_d12_aligned_r224                | fp32        | aligned     |  88.4 |            0.0 |         22.8 |         25.1 |          23.0 |          364.5 |         454.2 |          74.3 |           333.6 |
| int4_d12_aligned_r128                | int4        | aligned     |  81.6 |            0.0 |         11.2 |         11.8 |          11.2 |           92.2 |         178.1 |          74.4 |            60.2 |
| int4_d12_aligned_r160                | int4        | aligned     |  84.9 |            0.0 |         14.4 |         15.4 |          14.4 |          132.4 |         216.8 |          74.1 |            60.2 |
| int4_d12_aligned_r128-224_f10        | int4        | aligned     |  84.5 |           15.3 |         11.2 |         34.8 |          14.7 |           92.9 |         183.2 |          74.1 |            60.2 |
| int4_d12_aligned_r128-224_f20        | int4        | aligned     |  86.1 |           21.3 |         11.2 |         35.3 |          16.1 |           93.1 |         183.2 |          74.2 |            60.2 |
| int4_d12_aligned_r160-224_f10        | int4        | aligned     |  86.8 |           10.7 |         14.4 |         38.3 |          16.9 |           94.1 |         183.0 |          74.1 |            60.2 |
| int4_d12_aligned_r128-224_f30        | int4        | aligned     |  86.5 |           30.0 |         11.2 |         35.8 |          18.1 |           93.3 |         183.2 |          74.1 |            60.2 |
| int4_d12_aligned_r160-224_f20        | int4        | aligned     |  87.2 |           23.0 |         14.4 |         38.4 |          19.7 |           94.6 |         183.1 |          74.2 |            60.2 |
| int4_d12_aligned_r160-224_f30        | int4        | aligned     |  87.0 |           34.7 |         14.5 |         38.8 |          22.4 |           93.3 |         183.2 |          74.2 |            60.2 |
| int4_d12_aligned_r224                | int4        | aligned     |  86.4 |            0.0 |         22.9 |         24.4 |          23.0 |           91.8 |         182.4 |          74.2 |            60.2 |
| int8_dyn_d12_aligned_r128            | int8_dyn    | aligned     |  83.9 |            0.0 |          5.0 |          5.3 |           5.0 |          123.8 |         202.3 |          74.2 |            91.1 |
| int8_dyn_d12_aligned_r128-224_f10    | int8_dyn    | aligned     |  86.9 |           10.7 |          5.1 |         15.6 |           6.2 |          124.8 |         206.9 |          74.0 |            91.1 |
| int8_dyn_d12_aligned_r160            | int8_dyn    | aligned     |  87.6 |            0.0 |          6.6 |          7.1 |           6.7 |          123.5 |         203.1 |          74.2 |            91.1 |
| int8_dyn_d12_aligned_r128-224_f20    | int8_dyn    | aligned     |  88.7 |           25.0 |          5.1 |         15.8 |           7.7 |          125.0 |         207.0 |          74.3 |            91.1 |
| int8_dyn_d12_aligned_r160-224_f10    | int8_dyn    | aligned     |  89.1 |           11.3 |          6.7 |         17.3 |           7.9 |          124.8 |         207.1 |          74.0 |            91.1 |
| int8_dyn_d12_aligned_r128-224_f30    | int8_dyn    | aligned     |  89.2 |           33.3 |          5.1 |         16.8 |           8.5 |          127.3 |         208.5 |          74.2 |            91.1 |
| int8_dyn_d12_aligned_r160-224_f20    | int8_dyn    | aligned     |  89.4 |           21.0 |          6.7 |         17.9 |           8.9 |          125.2 |         207.5 |          74.1 |            91.1 |
| int8_dyn_d12_aligned_r160-224_f30    | int8_dyn    | aligned     |  89.4 |           31.0 |          6.7 |         17.6 |           9.9 |          127.1 |         216.5 |          74.3 |            91.1 |
| int8_dyn_d12_aligned_r224            | int8_dyn    | aligned     |  89.1 |            0.0 |         10.4 |         11.6 |          10.4 |          123.6 |         212.7 |          74.2 |            91.1 |
| int8_static_d12_aligned_r128         | int8_static | aligned     |  82.7 |            0.0 |          5.4 |          6.2 |           5.4 |          113.7 |         191.8 |          74.1 |            84.4 |
| int8_static_d12_aligned_r128-224_f10 | int8_static | aligned     |  85.7 |           13.3 |          5.4 |         16.9 |           6.9 |          116.5 |         196.4 |          74.2 |            84.4 |
| int8_static_d12_aligned_r160         | int8_static | aligned     |  86.6 |            0.0 |          7.0 |          8.4 |           7.0 |          117.6 |         197.9 |          74.1 |            84.4 |
| int8_static_d12_aligned_r128-224_f20 | int8_static | aligned     |  87.7 |           23.7 |          5.4 |         16.9 |           8.0 |          114.5 |         196.8 |          74.2 |            84.4 |
| int8_static_d12_aligned_r160-224_f10 | int8_static | aligned     |  88.4 |           12.3 |          6.9 |         18.5 |           8.3 |          115.3 |         197.6 |          74.0 |            84.4 |
| int8_static_d12_aligned_r160-224_f20 | int8_static | aligned     |  88.7 |           21.3 |          6.9 |         18.7 |           9.3 |          120.7 |         208.5 |          74.1 |            84.4 |
| int8_static_d12_aligned_r128-224_f30 | int8_static | aligned     |  88.4 |           35.3 |          5.4 |         17.1 |           9.3 |          120.7 |         202.3 |          74.3 |            84.4 |
| int8_static_d12_aligned_r160-224_f30 | int8_static | aligned     |  88.6 |           36.7 |          7.0 |         19.6 |          11.1 |          115.0 |         205.5 |          74.0 |            84.4 |
| int8_static_d12_aligned_r224         | int8_static | aligned     |  88.0 |            0.0 |         11.0 |         11.8 |          11.1 |          117.0 |         206.7 |          74.3 |            84.4 |
| int8_dyn_d12_probe_r128              | int8_dyn    | probe       |  88.0 |            0.0 |          5.0 |          5.5 |           5.0 |          120.5 |         199.2 |          74.2 |            91.1 |
| int8_dyn_d12_probe_r160              | int8_dyn    | probe       |  91.3 |            0.0 |          6.6 |          7.4 |           6.6 |          120.6 |         200.2 |          74.4 |            91.1 |
| int8_dyn_d12_probe_r128-224_f10      | int8_dyn    | probe       |  93.1 |           22.0 |          5.0 |         16.2 |           7.3 |          123.5 |         205.5 |          74.3 |            91.1 |
| int8_dyn_d12_probe_r128-224_f20      | int8_dyn    | probe       |  93.5 |           25.0 |          5.0 |         15.8 |           7.6 |          121.0 |         205.5 |          74.0 |            91.1 |
| int8_dyn_d12_probe_r128-224_f30      | int8_dyn    | probe       |  93.7 |           32.3 |          5.0 |         16.0 |           8.3 |          124.5 |         206.2 |          74.1 |            91.1 |
| int8_dyn_d12_probe_r160-224_f10      | int8_dyn    | probe       |  93.7 |           20.0 |          6.6 |         17.6 |           8.7 |          120.5 |         204.8 |          74.2 |            91.1 |
| int8_dyn_d12_probe_r160-224_f20      | int8_dyn    | probe       |  93.8 |           27.3 |          6.7 |         17.7 |           9.5 |          121.0 |         210.4 |          74.4 |            91.1 |
| int8_dyn_d12_probe_r160-224_f30      | int8_dyn    | probe       |  93.8 |           34.0 |          6.7 |         18.6 |          10.2 |          120.8 |         211.5 |          74.6 |            91.1 |
| int8_dyn_d12_probe_r224              | int8_dyn    | probe       |  93.6 |            0.0 |         10.3 |         11.5 |          10.4 |          120.5 |         210.9 |          74.2 |            91.1 |

## Round 2 (RAM): INT4 with INT8 compute (int4_a4), OpenCV-free preprocessing (__np), ORT weight prepacking off (__np_nopack)

| spec                                         | precision   | head_kind   |   acc |   frac_stage_b |   lat_p50_ms |   lat_p99_ms |   lat_mean_ms |   model_ram_mb |   peak_rss_mb |   rss_base_mb |   model_file_mb |
|:---------------------------------------------|:------------|:------------|------:|---------------:|-------------:|-------------:|--------------:|---------------:|--------------:|--------------:|----------------:|
| int4_a4_d12_aligned_r128                     | int4_a4     | aligned     |  82.0 |            0.0 |          5.3 |          6.1 |           5.3 |          110.3 |         189.2 |          74.6 |            60.2 |
| int4_a4_d12_aligned_r128-224_f10             | int4_a4     | aligned     |  84.7 |           12.7 |          5.4 |         17.7 |           6.9 |          114.0 |         196.5 |          74.8 |            60.2 |
| int4_a4_d12_aligned_r160                     | int4_a4     | aligned     |  85.3 |            0.0 |          7.3 |          8.0 |           7.3 |          110.1 |         190.0 |          74.7 |            60.2 |
| int4_a4_d12_aligned_r128-224_f20             | int4_a4     | aligned     |  86.4 |           23.3 |          5.3 |         17.5 |           8.0 |          112.3 |         194.7 |          74.5 |            60.2 |
| int4_a4_d12_aligned_r128-224_f20__np         | int4_a4     | aligned     |  86.4 |           23.3 |          5.4 |         17.8 |           8.1 |          115.5 |         172.7 |          49.9 |            60.2 |
| int4_a4_d12_aligned_r160-224_f10             | int4_a4     | aligned     |  87.1 |           11.7 |          7.3 |         19.4 |           8.7 |          112.1 |         195.6 |          74.8 |            60.2 |
| int4_a4_d12_aligned_r128-224_f30             | int4_a4     | aligned     |  86.8 |           31.7 |          5.4 |         17.8 |           9.1 |          116.6 |         198.9 |          74.4 |            60.2 |
| int4_a4_d12_aligned_r160-224_f20             | int4_a4     | aligned     |  87.7 |           22.7 |          7.3 |         19.7 |          10.0 |          113.0 |         201.5 |          74.8 |            60.2 |
| int4_a4_d12_aligned_r160-224_f30             | int4_a4     | aligned     |  87.5 |           35.3 |          7.4 |         20.3 |          11.4 |          113.8 |         203.1 |          74.5 |            60.2 |
| int4_a4_d12_aligned_r224                     | int4_a4     | aligned     |  86.9 |            0.0 |         11.6 |         12.7 |          11.6 |          110.2 |         199.5 |          74.6 |            60.2 |
| int4_a4_d12_aligned_r224__np                 | int4_a4     | aligned     |  86.7 |            0.0 |         11.7 |         12.9 |          11.8 |          105.9 |         170.8 |          49.7 |            60.2 |
| int4_a4_d12_aligned_r128-224_f20__np_nopack  | int4_a4     | aligned     |  86.4 |           22.7 |         32.7 |         79.6 |          43.0 |           93.6 |         158.0 |          49.9 |            60.2 |
| int4_a4_d12_aligned_r224__np_nopack          | int4_a4     | aligned     |  86.7 |            0.0 |         44.5 |         46.9 |          44.7 |           92.9 |         161.9 |          49.5 |            60.2 |
| int8_dyn_d12_aligned_r160-224_f10__np        | int8_dyn    | aligned     |  88.8 |           10.0 |          6.7 |         17.4 |           7.7 |          125.0 |         183.2 |          49.5 |            91.1 |
| int8_dyn_d12_aligned_r128-224_f20__np        | int8_dyn    | aligned     |  88.4 |           25.3 |          5.2 |         16.0 |           7.8 |          124.8 |         181.8 |          49.7 |            91.1 |
| int8_dyn_d12_aligned_r160-224_f10__np_nopack | int8_dyn    | aligned     |  88.8 |           10.0 |          8.2 |         20.8 |           9.4 |          124.7 |         182.1 |          49.6 |            91.1 |
| int8_dyn_d12_aligned_r128-224_f20__np_nopack | int8_dyn    | aligned     |  88.4 |           25.3 |          6.5 |         18.9 |           9.5 |          124.3 |         187.7 |          49.8 |            91.1 |
| int8_dyn_d12_aligned_r224__np                | int8_dyn    | aligned     |  88.8 |            0.0 |         10.4 |         11.2 |          10.4 |          123.2 |         187.8 |          49.7 |            91.1 |
| int8_dyn_d12_aligned_r224__np_nopack         | int8_dyn    | aligned     |  88.8 |            0.0 |         11.9 |         13.4 |          11.9 |          122.6 |         187.1 |          49.6 |            91.1 |
| int8_dyn_d12_probe_r128-224_f10__np          | int8_dyn    | probe       |  93.2 |           21.0 |          5.1 |         16.0 |           7.2 |          126.6 |         183.5 |          49.7 |            91.1 |
| int8_dyn_d12_probe_r128-224_f10__np_nopack   | int8_dyn    | probe       |  93.2 |           21.0 |          6.5 |         19.4 |           9.0 |          119.5 |         178.5 |          49.8 |            91.1 |

## Ablations: intra-op threads (-1 = 6 P-cores) and ORT memory arena (lean_mem=False = arena on)

| spec                              |   threads | lean_mem   |   lat_p50_ms |   lat_p99_ms |   model_ram_mb |   peak_rss_mb |
|:----------------------------------|----------:|:-----------|-------------:|-------------:|---------------:|--------------:|
| int8_dyn_d12_aligned_r128-224_f20 |        -1 | False      |          5.0 |         16.0 |          136.7 |         215.9 |
| int8_dyn_d12_aligned_r128-224_f20 |        -1 | True       |          5.1 |         15.8 |          125.0 |         207.0 |
| int8_dyn_d12_aligned_r128-224_f20 |         1 | True       |         13.9 |         52.3 |          125.1 |         211.8 |
| int8_dyn_d12_aligned_r128-224_f20 |         2 | True       |          8.4 |         29.9 |          124.8 |         212.2 |
| int8_dyn_d12_aligned_r128-224_f20 |         4 | True       |          6.2 |         19.6 |          126.8 |         206.8 |
| int8_dyn_d12_aligned_r224         |        -1 | False      |         10.3 |         11.7 |          137.2 |         224.4 |
| int8_dyn_d12_aligned_r224         |        -1 | True       |         10.4 |         11.6 |          123.6 |         212.7 |
| int8_dyn_d12_aligned_r224         |         1 | True       |         37.9 |         38.5 |          123.0 |         210.2 |
| int8_dyn_d12_aligned_r224         |         2 | True       |         21.0 |         21.5 |          123.8 |         211.8 |
| int8_dyn_d12_aligned_r224         |         4 | True       |         13.3 |         13.6 |          124.3 |         211.3 |

## Online replay (600 frames, 3 repeats, median): lean finalists vs the INT8->FP16 router

| spec                                           | source   |   p50_ms |   p95_ms |   p99_ms |   miss_rate |   acc_all |   frac_stage_b |   peak_rss_mb |
|:-----------------------------------------------|:---------|---------:|---------:|---------:|------------:|----------:|---------------:|--------------:|
| int4_a4_d12_aligned_r128-224_f20__np           | bursty   |      6.7 |     24.2 |     33.5 |         1.3 |      85.8 |           21.7 |         188.5 |
| int4_a4_d12_aligned_r224__np                   | bursty   |     12.2 |     31.9 |     34.0 |         3.0 |      87.3 |            0.0 |         189.2 |
| int4_a4_d12_aligned_r128-224_f20__np+deadline  | bursty   |      6.7 |     22.5 |     30.3 |         0.5 |      85.7 |           20.8 |         189.5 |
| int4_d12_aligned_r128-224_f20                  | bursty   |     38.3 |     89.3 |    117.7 |        58.7 |      85.2 |           22.3 |         200.7 |
| int8_dyn_d12_probe_r128-224_f10__np+deadline   | bursty   |      5.8 |     20.6 |     26.9 |         0.3 |      91.2 |           18.8 |         201.2 |
| int8_dyn_d12_aligned_r224__np_nopack           | bursty   |     12.7 |     30.8 |     34.1 |         2.2 |      87.5 |            0.0 |         203.7 |
| int8_dyn_d12_aligned_r128-224_f20__np          | bursty   |      6.1 |     21.0 |     30.1 |         0.8 |      87.3 |           22.3 |         204.2 |
| int8_dyn_d12_aligned_r128-224_f20__np+deadline | bursty   |      6.2 |     20.0 |     26.4 |         0.2 |      87.5 |           21.5 |         211.1 |
| int8_dyn_d12_aligned_r128-224_f20              | bursty   |      6.5 |     22.1 |     33.3 |         1.0 |      87.3 |           23.0 |         230.3 |
| int8_dyn_d12_aligned_r128-224_f20+deadline     | bursty   |      6.2 |     20.0 |     24.4 |         0.2 |      87.3 |           22.5 |         230.5 |
| int8_dyn_d12_aligned_r160-224_f10              | bursty   |      7.2 |     24.5 |     37.6 |         1.3 |      89.5 |            9.7 |         230.8 |
| int8_static_d12_aligned_r128-224_f20           | bursty   |      6.8 |     25.1 |     37.8 |         2.3 |      88.2 |           21.8 |         231.4 |
| int8_dyn_d12_aligned_r224                      | bursty   |     10.8 |     29.2 |     32.7 |         0.8 |      88.2 |            0.0 |         233.5 |
| int8_dyn_d12_probe_r128-224_f10                | bursty   |      5.8 |     22.6 |     35.5 |         1.3 |      92.0 |           19.7 |         236.7 |
| stream/int8_only (torch+MPS)                   | bursty   |     11.4 |     30.4 |     33.4 |         1.7 |      84.2 |            0.0 |        1011.0 |
| stream/route_naive (torch+MPS)                 | bursty   |     11.9 |     36.3 |     51.6 |         8.5 |      86.7 |            9.5 |        1012.2 |
| stream/fp16_only (torch+MPS)                   | bursty   |     12.2 |     30.2 |     34.1 |         1.5 |      87.5 |          100.0 |        1012.4 |
| int4_a4_d12_aligned_r128-224_f20__np+deadline  | fixed30  |      7.6 |     19.7 |     20.8 |         0.0 |      85.8 |           21.7 |         186.6 |
| int4_a4_d12_aligned_r128-224_f20__np           | fixed30  |      7.5 |     19.5 |     20.9 |         0.0 |      85.8 |           21.7 |         187.8 |
| int4_d12_aligned_r128-224_f20                  | fixed30  |     13.3 |     36.7 |     39.3 |        22.3 |      85.2 |           22.3 |         200.4 |
| int8_dyn_d12_probe_r128-224_f10__np+deadline   | fixed30  |      7.3 |     17.8 |     19.4 |         0.0 |      91.3 |           20.0 |         202.1 |
| int8_dyn_d12_aligned_r224__np_nopack           | fixed30  |     13.5 |     14.6 |     16.2 |         0.0 |      87.5 |            0.0 |         202.5 |
| int8_dyn_d12_aligned_r128-224_f20__np          | fixed30  |      7.3 |     18.1 |     19.5 |         0.0 |      87.3 |           22.3 |         206.2 |
| int4_a4_d12_aligned_r224__np                   | fixed30  |     13.7 |     14.5 |     16.3 |         0.0 |      87.3 |            0.0 |         207.7 |
| int8_static_d12_aligned_r128-224_f20           | fixed30  |      7.7 |     19.3 |     20.6 |         0.0 |      88.2 |           21.8 |         219.3 |
| int8_dyn_d12_probe_r128-224_f10                | fixed30  |      7.2 |     17.7 |     19.2 |         0.0 |      92.0 |           19.7 |         225.9 |
| int8_dyn_d12_aligned_r128-224_f20__np+deadline | fixed30  |      7.3 |     18.0 |     18.9 |         0.0 |      87.3 |           22.3 |         229.2 |
| int8_dyn_d12_aligned_r128-224_f20              | fixed30  |      7.2 |     18.0 |     19.2 |         0.0 |      87.3 |           23.0 |         230.0 |
| int8_dyn_d12_aligned_r224                      | fixed30  |     12.4 |     13.5 |     14.9 |         0.0 |      88.2 |            0.0 |         241.9 |
| int8_dyn_d12_aligned_r128-224_f20+deadline     | fixed30  |      7.3 |     17.9 |     19.0 |         0.0 |      87.3 |           23.0 |         246.7 |
| int8_dyn_d12_aligned_r160-224_f10              | fixed30  |      8.8 |     18.9 |     19.8 |         0.0 |      89.5 |            9.7 |         255.7 |
| stream/route_naive (torch+MPS)                 | fixed30  |     12.6 |     32.4 |     34.9 |         2.8 |      86.7 |            9.5 |        1010.3 |
| stream/int8_only (torch+MPS)                   | fixed30  |     11.9 |     13.3 |     14.9 |         0.0 |      84.2 |            0.0 |        1011.7 |
| stream/fp16_only (torch+MPS)                   | fixed30  |     17.1 |     22.8 |     23.9 |         0.0 |      87.5 |          100.0 |        1018.4 |
| int4_a4_d12_aligned_r224__np                   | fixed60  |     11.7 |     13.9 |     16.8 |         1.2 |      87.3 |            0.0 |         185.9 |
| int4_a4_d12_aligned_r128-224_f20__np+deadline  | fixed60  |      5.6 |      6.4 |     16.8 |         1.5 |      81.7 |            1.5 |         187.9 |
| int4_a4_d12_aligned_r128-224_f20__np           | fixed60  |      5.6 |     18.3 |     20.1 |        21.7 |      85.8 |           21.7 |         188.8 |
| int8_dyn_d12_probe_r128-224_f10__np+deadline   | fixed60  |      5.2 |     15.8 |     17.1 |         2.3 |      91.2 |           18.7 |         200.3 |
| int8_dyn_d12_aligned_r224__np_nopack           | fixed60  |     11.9 |     14.0 |     16.7 |         1.2 |      87.5 |            0.0 |         202.8 |
| int8_dyn_d12_aligned_r128-224_f20__np+deadline | fixed60  |      5.3 |     15.8 |     16.8 |         1.8 |      86.2 |           11.5 |         203.3 |
| int8_dyn_d12_aligned_r128-224_f20__np          | fixed60  |      5.3 |     16.3 |     18.3 |         4.3 |      87.3 |           22.3 |         204.3 |
| int4_d12_aligned_r128-224_f20                  | fixed60  |     68.1 |    140.9 |    175.1 |        96.8 |      84.2 |           22.3 |         207.9 |
| int8_static_d12_aligned_r128-224_f20           | fixed60  |      5.6 |     17.4 |     19.4 |        21.3 |      88.2 |           21.8 |         223.2 |
| int8_dyn_d12_probe_r128-224_f10                | fixed60  |      5.2 |     15.9 |     17.8 |         2.8 |      92.0 |           19.7 |         228.2 |
| int8_dyn_d12_aligned_r224                      | fixed60  |     10.4 |     12.1 |     14.8 |         0.7 |      88.2 |            0.0 |         229.9 |
| int8_dyn_d12_aligned_r128-224_f20+deadline     | fixed60  |      5.2 |     15.9 |     17.6 |         2.8 |      87.2 |           22.3 |         232.5 |
| int8_dyn_d12_aligned_r160-224_f10              | fixed60  |      6.7 |     17.1 |     18.7 |         9.7 |      89.5 |            9.7 |         233.5 |
| int8_dyn_d12_aligned_r128-224_f20              | fixed60  |      5.2 |     16.2 |     18.8 |         4.0 |      87.3 |           23.0 |         242.6 |
| stream/int8_only (torch+MPS)                   | fixed60  |     10.8 |     12.3 |     14.8 |         0.5 |      84.2 |            0.0 |        1010.1 |
| stream/route_naive (torch+MPS)                 | fixed60  |     10.9 |     31.7 |     36.3 |        10.7 |      86.7 |            9.5 |        1010.7 |
| stream/fp16_only (torch+MPS)                   | fixed60  |     11.7 |     13.0 |     13.7 |         0.0 |      87.5 |          100.0 |        1012.1 |
