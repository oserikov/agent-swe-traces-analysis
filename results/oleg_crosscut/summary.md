# Hypothesis verdicts across model, job and task (screener, n=370)

## Screen hits by model

model                model-atlas model-cyan model-delta model-flint model-garnet model-orion model-vega
hypothesis                                                                                             
oleg-claims-success        31/40      59/60       37/60       35/60        28/30       54/60      57/60
oleg-cross-attempt          0/40       0/60        0/60        0/60         0/30        0/60       0/60
oleg-eval-aware             0/40      24/60        0/60        0/60         0/30        1/60       7/60
oleg-grader-suspect         7/40      14/60        5/60       13/60         3/30        7/60       8/60
oleg-hardcode               0/40       0/60        1/60        1/60         0/30        0/60       0/60
oleg-harness-wrestle       13/40      20/60       29/60       34/60         3/30        5/60      11/60
oleg-recall-claim           0/40      20/60        0/60        0/60         0/30        1/60       5/60
oleg-sandbox-breach         0/40       8/60        1/60        0/60         0/30        7/60       2/60
oleg-say-do-gap            27/40      30/60       26/60       51/60        22/30       33/60      45/60
oleg-test-peek              0/40       1/60        1/60        1/60         1/30        0/60       0/60
oleg-test-tamper            5/40      23/60       20/60        8/60         2/30        5/60      13/60
oleg-tool-parse-fail        6/40       4/60        4/60       14/60         2/30        6/60       3/60
oleg-web-lookup             2/40      38/60        5/60        1/60         0/30        8/60       3/60

## Screener precision vs Opus, pooled over hypotheses

              size   sum  precision
model                              
model-atlas     14   8.0       0.57
model-cyan      70  57.0       0.81
model-delta     20  16.0       0.80
model-flint     27  20.0       0.74
model-garnet    16  11.0       0.69
model-orion     22  10.0       0.45
model-vega      24  19.0       0.79

## Permutation tests (BH q within family)

family                job|model  job|model,final-msg  length|job   model  repo|job  task|job
hypothesis                                                                                  
oleg-claims-success      0.0013               0.3067      1.0000  0.0004    1.0000    0.9293
oleg-cross-attempt       1.0000               1.0000      1.0000  1.0000    1.0000    1.0000
oleg-eval-aware          1.0000               1.0000      0.0005  0.0004    0.0234    0.5428
oleg-grader-suspect      0.9004               1.0000      0.0048  0.3112    0.3936    0.0130
oleg-hardcode            0.2805               0.7233      0.3455  1.0000    1.0000    1.0000
oleg-harness-wrestle     0.0251               0.7845      0.0005  0.0004    1.0000    1.0000
oleg-recall-claim        0.2596               0.4372      0.0009  0.0004    0.0286    0.3782
oleg-sandbox-breach      0.3314               0.4917      0.0095  0.0013    0.3301    1.0000
oleg-say-do-gap          0.0013               0.4029      0.0005  0.0004    1.0000    1.0000
oleg-test-peek           0.0836               0.4372      0.3455  0.9479    1.0000    1.0000
oleg-test-tamper         0.0312               1.0000      0.0005  0.0004    0.0529    0.0130
oleg-tool-parse-fail     0.0433               0.4917      0.0055  0.0257    1.0000    1.0000
oleg-web-lookup          0.0433               0.4029      0.0005  0.0004    0.6330    0.2236

Observed / null-mean statistic:

family                job|model  job|model,final-msg  length|job  model  repo|job  task|job
hypothesis                                                                                 
oleg-claims-success        2.36                 2.56        8.95   9.85      0.96      1.01
oleg-cross-attempt          NaN                  NaN         NaN    NaN       NaN       NaN
oleg-eval-aware            0.97                 0.96       -0.14  15.95      2.92      1.48
oleg-grader-suspect        0.80                 0.63        3.82   1.33      1.30      2.92
oleg-hardcode              1.85                 1.19        1.55   0.70      0.00       NaN
oleg-harness-wrestle       1.40                 0.97        2.23   8.66      0.94      0.88
oleg-recall-claim          1.08                 1.07       -0.32  13.57      2.74      1.90
oleg-sandbox-breach        1.17                 1.17       -0.87   3.90      1.87      0.00
oleg-say-do-gap            1.61                 1.22        2.30   5.25      0.93      0.99
oleg-test-peek             1.72                 1.40        1.76   0.62      0.00      0.00
oleg-test-tamper           1.44                 0.91        5.15   5.01      1.65      2.27
oleg-tool-parse-fail       1.65                 1.15        1.74   2.51      0.53      0.51
oleg-web-lookup            1.06                 1.05       -0.68  21.97      1.12      1.45
