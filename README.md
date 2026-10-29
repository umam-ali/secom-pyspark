# SECOM semiconductor yield with PySpark

A PySpark pipeline on the public UCI SECOM dataset: 1,567 production runs from a chip fab, 590 sensor readings per run, about 5% of readings missing, and a pass/fail label (104 fails, 6.6%). It cleans the data, flags outlier runs, checks for drift between an earlier and a later period, and fits two Spark ML classifiers that are scored on the later period only.

Dataset: https://archive.ics.uci.edu/dataset/179/secom

## Run it

You need JDK 17 and an unpacked Apache Spark 4.2.1 (the `bin-hadoop3` build). The two paths are at the top of `run.ps1`.

```powershell
python get_data.py     # downloads the dataset into data/raw (about 7 MB)
.\run.ps1              # about 90 seconds, writes results/metrics.json
```

PowerShell shows some Spark warnings in red (winutils, incubator modules, a pandas 3 notice). They are harmless here.

## What `secom_pipeline.py` does

1. Loads both files with Spark, turns NaN into null, joins sensors to labels and timestamps.
2. Splits by time: the earliest 70% of runs train, the latest 30% test. A random split would let the model see the future.
3. Data quality, using the training period only: drops sensors that are more than 40% missing (28) or constant (122), leaving 440.
4. Outliers: IQR fences at 3 x IQR per sensor, a count of outlier sensors per run, the fail rate by that count, then clipping to the fences.
5. Drift: how far each sensor's mean moved from the training period to the test period, in training standard deviations.
6. A Spark ML `Pipeline`: median imputer, vector assembler, standard scaler, univariate feature selector (top 40 by ANOVA F-score), then a classifier. Fails are about 1 in 14 runs, so training rows are weighted by inverse class frequency.
7. Scores a weighted logistic regression and a 200-tree random forest on the test period. AUC-ROC is also computed by hand from the scores and must agree with Spark's value.

Everything learned from data (drop list, fences, medians, scaler, selected sensors) is fitted on the training period only.

## Results (run on 29 Oct 2026, `results/metrics.json`)

Test period: 470 runs, 26 fails (5.5%).

- Logistic regression: AUC-ROC 0.591, AUC-PR 0.070. At the 0.5 threshold it caught 5 of 26 fails with 46 false alarms (precision 0.10, recall 0.19).
- Random forest: AUC-ROC 0.601, AUC-PR 0.067. At the 0.5 threshold it flagged no runs at all, so it caught 0 of 26 fails.
- A model with no skill scores AUC-ROC 0.5 and an AUC-PR equal to the fail rate, 0.055.
- 71 of 440 sensors moved by more than half a standard deviation between the two periods. The largest shift is about 1.1 standard deviations (f116).
- Runs with 21 or more outlier sensors failed 8 times in 42 (19%), against 6.6% overall. That is a small sample, so treat it as a lead and not a finding.

## How to read this

The signal in SECOM is weak, and both models are only slightly better than chance on the later period. That matches what is usually reported for this dataset. The drift check suggests part of the reason: the sensors do not look the same later in the run. The pipeline is the point here, not the score.

Not done, on purpose or for lack of time:

- No hyperparameter tuning and a single time split. Tuning on the test period would leak it, so any tuning needs a validation slice cut from the end of the training period.
- No threshold selection. The 0.5 threshold is the default, which is why the forest flags nothing. Choosing a threshold on a validation slice is the obvious next step.
- No per-sensor drift handling, only the check.

## Questions to be able to answer about this code

- Why is the split by time and not random?
- Why are the imputer, scaler and feature selector fitted on the training period only?
- Why median imputation, and why drop sensors above 40% missing?
- What does an IQR fence at 3 x IQR flag, and why skip sensors with IQR = 0?
- Why weight the classes, and what happens without weights?
- Why is AUC-PR compared against 0.055 and not 0.5?
- Why did the random forest flag nothing at 0.5, and what would you change?
- What is lazy evaluation in Spark, and what does `cache()` do here?
- Why is `monotonically_increasing_id()` safe for the join here, and when would it not be?
