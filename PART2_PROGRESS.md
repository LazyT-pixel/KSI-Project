# Part 2 Progress Notes (study reference)

Last updated: Aug 4, 2026. Branch: `part2` (main is untouched, tagged `part1-submission`).

## What's actually done so far

One thing only: a **baseline model comparison** (`part2/model_building.py`). It reuses
the Part 1 pipeline in `KSI.py` exactly as-is (imports it as a module so cleaning/
encoding/train-test-split logic isn't duplicated), then trains 5 classifiers with
mostly-default settings and scores each on the 20% held-out test set (3,792 rows).

Nothing has been tuned yet. Nothing has been deployed yet. This is step 1 of several.

## The 5 models and why these 5

Logistic Regression, Decision Tree, Random Forest, Linear SVM, Neural Network (MLP).
This is my best guess at a reasonable, standard classification line-up for a course
project — **not yet confirmed against the actual Part 2 assignment sheet**. If the
sheet names specific algorithms or a specific count, swap accordingly before this is
final.

## Baseline results (untuned)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear SVM | 0.791 | 0.367 | 0.670 | 0.474 |
| Logistic Regression | 0.792 | 0.367 | 0.663 | 0.473 |
| Decision Tree | 0.896 | 0.625 | 0.661 | 0.642 |
| Neural Network (MLP) | 0.910 | 0.825 | 0.459 | 0.590 |
| Random Forest | 0.922 | 0.996 | 0.448 | 0.618 |

**How to read this, for the review tomorrow:**

- **Accuracy is misleading here.** The test set is ~86% "not fatal," so a model
  that just guessed "not fatal" every time would score ~86% accuracy while being
  useless. That's why Random Forest's 92% accuracy isn't actually the best model —
  look at recall instead.
- **Recall** = of all the *real* fatal cases, how many did the model catch. This is
  the number that matters most for this project — missing a real fatal case is a
  worse mistake than a false alarm.
- **Precision** = of everything the model *called* fatal, how many actually were.
  Random Forest's precision of 0.996 means it almost never falsely calls something
  fatal — but its low recall (0.448) means it misses over half of the real fatal
  cases. That's the opposite of what we want if the goal is catching risk early.
- **The trade-off in one sentence:** Linear SVM and Logistic Regression catch the
  most real fatal cases (~67%) but also raise a lot of false alarms (precision
  ~0.37). Decision Tree is the best all-around balance right now (F1 0.642,
  recall 0.661, and reasonable precision 0.625).
- `class_weight="balanced"` is applied everywhere it's supported — this is what's
  pushing recall up across the board, since fatal cases are only ~14% of the data
  and models would otherwise mostly ignore that class.

## What's next (in progress now)

Splitting hyperparameter tuning across the team so no one model person is a
bottleneck:

- **Aboud** — tune Logistic Regression + Linear SVM (`part2/tune_aboud.py`)
- **Ibrahim** — tune Decision Tree + Random Forest (`part2/tune_ibrahim.py`)
- **Aidan (me)** — Neural Network tuning, picking the final model, and deployment
  (Flask API)
- **Ali** — Report write-up (Executive Summary + Solution Overview sections)

Both tuning scripts are set up to run standalone (`python3 part2/tune_aboud.py` /
`part2/tune_ibrahim.py`, run from the main project folder) — they reuse the same
`KSI.py` pipeline via `sys.path`, run a small grid search (`GridSearchCV`, scored
on recall), and print best settings + test-set results in the same format as above,
so results are directly comparable to the baseline table.

## After tuning comes back

1. Compare all 5 tuned models, pick the strongest one (recall-weighted, not just
   accuracy) as the final model for deployment.
2. Model scoring/evaluation write-up (this is a separate graded component —
   worth documenting the accuracy-vs-recall reasoning above properly, not just
   picking a winner).
3. Deployment: pickle the final pipeline + model, build a small Flask API around
   it, test it with sample inputs.
4. Finish the report (Executive Summary + Solution Overview were left as
   placeholders in Part 1 — Ali's task now) and presentation prep.

## Open questions not yet resolved

- Part 1 grade/feedback: not received yet.
- Extension request: no reply yet; a submission drop box opened after the fact
  and was missed by ~2 days — unclear if that counts as late, worth a short
  follow-up email once there's a reply.
- Confirmed Part 2 due date: not yet confirmed from the course shell.
- Aboud's participation: given another chance, with an earlier internal deadline
  this time so there's buffer if he doesn't deliver again.
